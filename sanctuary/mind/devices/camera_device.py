"""
Camera Device: Video capture device implementation.

This module provides the CameraDevice class, which implements DeviceProtocol
for video input devices. It uses OpenCV (cv2) with the DirectShow backend
for Windows compatibility.

Key Features:
- Uses OpenCV for cross-platform video capture
- DirectShow backend (CAP_DSHOW) for Windows
- Emits DeviceDataPacket with modality="image"
- Frame data as numpy arrays (BGR format)
- Configurable resolution and frame rate
"""

from __future__ import annotations

import asyncio
import logging
import platform
from typing import Any, Dict, List, Optional

from sanctuary.core.supervision import supervise

import numpy as np
from numpy.typing import NDArray

from .protocol import (
    DeviceCapabilities,
    DeviceInfo,
    DeviceProtocol,
    DeviceState,
    DeviceType,
)

logger = logging.getLogger(__name__)

# Optional OpenCV import
try:
    import cv2
    HAS_OPENCV = True
except ImportError:
    HAS_OPENCV = False
    cv2 = None
    logger.warning("opencv-python not installed - camera devices unavailable")


# Use DirectShow on Windows for better compatibility
IS_WINDOWS = platform.system() == "Windows"
DEFAULT_BACKEND = cv2.CAP_DSHOW if (HAS_OPENCV and IS_WINDOWS) else (cv2.CAP_ANY if HAS_OPENCV else 0)


class CameraDevice(DeviceProtocol):
    """
    Camera device implementation using OpenCV.

    This class provides video capture through OpenCV's VideoCapture interface.
    On Windows, it uses the DirectShow backend (CAP_DSHOW) for improved
    compatibility and performance.

    Frame data is emitted as BGR numpy arrays via the data callback.
    The callback receives DeviceDataPacket with modality="image".

    Example usage:
        # Enumerate available cameras
        cameras = CameraDevice.enumerate_devices()
        for cam in cameras:
            print(f"{cam.device_id}: {cam.name}")

        # Connect and stream
        device = CameraDevice(resolution=(640, 480), fps=30)
        device.set_data_callback(process_frame)
        await device.connect(cameras[0].device_id)
        await device.start_streaming()
    """

    # Maximum device index to probe during enumeration
    MAX_DEVICE_INDEX = 10

    def __init__(
        self,
        resolution: tuple[int, int] = (640, 480),
        fps: float = 30.0,
        backend: Optional[int] = None
    ) -> None:
        """
        Initialize camera device.

        Args:
            resolution: Desired resolution as (width, height)
            fps: Desired frames per second
            backend: OpenCV backend (default: CAP_DSHOW on Windows)
        """
        super().__init__()

        self.resolution = resolution
        self.fps = fps
        self.backend = backend if backend is not None else DEFAULT_BACKEND

        # Capture state
        self._capture: Optional[cv2.VideoCapture] = None
        self._device_index: int = 0

        # Statistics
        self._frames_captured: int = 0
        self._frames_dropped: int = 0
        self._last_frame_time: float = 0

    @classmethod
    def enumerate_devices(cls) -> List[DeviceInfo]:
        """
        Discover available camera devices.

        OpenCV doesn't provide a native device enumeration API, so we probe
        device indices 0 through MAX_DEVICE_INDEX to find available cameras.

        Returns:
            List of DeviceInfo for all available cameras
        """
        if not HAS_OPENCV:
            logger.warning("opencv-python not available")
            return []

        devices = []
        backend = DEFAULT_BACKEND

        for idx in range(cls.MAX_DEVICE_INDEX):
            try:
                # Try to open the device
                cap = cv2.VideoCapture(idx, backend)

                if cap.isOpened():
                    # Get device properties
                    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

                    # Try to get device name (not always available)
                    # OpenCV doesn't expose device names directly
                    name = f"Camera {idx}"

                    device_id = f"cam_{idx}"
                    info = DeviceInfo(
                        device_id=device_id,
                        name=name,
                        manufacturer=None,
                        capabilities=DeviceCapabilities(
                            device_type=DeviceType.CAMERA,
                            modality="image",
                            resolution=(width, height),
                            channels=3,  # BGR
                            extra={"fps": fps, "backend": backend},
                        ),
                        is_default=(idx == 0),
                    )
                    devices.append(info)

                    cap.release()

            except Exception as e:
                logger.debug(f"Error probing camera {idx}: {e}")

        logger.debug(f"Found {len(devices)} camera(s)")
        return devices

    async def connect(self, device_id: str) -> bool:
        """
        Connect to a specific camera device.

        Args:
            device_id: Device ID from enumerate_devices() (e.g., "cam_0")

        Returns:
            True if connection succeeded
        """
        if not HAS_OPENCV:
            self._emit_error(ImportError("opencv-python not installed"))
            return False

        if self.is_connected:
            logger.warning(f"Already connected to {self._device_id}")
            return True

        try:
            # Parse device index from ID
            if device_id.startswith("cam_"):
                device_index = int(device_id.split("_")[1])
            else:
                device_index = int(device_id)

            # Open camera
            self._capture = cv2.VideoCapture(device_index, self.backend)

            if not self._capture.isOpened():
                raise RuntimeError(f"Failed to open camera {device_index}")

            # Configure resolution
            self._capture.set(cv2.CAP_PROP_FRAME_WIDTH, self.resolution[0])
            self._capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self.resolution[1])
            self._capture.set(cv2.CAP_PROP_FPS, self.fps)

            # Get actual properties (may differ from requested)
            actual_width = int(self._capture.get(cv2.CAP_PROP_FRAME_WIDTH))
            actual_height = int(self._capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
            actual_fps = self._capture.get(cv2.CAP_PROP_FPS) or self.fps

            self._device_id = device_id
            self._device_index = device_index
            self._device_info = DeviceInfo(
                device_id=device_id,
                name=f"Camera {device_index}",
                capabilities=DeviceCapabilities(
                    device_type=DeviceType.CAMERA,
                    modality="image",
                    resolution=(actual_width, actual_height),
                    channels=3,
                    extra={"fps": actual_fps, "backend": self.backend},
                ),
            )

            self._set_state(DeviceState.CONNECTED)
            logger.info(f"Connected to camera {device_index} ({actual_width}x{actual_height} @ {actual_fps}fps)")
            return True

        except Exception as e:
            logger.error(f"Failed to connect to {device_id}: {e}")
            self._emit_error(e)
            self._set_state(DeviceState.ERROR)
            if self._capture:
                self._capture.release()
                self._capture = None
            return False

    async def disconnect(self) -> None:
        """Disconnect from the camera and release resources."""
        if self.is_streaming:
            await self.stop_streaming()

        if self._capture:
            self._capture.release()
            self._capture = None

        self._device_id = None
        self._device_info = None
        self._set_state(DeviceState.DISCONNECTED)
        logger.info("Camera disconnected")

    async def start_streaming(self) -> bool:
        """
        Start capturing frames from the camera.

        Frames are delivered via the data callback as DeviceDataPacket
        with modality="image" and raw_data as BGR numpy array.

        Returns:
            True if streaming started successfully
        """
        if not self.is_connected:
            logger.error("Cannot start streaming - not connected")
            return False

        if self.is_streaming:
            logger.warning("Already streaming")
            return True

        if not self._capture or not self._capture.isOpened():
            logger.error("Camera not opened")
            return False

        try:
            self._set_state(DeviceState.STARTING)
            self._stop_event.clear()

            # Start capture task
            self._streaming_task = supervise(
                asyncio.create_task(self._capture_loop()),
                name="camera_device_stream",
            )

            self._set_state(DeviceState.STREAMING)
            logger.info(f"Started streaming from {self._device_info.name}")
            return True

        except Exception as e:
            logger.error(f"Failed to start streaming: {e}")
            self._emit_error(e)
            self._set_state(DeviceState.ERROR)
            return False

    async def stop_streaming(self) -> None:
        """Stop capturing frames from the camera."""
        if not self.is_streaming:
            return

        self._set_state(DeviceState.STOPPING)
        self._stop_event.set()

        if self._streaming_task:
            self._streaming_task.cancel()
            try:
                await self._streaming_task
            except asyncio.CancelledError:
                pass
            self._streaming_task = None

        self._set_state(DeviceState.CONNECTED)
        logger.info("Stopped streaming")

    async def _capture_loop(self) -> None:
        """Async loop to capture and emit frames."""
        frame_interval = 1.0 / self.fps
        loop = asyncio.get_event_loop()

        while not self._stop_event.is_set():
            try:
                # Capture frame in thread pool to avoid blocking
                frame = await loop.run_in_executor(None, self._read_frame)

                if frame is not None:
                    self._frames_captured += 1

                    # Get actual resolution
                    height, width = frame.shape[:2]

                    # Emit as DeviceDataPacket
                    self._emit_data(
                        modality="image",
                        raw_data=frame,
                        metadata={
                            "width": width,
                            "height": height,
                            "channels": 3,
                            "format": "BGR",
                            "dtype": str(frame.dtype),
                        }
                    )
                else:
                    self._frames_dropped += 1
                    logger.debug("Frame capture returned None")

                # Rate limiting
                await asyncio.sleep(frame_interval)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error capturing frame: {e}")
                await asyncio.sleep(frame_interval)

    def _read_frame(self) -> Optional[NDArray[np.uint8]]:
        """Read a single frame from the camera (runs in thread pool)."""
        if not self._capture or not self._capture.isOpened():
            return None

        ret, frame = self._capture.read()
        if ret:
            return frame
        return None

    async def configure(self, **settings) -> bool:
        """
        Configure device settings.

        Supported settings:
            resolution: Tuple of (width, height)
            fps: Frames per second

        Args:
            **settings: Settings to configure

        Returns:
            True if configuration succeeded
        """
        was_streaming = self.is_streaming
        if was_streaming:
            await self.stop_streaming()

        if "resolution" in settings:
            self.resolution = settings["resolution"]
            if self._capture:
                self._capture.set(cv2.CAP_PROP_FRAME_WIDTH, self.resolution[0])
                self._capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self.resolution[1])

        if "fps" in settings:
            self.fps = settings["fps"]
            if self._capture:
                self._capture.set(cv2.CAP_PROP_FPS, self.fps)

        if was_streaming:
            await self.start_streaming()

        return True

    def get_stats(self) -> Dict[str, Any]:
        """Get device statistics."""
        base_stats = super().get_stats()
        return {
            **base_stats,
            "resolution": self.resolution,
            "fps": self.fps,
            "backend": self.backend,
            "frames_captured": self._frames_captured,
            "frames_dropped": self._frames_dropped,
        }

    def capture_single_frame(self) -> Optional[NDArray[np.uint8]]:
        """
        Capture a single frame without streaming.

        Useful for snapshot functionality.

        Returns:
            BGR numpy array or None if capture failed
        """
        if not self._capture or not self._capture.isOpened():
            logger.error("Camera not connected")
            return None

        return self._read_frame()


# Register the device class
if HAS_OPENCV:
    DeviceProtocol.register_device_class(DeviceType.CAMERA, CameraDevice)


__all__ = [
    "CameraDevice",
    "HAS_OPENCV",
]
