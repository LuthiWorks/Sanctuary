"""
Audio Device: Microphone device implementation.

This module provides the MicrophoneDevice class, which implements DeviceProtocol
for audio input devices. It wraps the existing MicrophoneClient functionality
while conforming to the unified device abstraction layer.

Key Features:
- Uses sounddevice for cross-platform audio capture
- Emits DeviceDataPacket with modality="audio"
- Supports device enumeration and hot-plug
- Integrates with existing ASR pipeline (optional)
"""

from __future__ import annotations

import asyncio
import logging
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

# Optional sounddevice import
try:
    import sounddevice as sd
    HAS_SOUNDDEVICE = True
except ImportError:
    HAS_SOUNDDEVICE = False
    sd = None
    logger.warning("sounddevice not installed - microphone devices unavailable")


class MicrophoneDevice(DeviceProtocol):
    """
    Microphone device implementation using sounddevice.

    This class wraps the existing MicrophoneClient functionality into the
    DeviceProtocol interface. Audio data is captured and emitted as
    DeviceDataPacket objects via the data callback.

    Audio is captured as float32 normalized samples and can be converted
    to int16 for transmission or processing as needed.

    Example usage:
        # Enumerate available microphones
        mics = MicrophoneDevice.enumerate_devices()
        for mic in mics:
            print(f"{mic.device_id}: {mic.name}")

        # Connect and stream
        device = MicrophoneDevice(sample_rate=16000)
        device.set_data_callback(process_audio)
        await device.connect(mics[0].device_id)
        await device.start_streaming()
    """

    def __init__(
        self,
        sample_rate: int = 16000,
        channels: int = 1,
        chunk_duration: float = 0.1,
        dtype: str = "float32"
    ) -> None:
        """
        Initialize microphone device.

        Args:
            sample_rate: Audio sample rate in Hz (default: 16000)
            channels: Number of audio channels (default: 1, mono)
            chunk_duration: Duration of each audio chunk in seconds (default: 0.1)
            dtype: Audio data type (default: "float32")
        """
        super().__init__()

        self.sample_rate = sample_rate
        self.channels = channels
        self.chunk_duration = chunk_duration
        self.dtype = dtype
        self.chunk_size = int(sample_rate * chunk_duration)

        # Audio stream
        self._stream: Optional[sd.InputStream] = None
        self._audio_queue: asyncio.Queue = asyncio.Queue(maxsize=100)

        # Statistics
        self._frames_captured = 0
        self._frames_dropped = 0

    @classmethod
    def enumerate_devices(cls) -> List[DeviceInfo]:
        """
        Discover available audio input devices.

        Returns:
            List of DeviceInfo for all available microphones
        """
        if not HAS_SOUNDDEVICE:
            logger.warning("sounddevice not available")
            return []

        devices = []
        try:
            all_devices = sd.query_devices()
            default_input = sd.default.device[0]  # Default input device index

            for idx, device in enumerate(all_devices):
                # Only include input devices
                if device["max_input_channels"] > 0:
                    device_id = f"mic_{idx}"
                    is_default = (idx == default_input)

                    info = DeviceInfo(
                        device_id=device_id,
                        name=device["name"],
                        manufacturer=None,  # sounddevice doesn't provide this
                        capabilities=DeviceCapabilities(
                            device_type=DeviceType.MICROPHONE,
                            modality="audio",
                            sample_rate=int(device["default_samplerate"]),
                            channels=device["max_input_channels"],
                            bit_depth=16,
                        ),
                        is_default=is_default,
                    )
                    devices.append(info)

        except Exception as e:
            logger.error(f"Error enumerating audio devices: {e}")

        return devices

    async def connect(self, device_id: str) -> bool:
        """
        Connect to a specific microphone device.

        Args:
            device_id: Device ID from enumerate_devices() (e.g., "mic_0")

        Returns:
            True if connection succeeded
        """
        if not HAS_SOUNDDEVICE:
            self._emit_error(ImportError("sounddevice not installed"))
            return False

        if self.is_connected:
            logger.warning(f"Already connected to {self._device_id}")
            return True

        try:
            # Parse device index from ID
            if device_id.startswith("mic_"):
                device_index = int(device_id.split("_")[1])
            else:
                device_index = int(device_id)

            # Verify device exists
            devices = sd.query_devices()
            if device_index >= len(devices):
                raise ValueError(f"Device index {device_index} out of range")

            device = devices[device_index]
            if device["max_input_channels"] <= 0:
                raise ValueError(f"Device {device_index} is not an input device")

            # Store device info
            self._device_id = device_id
            self._device_info = DeviceInfo(
                device_id=device_id,
                name=device["name"],
                capabilities=DeviceCapabilities(
                    device_type=DeviceType.MICROPHONE,
                    modality="audio",
                    sample_rate=self.sample_rate,
                    channels=self.channels,
                    bit_depth=16,
                ),
            )
            self._device_index = device_index

            self._set_state(DeviceState.CONNECTED)
            logger.info(f"Connected to microphone: {device['name']}")
            return True

        except Exception as e:
            logger.error(f"Failed to connect to {device_id}: {e}")
            self._emit_error(e)
            self._set_state(DeviceState.ERROR)
            return False

    async def disconnect(self) -> None:
        """Disconnect from the microphone and clean up resources."""
        if self.is_streaming:
            await self.stop_streaming()

        self._device_id = None
        self._device_info = None
        self._set_state(DeviceState.DISCONNECTED)
        logger.info("Microphone disconnected")

    async def start_streaming(self) -> bool:
        """
        Start capturing audio from the microphone.

        Audio data is delivered via the data callback as DeviceDataPacket
        with modality="audio" and raw_data as float32 numpy array.

        Returns:
            True if streaming started successfully
        """
        if not self.is_connected:
            logger.error("Cannot start streaming - not connected")
            return False

        if self.is_streaming:
            logger.warning("Already streaming")
            return True

        try:
            self._set_state(DeviceState.STARTING)
            self._stop_event.clear()

            # Create audio stream with callback
            self._stream = sd.InputStream(
                samplerate=self.sample_rate,
                channels=self.channels,
                dtype=self.dtype,
                blocksize=self.chunk_size,
                device=self._device_index,
                callback=self._audio_callback,
            )

            # Start the stream
            self._stream.start()
            self._set_state(DeviceState.STREAMING)

            # Start async task to process audio queue
            self._streaming_task = supervise(
                asyncio.create_task(self._process_audio_queue()),
                name="audio_device_stream",
            )

            logger.info(f"Started streaming from {self._device_info.name}")
            return True

        except Exception as e:
            logger.error(f"Failed to start streaming: {e}")
            self._emit_error(e)
            self._set_state(DeviceState.ERROR)
            return False

    async def stop_streaming(self) -> None:
        """Stop capturing audio from the microphone."""
        if not self.is_streaming:
            return

        self._set_state(DeviceState.STOPPING)
        self._stop_event.set()

        # Stop the audio stream
        if self._stream:
            self._stream.stop()
            self._stream.close()
            self._stream = None

        # Cancel streaming task
        if self._streaming_task:
            self._streaming_task.cancel()
            try:
                await self._streaming_task
            except asyncio.CancelledError:
                pass
            self._streaming_task = None

        # Clear the queue
        while not self._audio_queue.empty():
            try:
                self._audio_queue.get_nowait()
            except asyncio.QueueEmpty:
                break

        self._set_state(DeviceState.CONNECTED)
        logger.info("Stopped streaming")

    def _audio_callback(
        self,
        indata: NDArray[np.float32],
        frames: int,
        time_info: Any,
        status: sd.CallbackFlags
    ) -> None:
        """
        Callback for sounddevice audio capture.

        This runs in the audio thread, so we just queue the data
        for processing in the async task.
        """
        if status:
            logger.warning(f"Audio callback status: {status}")

        self._frames_captured += frames

        # Queue audio data (non-blocking)
        try:
            self._audio_queue.put_nowait(indata.copy())
        except asyncio.QueueFull:
            self._frames_dropped += frames
            logger.debug("Audio queue full, dropping frame")

    async def _process_audio_queue(self) -> None:
        """Async task to process queued audio and emit data packets."""
        while not self._stop_event.is_set():
            try:
                # Get audio from queue with timeout
                audio_data = await asyncio.wait_for(
                    self._audio_queue.get(),
                    timeout=0.5
                )

                # Emit as DeviceDataPacket
                self._emit_data(
                    modality="audio",
                    raw_data=audio_data,
                    metadata={
                        "sample_rate": self.sample_rate,
                        "channels": self.channels,
                        "dtype": self.dtype,
                        "frames": len(audio_data),
                    }
                )

            except asyncio.TimeoutError:
                continue
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error processing audio: {e}")

    async def configure(self, **settings) -> bool:
        """
        Configure device settings.

        Supported settings:
            sample_rate: Audio sample rate in Hz
            channels: Number of channels
            chunk_duration: Duration of each chunk in seconds

        Args:
            **settings: Settings to configure

        Returns:
            True if configuration succeeded
        """
        was_streaming = self.is_streaming
        if was_streaming:
            await self.stop_streaming()

        if "sample_rate" in settings:
            self.sample_rate = settings["sample_rate"]
            self.chunk_size = int(self.sample_rate * self.chunk_duration)

        if "channels" in settings:
            self.channels = settings["channels"]

        if "chunk_duration" in settings:
            self.chunk_duration = settings["chunk_duration"]
            self.chunk_size = int(self.sample_rate * self.chunk_duration)

        if was_streaming:
            await self.start_streaming()

        return True

    def get_stats(self) -> Dict[str, Any]:
        """Get device statistics."""
        base_stats = super().get_stats()
        return {
            **base_stats,
            "sample_rate": self.sample_rate,
            "channels": self.channels,
            "chunk_size": self.chunk_size,
            "frames_captured": self._frames_captured,
            "frames_dropped": self._frames_dropped,
            "queue_size": self._audio_queue.qsize(),
        }


# Register the device class
DeviceProtocol.register_device_class(DeviceType.MICROPHONE, MicrophoneDevice)


__all__ = [
    "MicrophoneDevice",
    "HAS_SOUNDDEVICE",
]
