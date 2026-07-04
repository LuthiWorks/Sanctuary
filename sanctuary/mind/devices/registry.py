"""
Device Registry: Central management of all peripheral devices.

This module provides the DeviceRegistry class, which serves as the central hub for:
- Registering device implementation classes
- Discovering available devices across all types
- Managing device connections and lifecycles
- Routing device data to the cognitive pipeline (InputQueue)
- Monitoring for hot-plug events (device connect/disconnect)

The registry integrates with the cognitive architecture by forwarding all device
data to the InputQueue, which then feeds the perception subsystem.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Set, Type

from sanctuary.core.supervision import supervise

from .protocol import (
    DeviceDataPacket,
    DeviceInfo,
    DeviceProtocol,
    DeviceState,
    DeviceType,
)

logger = logging.getLogger(__name__)


@dataclass
class DeviceEntry:
    """
    Tracks a connected device in the registry.

    Attributes:
        device: The device instance
        info: Device information
        connected_at: When the device was connected
        auto_stream: Whether to auto-start streaming
    """
    device: DeviceProtocol
    info: DeviceInfo
    connected_at: datetime = field(default_factory=datetime.now)
    auto_stream: bool = False


# Type for InputQueue-compatible add_input method
InputQueueCallback = Callable[[Any, str, str, Optional[Dict[str, Any]]], Any]


class DeviceRegistry:
    """
    Central registry for all peripheral devices.

    The DeviceRegistry acts as a facade for device management, providing:

    1. Device Class Registration: Register implementations for each DeviceType
    2. Device Discovery: Enumerate all available devices across all types
    3. Connection Management: Connect/disconnect devices with lifecycle tracking
    4. Data Routing: Forward device data to the cognitive pipeline
    5. Hot-Plug Monitoring: Detect device connect/disconnect events

    Integration with Cognitive Architecture:
    - All device data flows through DeviceRegistry -> InputQueue -> Perception
    - Device data is wrapped in DeviceDataPacket before routing
    - The registry handles async coordination between devices and the cognitive loop

    Example usage:
        registry = DeviceRegistry()
        registry.register_device_class(DeviceType.MICROPHONE, MicrophoneDevice)
        registry.register_device_class(DeviceType.CAMERA, CameraDevice)

        # Set up data routing to InputQueue
        registry.set_input_queue(input_queue)

        # Discover and connect devices
        all_devices = registry.enumerate_all_devices()
        await registry.connect_device(DeviceType.MICROPHONE, "mic_0", auto_stream=True)

        # Start hot-plug monitoring
        await registry.start_hot_plug_monitoring()
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        """
        Initialize the device registry.

        Args:
            config: Optional configuration dict with keys:
                - hot_plug_interval: Seconds between hot-plug checks (default: 5.0)
                - auto_connect_defaults: Auto-connect default devices (default: False)
                - enabled_types: List of enabled DeviceType names (default: all)
        """
        self.config = config or {}

        # Device class registry (DeviceType -> implementation class)
        self._device_classes: Dict[DeviceType, Type[DeviceProtocol]] = {}

        # Connected devices (device_id -> DeviceEntry)
        self._connected_devices: Dict[str, DeviceEntry] = {}

        # Known devices from last enumeration (for hot-plug detection)
        self._known_devices: Dict[DeviceType, Set[str]] = {}

        # Input queue callback for routing data
        self._input_queue_callback: Optional[InputQueueCallback] = None

        # Hot-plug monitoring
        self._hot_plug_task: Optional[asyncio.Task] = None
        self._hot_plug_interval: float = self.config.get("hot_plug_interval", 5.0)
        self._hot_plug_running: bool = False

        # Event callbacks
        self._on_device_connected: Optional[Callable[[DeviceInfo], None]] = None
        self._on_device_disconnected: Optional[Callable[[str], None]] = None

        # Statistics
        self._stats = {
            "total_packets_routed": 0,
            "packets_by_modality": {},
            "connection_count": 0,
            "disconnection_count": 0,
        }

        logger.info("DeviceRegistry initialized")

    # ========== Device Class Registration ==========

    def register_device_class(
        self,
        device_type: DeviceType,
        device_class: Type[DeviceProtocol]
    ) -> None:
        """
        Register a device implementation for a device type.

        Args:
            device_type: The type of device
            device_class: Class implementing DeviceProtocol
        """
        self._device_classes[device_type] = device_class
        DeviceProtocol.register_device_class(device_type, device_class)
        logger.info(f"Registered {device_class.__name__} for {device_type.name}")

    def get_device_class(self, device_type: DeviceType) -> Optional[Type[DeviceProtocol]]:
        """Get the registered class for a device type."""
        return self._device_classes.get(device_type)

    def get_registered_types(self) -> List[DeviceType]:
        """Get list of device types that have registered implementations."""
        return list(self._device_classes.keys())

    # ========== Input Queue Integration ==========

    def set_input_queue(self, input_queue: Any) -> None:
        """
        Set the InputQueue to route device data to.

        Args:
            input_queue: InputQueue instance with add_input() method
        """
        if hasattr(input_queue, "add_input"):
            self._input_queue_callback = input_queue.add_input
            logger.info("InputQueue connected to DeviceRegistry")
        else:
            raise ValueError("input_queue must have an add_input() method")

    def set_input_callback(self, callback: InputQueueCallback) -> None:
        """
        Set a custom callback for receiving device data.

        This is an alternative to set_input_queue() for custom data handling.

        Args:
            callback: Async function(data, modality, source, metadata)
        """
        self._input_queue_callback = callback

    def _on_device_data(self, packet: DeviceDataPacket) -> None:
        """
        Handle incoming data from a device.

        Routes data to the InputQueue for processing by the cognitive pipeline.
        """
        self._stats["total_packets_routed"] += 1

        modality = packet.modality
        if modality not in self._stats["packets_by_modality"]:
            self._stats["packets_by_modality"][modality] = 0
        self._stats["packets_by_modality"][modality] += 1

        if self._input_queue_callback:
            # Route to InputQueue with device metadata
            metadata = {
                "device_id": packet.device_id,
                "timestamp": packet.timestamp.isoformat(),
                "sequence_number": packet.sequence_number,
                **packet.metadata
            }

            # Handle async callback
            try:
                result = self._input_queue_callback(
                    packet.raw_data,
                    packet.modality,
                    f"device:{packet.device_id}",
                    metadata
                )
                # If it's a coroutine, schedule it
                if asyncio.iscoroutine(result):
                    asyncio.create_task(result)
            except Exception as e:
                logger.error(f"Failed to route device data: {e}")

    # ========== Device Discovery ==========

    def enumerate_all_devices(self) -> Dict[DeviceType, List[DeviceInfo]]:
        """
        Discover all available devices across all registered types.

        Returns:
            Dict mapping DeviceType to list of discovered DeviceInfo
        """
        results: Dict[DeviceType, List[DeviceInfo]] = {}

        for device_type, device_class in self._device_classes.items():
            try:
                devices = device_class.enumerate_devices()
                results[device_type] = devices

                # Update known devices for hot-plug detection
                self._known_devices[device_type] = {d.device_id for d in devices}

                logger.debug(f"Found {len(devices)} {device_type.name} devices")
            except Exception as e:
                logger.error(f"Error enumerating {device_type.name} devices: {e}")
                results[device_type] = []

        return results

    def enumerate_devices_by_type(self, device_type: DeviceType) -> List[DeviceInfo]:
        """
        Discover available devices of a specific type.

        Args:
            device_type: Type of devices to enumerate

        Returns:
            List of discovered DeviceInfo
        """
        device_class = self._device_classes.get(device_type)
        if not device_class:
            logger.warning(f"No implementation registered for {device_type.name}")
            return []

        try:
            devices = device_class.enumerate_devices()
            self._known_devices[device_type] = {d.device_id for d in devices}
            return devices
        except Exception as e:
            logger.error(f"Error enumerating {device_type.name} devices: {e}")
            return []

    # ========== Device Connection Management ==========

    async def connect_device(
        self,
        device_type: DeviceType,
        device_id: str,
        auto_stream: bool = False
    ) -> bool:
        """
        Connect to a device and optionally start streaming.

        Args:
            device_type: Type of device to connect
            device_id: ID of specific device
            auto_stream: Start streaming immediately after connection

        Returns:
            True if connection (and optional streaming) succeeded
        """
        device_class = self._device_classes.get(device_type)
        if not device_class:
            logger.error(f"No implementation for {device_type.name}")
            return False

        if device_id in self._connected_devices:
            logger.warning(f"Device {device_id} already connected")
            return True

        try:
            # Create device instance
            device = device_class()

            # Set up data callback
            device.set_data_callback(self._on_device_data)

            # Connect
            success = await device.connect(device_id)
            if not success:
                logger.error(f"Failed to connect to {device_id}")
                return False

            # Get device info
            info = device.device_info
            if not info:
                # Create minimal info if device didn't provide it
                info = DeviceInfo(
                    device_id=device_id,
                    name=device_id,
                    capabilities=device_class.enumerate_devices()[0].capabilities
                    if device_class.enumerate_devices() else None
                )

            # Track in registry
            entry = DeviceEntry(
                device=device,
                info=info,
                auto_stream=auto_stream
            )
            self._connected_devices[device_id] = entry
            self._stats["connection_count"] += 1

            logger.info(f"Connected to device: {info.name} ({device_id})")

            # Notify callback
            if self._on_device_connected:
                self._on_device_connected(info)

            # Auto-start streaming if requested
            if auto_stream:
                stream_success = await device.start_streaming()
                if not stream_success:
                    logger.warning(f"Connected but failed to start streaming: {device_id}")

            return True

        except Exception as e:
            logger.error(f"Error connecting to {device_id}: {e}")
            return False

    async def disconnect_device(self, device_id: str) -> bool:
        """
        Disconnect a device and clean up resources.

        Args:
            device_id: ID of device to disconnect

        Returns:
            True if disconnection succeeded
        """
        entry = self._connected_devices.get(device_id)
        if not entry:
            logger.warning(f"Device {device_id} not found in registry")
            return False

        try:
            # Stop streaming if active
            if entry.device.is_streaming:
                await entry.device.stop_streaming()

            # Disconnect
            await entry.device.disconnect()

            # Remove from registry
            del self._connected_devices[device_id]
            self._stats["disconnection_count"] += 1

            logger.info(f"Disconnected device: {device_id}")

            # Notify callback
            if self._on_device_disconnected:
                self._on_device_disconnected(device_id)

            return True

        except Exception as e:
            logger.error(f"Error disconnecting {device_id}: {e}")
            return False

    async def disconnect_all_devices(self) -> None:
        """Disconnect all connected devices."""
        device_ids = list(self._connected_devices.keys())
        for device_id in device_ids:
            await self.disconnect_device(device_id)

    # ========== Streaming Control ==========

    async def start_streaming(self, device_id: str) -> bool:
        """Start streaming on a connected device."""
        entry = self._connected_devices.get(device_id)
        if not entry:
            logger.error(f"Device {device_id} not connected")
            return False

        return await entry.device.start_streaming()

    async def stop_streaming(self, device_id: str) -> bool:
        """Stop streaming on a connected device."""
        entry = self._connected_devices.get(device_id)
        if not entry:
            logger.error(f"Device {device_id} not connected")
            return False

        await entry.device.stop_streaming()
        return True

    # ========== Hot-Plug Monitoring ==========

    async def start_hot_plug_monitoring(self) -> None:
        """Start background task to detect device connect/disconnect events."""
        if self._hot_plug_running:
            logger.warning("Hot-plug monitoring already running")
            return

        self._hot_plug_running = True
        self._hot_plug_task = supervise(
            asyncio.create_task(self._hot_plug_loop()), name="device_hot_plug"
        )
        logger.info(f"Hot-plug monitoring started (interval: {self._hot_plug_interval}s)")

    async def stop_hot_plug_monitoring(self) -> None:
        """Stop hot-plug monitoring."""
        self._hot_plug_running = False
        if self._hot_plug_task:
            self._hot_plug_task.cancel()
            try:
                await self._hot_plug_task
            except asyncio.CancelledError:
                pass
            self._hot_plug_task = None
        logger.info("Hot-plug monitoring stopped")

    async def _hot_plug_loop(self) -> None:
        """Background loop to detect device changes."""
        while self._hot_plug_running:
            try:
                await asyncio.sleep(self._hot_plug_interval)
                await self._check_device_changes()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Hot-plug monitoring error: {e}")

    async def _check_device_changes(self) -> None:
        """Check for newly connected or disconnected devices."""
        for device_type, device_class in self._device_classes.items():
            try:
                current_devices = {d.device_id for d in device_class.enumerate_devices()}
                known_devices = self._known_devices.get(device_type, set())

                # Detect new devices
                new_devices = current_devices - known_devices
                for device_id in new_devices:
                    logger.info(f"Hot-plug: New {device_type.name} device detected: {device_id}")
                    # Could auto-connect here based on config

                # Detect removed devices
                removed_devices = known_devices - current_devices
                for device_id in removed_devices:
                    logger.info(f"Hot-plug: {device_type.name} device removed: {device_id}")
                    # Disconnect if it was connected
                    if device_id in self._connected_devices:
                        await self.disconnect_device(device_id)

                # Update known devices
                self._known_devices[device_type] = current_devices

            except Exception as e:
                logger.error(f"Error checking {device_type.name} devices: {e}")

    # ========== Event Callbacks ==========

    def set_on_device_connected(self, callback: Callable[[DeviceInfo], None]) -> None:
        """Set callback for device connection events."""
        self._on_device_connected = callback

    def set_on_device_disconnected(self, callback: Callable[[str], None]) -> None:
        """Set callback for device disconnection events."""
        self._on_device_disconnected = callback

    # ========== Status and Statistics ==========

    def get_connected_devices(self) -> Dict[str, DeviceInfo]:
        """Get info about all connected devices."""
        return {
            device_id: entry.info
            for device_id, entry in self._connected_devices.items()
        }

    def get_device(self, device_id: str) -> Optional[DeviceProtocol]:
        """Get a connected device instance by ID."""
        entry = self._connected_devices.get(device_id)
        return entry.device if entry else None

    def get_stats(self) -> Dict[str, Any]:
        """Get registry statistics."""
        return {
            **self._stats,
            "registered_types": [t.name for t in self._device_classes.keys()],
            "connected_device_count": len(self._connected_devices),
            "connected_devices": list(self._connected_devices.keys()),
            "hot_plug_running": self._hot_plug_running,
        }


__all__ = [
    "DeviceRegistry",
    "DeviceEntry",
]
