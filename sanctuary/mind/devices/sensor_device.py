"""
Sensor Device: Generic sensor device implementations.

This module provides sensor device implementations for various input types:
- Environmental sensors (temperature, humidity, pressure)
- Motion sensors (accelerometer, gyroscope)
- Serial-connected sensors (Arduino, ESP32, Raspberry Pi Pico)

Key Features:
- Generic SensorDevice base class with polling loop
- SerialSensorDevice for UART-connected microcontrollers
- Structured SensorReading output
- Hot-plug support for serial devices
"""

from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto
from typing import Any, Dict, List, Optional, Union

from sanctuary.core.supervision import supervise

from .protocol import (
    DeviceCapabilities,
    DeviceInfo,
    DeviceProtocol,
    DeviceState,
    DeviceType,
)

logger = logging.getLogger(__name__)

# Optional pyserial import
try:
    import serial
    import serial.tools.list_ports
    HAS_SERIAL = True
except ImportError:
    HAS_SERIAL = False
    serial = None
    logger.warning("pyserial not installed - serial sensors unavailable")


class SensorType(Enum):
    """Types of sensor readings."""
    TEMPERATURE = auto()
    HUMIDITY = auto()
    PRESSURE = auto()
    LIGHT = auto()
    SOUND_LEVEL = auto()
    ACCELEROMETER = auto()
    GYROSCOPE = auto()
    MAGNETOMETER = auto()
    DISTANCE = auto()
    TOUCH = auto()
    FORCE = auto()
    CUSTOM = auto()


@dataclass
class SensorReading:
    """
    A single reading from a sensor.

    Attributes:
        sensor_type: Type of sensor
        value: The reading value (float or dict for multi-axis)
        unit: Unit of measurement
        timestamp: When the reading was taken
        confidence: Confidence in the reading (0.0-1.0)
        metadata: Additional context
    """
    sensor_type: SensorType
    value: Union[float, Dict[str, float]]
    unit: str
    timestamp: datetime = field(default_factory=datetime.now)
    confidence: float = 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "sensor_type": self.sensor_type.name,
            "value": self.value,
            "unit": self.unit,
            "timestamp": self.timestamp.isoformat(),
            "confidence": self.confidence,
            "metadata": self.metadata,
        }


class SensorDevice(DeviceProtocol):
    """
    Base class for sensor devices.

    This class provides a polling-based interface for sensors that don't
    support streaming or callbacks. Subclasses implement the _read_sensor()
    method to get readings.

    Readings are emitted as DeviceDataPacket with modality="sensor" and
    raw_data containing a SensorReading object (as dict).

    Example usage:
        class TemperatureSensor(SensorDevice):
            def _read_sensor(self) -> Optional[SensorReading]:
                # Read from hardware
                temp = self.hardware.read_temperature()
                return SensorReading(
                    sensor_type=SensorType.TEMPERATURE,
                    value=temp,
                    unit="celsius"
                )
    """

    def __init__(
        self,
        poll_interval: float = 1.0,
        sensor_types: Optional[List[SensorType]] = None
    ) -> None:
        """
        Initialize sensor device.

        Args:
            poll_interval: Seconds between readings (default: 1.0)
            sensor_types: List of sensor types this device provides
        """
        super().__init__()

        self.poll_interval = poll_interval
        self.sensor_types = sensor_types or [SensorType.CUSTOM]

        # Statistics
        self._readings_count: int = 0
        self._errors_count: int = 0

    @classmethod
    def enumerate_devices(cls) -> List[DeviceInfo]:
        """
        Discover available sensor devices.

        Override in subclass for specific sensor discovery.

        Returns:
            List of DeviceInfo for discovered sensors
        """
        return []

    async def connect(self, device_id: str) -> bool:
        """
        Connect to a sensor device.

        Override in subclass for specific connection logic.

        Args:
            device_id: Device identifier

        Returns:
            True if connection succeeded
        """
        self._device_id = device_id
        self._set_state(DeviceState.CONNECTED)
        return True

    async def disconnect(self) -> None:
        """Disconnect from the sensor."""
        if self.is_streaming:
            await self.stop_streaming()
        self._device_id = None
        self._set_state(DeviceState.DISCONNECTED)

    async def start_streaming(self) -> bool:
        """Start polling the sensor for readings."""
        if not self.is_connected:
            logger.error("Cannot start streaming - not connected")
            return False

        if self.is_streaming:
            return True

        self._set_state(DeviceState.STARTING)
        self._stop_event.clear()
        self._streaming_task = supervise(
            asyncio.create_task(self._poll_loop()), name="sensor_device_stream"
        )
        self._set_state(DeviceState.STREAMING)
        return True

    async def stop_streaming(self) -> None:
        """Stop polling the sensor."""
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

    async def _poll_loop(self) -> None:
        """Polling loop for sensor readings."""
        while not self._stop_event.is_set():
            try:
                reading = await self._read_sensor_async()

                if reading:
                    self._readings_count += 1

                    # Emit as DeviceDataPacket
                    self._emit_data(
                        modality="sensor",
                        raw_data=reading.to_dict(),
                        metadata={
                            "sensor_type": reading.sensor_type.name,
                            "unit": reading.unit,
                        }
                    )

            except asyncio.CancelledError:
                break
            except Exception as e:
                self._errors_count += 1
                logger.error(f"Error reading sensor: {e}")

            await asyncio.sleep(self.poll_interval)

    async def _read_sensor_async(self) -> Optional[SensorReading]:
        """
        Async wrapper for reading sensor.

        Runs _read_sensor() in executor if it's blocking.
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self._read_sensor)

    def _read_sensor(self) -> Optional[SensorReading]:
        """
        Read current sensor value.

        Override in subclass to implement actual sensor reading.

        Returns:
            SensorReading or None if read failed
        """
        raise NotImplementedError("Subclass must implement _read_sensor()")

    def get_stats(self) -> Dict[str, Any]:
        """Get device statistics."""
        base_stats = super().get_stats()
        return {
            **base_stats,
            "poll_interval": self.poll_interval,
            "sensor_types": [s.name for s in self.sensor_types],
            "readings_count": self._readings_count,
            "errors_count": self._errors_count,
        }


class SerialSensorDevice(SensorDevice):
    """
    Sensor device connected via serial port (UART).

    This class supports microcontrollers like Arduino, ESP32, and
    Raspberry Pi Pico that send sensor data over serial/USB.

    Expected serial protocol:
    - JSON lines: {"type": "temperature", "value": 23.5, "unit": "celsius"}
    - Simple CSV: temperature,23.5,celsius

    Example usage:
        # Enumerate available serial ports
        ports = SerialSensorDevice.enumerate_devices()

        # Connect to Arduino
        device = SerialSensorDevice(baudrate=115200)
        await device.connect("COM3")  # or "/dev/ttyUSB0" on Linux
        await device.start_streaming()
    """

    def __init__(
        self,
        baudrate: int = 9600,
        timeout: float = 1.0,
        poll_interval: float = 0.1
    ) -> None:
        """
        Initialize serial sensor device.

        Args:
            baudrate: Serial baud rate
            timeout: Read timeout in seconds
            poll_interval: Seconds between reads
        """
        super().__init__(poll_interval=poll_interval)

        self.baudrate = baudrate
        self.timeout = timeout

        self._serial: Optional[serial.Serial] = None
        self._buffer: str = ""

    @classmethod
    def enumerate_devices(cls) -> List[DeviceInfo]:
        """
        Discover available serial ports.

        Returns:
            List of DeviceInfo for available serial ports
        """
        if not HAS_SERIAL:
            logger.warning("pyserial not available")
            return []

        devices = []
        try:
            ports = serial.tools.list_ports.comports()

            for port in ports:
                device_id = port.device
                info = DeviceInfo(
                    device_id=device_id,
                    name=port.description or device_id,
                    manufacturer=port.manufacturer,
                    capabilities=DeviceCapabilities(
                        device_type=DeviceType.SENSOR,
                        modality="sensor",
                        extra={
                            "vid": port.vid,
                            "pid": port.pid,
                            "serial_number": port.serial_number,
                        },
                    ),
                    is_default=False,
                )
                devices.append(info)

        except Exception as e:
            logger.error(f"Error enumerating serial ports: {e}")

        return devices

    async def connect(self, device_id: str) -> bool:
        """
        Connect to a serial port.

        Args:
            device_id: Serial port name (e.g., "COM3" or "/dev/ttyUSB0")

        Returns:
            True if connection succeeded
        """
        if not HAS_SERIAL:
            self._emit_error(ImportError("pyserial not installed"))
            return False

        if self.is_connected:
            logger.warning(f"Already connected to {self._device_id}")
            return True

        try:
            self._serial = serial.Serial(
                port=device_id,
                baudrate=self.baudrate,
                timeout=self.timeout,
            )

            # Wait for device to be ready
            await asyncio.sleep(0.5)

            self._device_id = device_id
            self._device_info = DeviceInfo(
                device_id=device_id,
                name=device_id,
                capabilities=DeviceCapabilities(
                    device_type=DeviceType.SENSOR,
                    modality="sensor",
                    extra={"baudrate": self.baudrate},
                ),
            )

            self._set_state(DeviceState.CONNECTED)
            logger.info(f"Connected to serial port: {device_id} @ {self.baudrate} baud")
            return True

        except Exception as e:
            logger.error(f"Failed to connect to {device_id}: {e}")
            self._emit_error(e)
            self._set_state(DeviceState.ERROR)
            return False

    async def disconnect(self) -> None:
        """Disconnect from the serial port."""
        if self.is_streaming:
            await self.stop_streaming()

        if self._serial:
            self._serial.close()
            self._serial = None

        self._device_id = None
        self._device_info = None
        self._set_state(DeviceState.DISCONNECTED)
        logger.info("Serial port disconnected")

    def _read_sensor(self) -> Optional[SensorReading]:
        """
        Read and parse data from serial port.

        Supports JSON lines or simple CSV format.

        Returns:
            SensorReading or None if no data available
        """
        if not self._serial or not self._serial.is_open:
            return None

        try:
            # Read available data
            if self._serial.in_waiting > 0:
                data = self._serial.readline().decode("utf-8", errors="ignore").strip()

                if not data:
                    return None

                # Try JSON parsing first
                try:
                    parsed = json.loads(data)
                    return self._parse_json_reading(parsed)
                except json.JSONDecodeError:
                    pass

                # Try CSV parsing
                parts = data.split(",")
                if len(parts) >= 2:
                    return self._parse_csv_reading(parts)

                logger.debug(f"Unparseable serial data: {data}")

        except Exception as e:
            logger.error(f"Error reading serial data: {e}")

        return None

    def _parse_json_reading(self, data: Dict[str, Any]) -> Optional[SensorReading]:
        """Parse JSON sensor data."""
        try:
            # Map string type to SensorType enum
            type_str = data.get("type", "custom").upper()
            try:
                sensor_type = SensorType[type_str]
            except KeyError:
                sensor_type = SensorType.CUSTOM

            return SensorReading(
                sensor_type=sensor_type,
                value=data.get("value", 0.0),
                unit=data.get("unit", ""),
                confidence=data.get("confidence", 1.0),
                metadata=data.get("metadata", {}),
            )

        except Exception as e:
            logger.error(f"Error parsing JSON reading: {e}")
            return None

    def _parse_csv_reading(self, parts: List[str]) -> Optional[SensorReading]:
        """Parse CSV sensor data (type,value,unit)."""
        try:
            type_str = parts[0].strip().upper()
            try:
                sensor_type = SensorType[type_str]
            except KeyError:
                sensor_type = SensorType.CUSTOM

            value = float(parts[1].strip())
            unit = parts[2].strip() if len(parts) > 2 else ""

            return SensorReading(
                sensor_type=sensor_type,
                value=value,
                unit=unit,
            )

        except Exception as e:
            logger.debug(f"Error parsing CSV reading: {e}")
            return None

    async def send_command(self, command: str) -> bool:
        """
        Send a command to the serial device.

        Args:
            command: Command string to send

        Returns:
            True if send succeeded
        """
        if not self._serial or not self._serial.is_open:
            logger.error("Serial port not connected")
            return False

        try:
            self._serial.write(f"{command}\n".encode("utf-8"))
            return True
        except Exception as e:
            logger.error(f"Error sending command: {e}")
            return False

    def get_stats(self) -> Dict[str, Any]:
        """Get device statistics."""
        base_stats = super().get_stats()
        return {
            **base_stats,
            "baudrate": self.baudrate,
            "in_waiting": self._serial.in_waiting if self._serial else 0,
        }


# Register the device class
DeviceProtocol.register_device_class(DeviceType.SENSOR, SerialSensorDevice)


__all__ = [
    "SensorType",
    "SensorReading",
    "SensorDevice",
    "SerialSensorDevice",
    "HAS_SERIAL",
]
