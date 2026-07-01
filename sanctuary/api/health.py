"""Health check HTTP server for container health monitoring.

Phase 3.2: Containerization.

Provides lightweight HTTP endpoints for Docker health checks and
monitoring tools. Runs alongside the cognitive cycle without
interfering with it.

Endpoints:
    GET /health  — Simple liveness check (200 OK / 503 unhealthy)
    GET /status  — Detailed system status (JSON)
    GET /metrics — Resource usage metrics (JSON)

Usage::

    from sanctuary.api.health import HealthServer

    server = HealthServer(runner=runner, resource_monitor=monitor)
    await server.start(port=8000)
    # ... later ...
    await server.stop()
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from typing import Any, Optional

logger = logging.getLogger(__name__)

# Default port for health server
DEFAULT_HEALTH_PORT = 8000
# Secure by default: loopback only. /status and /metrics expose detailed
# internal state (goals, authority levels, introspection) with no auth, so the
# server must not be reachable from the network unless an operator explicitly
# opts in (e.g. SANCTUARY_HEALTH_HOST=0.0.0.0). The in-container Docker
# HEALTHCHECK curls localhost, so loopback binding does not affect liveness.
DEFAULT_HEALTH_HOST = "127.0.0.1"


class HealthServer:
    """Lightweight HTTP health check server.

    Uses raw asyncio HTTP handling to avoid heavy framework dependencies
    in the health check path. Quart is available but we want the health
    server to be as lightweight as possible — it must respond even when
    the cognitive system is under heavy load.
    """

    def __init__(
        self,
        runner: Any = None,
        resource_monitor: Any = None,
        host: str = DEFAULT_HEALTH_HOST,
        port: int = DEFAULT_HEALTH_PORT,
    ):
        self._runner = runner
        self._resource_monitor = resource_monitor
        self._host = host
        self._port = port
        self._server: Optional[asyncio.Server] = None
        self._start_time = time.monotonic()

    async def start(self) -> None:
        """Start the health check server."""
        self._start_time = time.monotonic()
        self._server = await asyncio.start_server(
            self._handle_connection, self._host, self._port
        )
        logger.info("Health server listening on %s:%d", self._host, self._port)

    async def stop(self) -> None:
        """Stop the health check server."""
        if self._server:
            self._server.close()
            await self._server.wait_closed()
            self._server = None
            logger.info("Health server stopped")

    @property
    def running(self) -> bool:
        return self._server is not None and self._server.is_serving()

    # ------------------------------------------------------------------
    # Connection handler
    # ------------------------------------------------------------------

    async def _handle_connection(
        self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter
    ) -> None:
        """Handle an incoming HTTP connection."""
        try:
            # Read the request line (with timeout to prevent hangs)
            data = await asyncio.wait_for(reader.readline(), timeout=5.0)
            request_line = data.decode("utf-8", errors="replace").strip()

            # Parse method and path
            parts = request_line.split()
            if len(parts) < 2:
                await self._send_response(writer, 400, {"error": "Bad request"})
                return

            method, path = parts[0], parts[1]

            # Drain remaining headers (we don't need them)
            while True:
                line = await asyncio.wait_for(reader.readline(), timeout=5.0)
                if line in (b"\r\n", b"\n", b""):
                    break

            # Route
            if method == "GET" and path == "/health":
                await self._handle_health(writer)
            elif method == "GET" and path == "/status":
                await self._handle_status(writer)
            elif method == "GET" and path == "/metrics":
                await self._handle_metrics(writer)
            else:
                await self._send_response(writer, 404, {"error": "Not found"})

        except asyncio.TimeoutError:
            await self._send_response(writer, 408, {"error": "Request timeout"})
        except Exception as exc:
            logger.warning("Health server error: %s", exc)
            try:
                await self._send_response(writer, 500, {"error": "Internal error"})
            except Exception as send_err:
                logger.debug("Failed to send error response: %s", send_err)
        finally:
            try:
                writer.close()
                await writer.wait_closed()
            except Exception as close_err:
                logger.debug("Connection cleanup failed: %s", close_err)

    # ------------------------------------------------------------------
    # Endpoint handlers
    # ------------------------------------------------------------------

    async def _handle_health(self, writer: asyncio.StreamWriter) -> None:
        """GET /health — simple liveness/readiness check.

        Returns 200 if the cognitive cycle is running, 503 otherwise.
        Docker HEALTHCHECK should use this endpoint.
        """
        healthy = self._check_healthy()
        status_code = 200 if healthy else 503

        uptime = time.monotonic() - self._start_time

        body = {
            "status": "healthy" if healthy else "unhealthy",
            "uptime_seconds": round(uptime, 1),
        }

        if self._runner:
            body["cycle_count"] = self._runner.cycle_count
            body["booted"] = getattr(self._runner, "_booted", False)

        await self._send_response(writer, status_code, body)

    async def _handle_status(self, writer: asyncio.StreamWriter) -> None:
        """GET /status — detailed system status."""
        status: dict[str, Any] = {
            "uptime_seconds": round(time.monotonic() - self._start_time, 1),
        }

        if self._runner:
            try:
                runner_status = self._runner.get_status()
                # Convert non-serializable values
                sanitized = {}
                for k, v in runner_status.items():
                    try:
                        json.dumps(v)
                        sanitized[k] = v
                    except (TypeError, ValueError):
                        sanitized[k] = str(v)
                status["runner"] = sanitized
            except Exception as exc:
                status["runner_error"] = str(exc)

        if self._resource_monitor:
            try:
                status["resources"] = self._resource_monitor.snapshot()
            except Exception as exc:
                status["resources_error"] = str(exc)

        await self._send_response(writer, 200, status)

    async def _handle_metrics(self, writer: asyncio.StreamWriter) -> None:
        """GET /metrics — resource usage metrics."""
        metrics: dict[str, Any] = {}

        if self._resource_monitor:
            try:
                metrics = self._resource_monitor.snapshot()
            except Exception as exc:
                metrics["error"] = str(exc)

        if self._runner:
            metrics["cycle_count"] = self._runner.cycle_count

        await self._send_response(writer, 200, metrics)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _check_healthy(self) -> bool:
        """Determine if the system is healthy.

        Healthy means:
        - The runner exists and has booted
        - The cognitive cycle is running OR we're still in the boot window
        """
        if not self._runner:
            # No runner attached — health server is standalone (testing mode)
            return True

        booted = getattr(self._runner, "_booted", False)
        if not booted:
            # Still booting — healthy if within startup grace period (120s)
            uptime = time.monotonic() - self._start_time
            return uptime < 120.0

        # Booted — check if cycle is running
        running = getattr(self._runner, "running", False)
        return running

    @staticmethod
    async def _send_response(
        writer: asyncio.StreamWriter,
        status_code: int,
        body: dict,
    ) -> None:
        """Send an HTTP JSON response."""
        status_text = {
            200: "OK",
            400: "Bad Request",
            404: "Not Found",
            408: "Request Timeout",
            500: "Internal Server Error",
            503: "Service Unavailable",
        }.get(status_code, "Unknown")

        body_bytes = json.dumps(body, default=str).encode("utf-8")

        response = (
            f"HTTP/1.1 {status_code} {status_text}\r\n"
            f"Content-Type: application/json\r\n"
            f"Content-Length: {len(body_bytes)}\r\n"
            f"Connection: close\r\n"
            f"\r\n"
        ).encode("utf-8") + body_bytes

        writer.write(response)
        await writer.drain()
