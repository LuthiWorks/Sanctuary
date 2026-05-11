"""
Security module for sandboxed Python code execution
"""
import logging
import asyncio
import docker
from typing import Optional

logger = logging.getLogger(__name__)

async def sandbox_python_execution(code: str, timeout: int = 30) -> str:
    """
    Execute Python code in a secure Docker container
    
    Args:
        code: Python code to execute
        timeout: Maximum execution time in seconds
    
    Returns:
        String containing execution output or error message
    """
    client = docker.from_env()
    
    try:
        # Create container with Python environment.
        # python:3.12-slim — bumped from 3.9-slim (EOL October 2025) so the
        # security sandbox runs on a maintained base. The interpreter is
        # only used to exec untrusted code in a network-less, read-only,
        # capability-dropped jail, so newer-language-features aren't the
        # concern — patched runtime is.
        container = client.containers.run(
            "python:3.12-slim",
            [
                "python",
                "-c",
                code
            ],
            detach=True,
            mem_limit="100m",  # Limited memory
            cpu_period=100000,  # CPU quota limiting
            cpu_quota=50000,    # 50% CPU max
            network_mode="none",  # No network access
            cap_drop=["ALL"],  # Drop all capabilities
            security_opt=["no-new-privileges"],  # Prevent privilege escalation
            read_only=True,  # Make filesystem read-only
            tmpfs={"/tmp": "size=64m,exec,nodev,nosuid"},  # Temporary filesystem for /tmp
            # remove=False — we want to read logs() *after* wait() returns.
            # With auto-remove the container would race-delete by the time
            # we call logs(), intermittently raising docker.errors.NotFound.
            # The explicit container.remove() in finally below handles cleanup.
            remove=False,
        )

        # Wait for container to finish with timeout
        try:
            output = await asyncio.wait_for(
                asyncio.to_thread(container.wait),
                timeout=timeout
            )

            # Get container logs while it still exists (we own removal).
            logs = container.logs().decode('utf-8')

            if output['StatusCode'] == 0:
                return logs
            else:
                return f"Error (exit code {output['StatusCode']}):\n{logs}"

        except asyncio.TimeoutError:
            container.kill()
            return "Execution timed out"

    except docker.errors.DockerException as e:
        logger.error(f"Sandbox execution error: {e}")
        return f"Error in sandbox execution: {str(e)}"
    finally:
        try:
            container.remove(force=True)
        except (docker.errors.APIError, docker.errors.NotFound, NameError):
            # Container may never have been created (NameError) or already
            # be gone (NotFound) — both fine in cleanup.
            pass