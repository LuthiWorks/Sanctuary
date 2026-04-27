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
        # Create container with Python environment
        container = client.containers.run(
            "python:3.9-slim",
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
            remove=True  # Auto-remove container after execution
        )
        
        # Wait for container to finish with timeout
        try:
            output = await asyncio.wait_for(
                asyncio.to_thread(container.wait),
                timeout=timeout
            )
            
            # Get container logs
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
        except docker.errors.APIError:
            # Container may already be removed (auto-remove=True) or
            # never created — both are fine in cleanup.
            pass