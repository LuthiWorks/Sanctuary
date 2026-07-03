"""
Memory Backup System

Automated daily backups with retention policy for ChromaDB and memory files.

Author: Sanctuary Team
Date: January 2, 2026
"""

import asyncio
import logging
import shutil
import tarfile
import tempfile
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Optional, Dict, Any
import json

from sanctuary.core.atomic_io import atomic_write_json

from ..exceptions import MemoryError
from ..logging_config import get_logger, OperationContext

logger = get_logger(__name__)


class BackupManager:
    """
    Manages backups of memory system with retention policy.
    
    Features:
    - Daily automated backups
    - Timestamped backup directories
    - Compression (tar.gz)
    - 30-day retention policy
    - Automatic cleanup of old backups
    
    Example:
        manager = BackupManager(
            source_dir="memories",
            backup_dir="backups",
            retention_days=30
        )
        
        # Create backup
        backup_path = await manager.create_backup()
        
        # List available backups
        backups = manager.list_backups()
        
        # Restore from backup
        await manager.restore_backup(backup_path, target_dir="restored")
    """
    
    def __init__(
        self,
        source_dir: Path,
        backup_dir: Path,
        retention_days: int = 30,
        compress: bool = True,
        include_chroma: bool = True,
        min_keep: int = 3
    ):
        """
        Initialize backup manager.

        Args:
            source_dir: Source directory to backup (memory files)
            backup_dir: Directory to store backups
            retention_days: Number of days to keep backups
            compress: Whether to compress backups
            include_chroma: Whether to include ChromaDB directory
            min_keep: Newest backups always retained, regardless of age.
                Floors at 1 so cleanup can never delete the last backup.
        """
        self.source_dir = Path(source_dir)
        self.backup_dir = Path(backup_dir)
        self.retention_days = retention_days
        self.compress = compress
        self.include_chroma = include_chroma
        self.min_keep = max(1, min_keep)
        
        # Ensure backup directory exists
        self.backup_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(
            f"Backup manager initialized: source={source_dir}, "
            f"backup={backup_dir}, retention={retention_days} days"
        )
    
    async def create_backup(
        self,
        name_prefix: str = "sanctuary_memory",
        metadata: Optional[Dict[str, Any]] = None
    ) -> Path:
        """
        Create a backup of memory system.
        
        Args:
            name_prefix: Prefix for backup name
            metadata: Optional metadata to store with backup
        
        Returns:
            Path to created backup
        
        Raises:
            MemoryError: If backup creation fails
        """
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        backup_name = f"{name_prefix}_{timestamp}"
        
        with OperationContext(operation="backup_create", backup_name=backup_name):
            try:
                if self.compress:
                    backup_path = self.backup_dir / f"{backup_name}.tar.gz"
                    await self._create_compressed_backup(backup_path, backup_name, metadata)
                else:
                    backup_path = self.backup_dir / backup_name
                    await self._create_directory_backup(backup_path, metadata)
                
                logger.info(f"Backup created successfully: {backup_path}")
                return backup_path
            
            except Exception as e:
                logger.error(f"Failed to create backup: {e}")
                raise MemoryError(
                    "Backup creation failed",
                    operation="backup_create",
                    context={"error": str(e), "backup_name": backup_name}
                )
    
    async def _create_compressed_backup(
        self,
        backup_path: Path,
        backup_name: str,
        metadata: Optional[Dict[str, Any]]
    ):
        """Create compressed tar.gz backup."""
        # Use asyncio to run blocking operation in executor
        def _create_tar():
            with tarfile.open(backup_path, "w:gz") as tar:
                # Add source directory
                tar.add(
                    self.source_dir,
                    arcname=backup_name,
                    recursive=True
                )
                
                # Add metadata file if provided
                if metadata:
                    metadata_file = self.backup_dir / "metadata.json"
                    try:
                        atomic_write_json(metadata_file, {
                            **metadata,
                            "backup_name": backup_name,
                            "timestamp": datetime.utcnow().isoformat(),
                            "source_dir": str(self.source_dir)
                        })

                        tar.add(metadata_file, arcname=f"{backup_name}/metadata.json")
                    finally:
                        if metadata_file.exists():
                            metadata_file.unlink()
        
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, _create_tar)
        
        logger.debug(f"Compressed backup created: {backup_path}")
    
    async def _create_directory_backup(
        self,
        backup_path: Path,
        metadata: Optional[Dict[str, Any]]
    ):
        """Create uncompressed directory backup."""
        def _copy_dir():
            if backup_path.exists():
                shutil.rmtree(backup_path)
            shutil.copytree(self.source_dir, backup_path)
            
            # Add metadata
            if metadata:
                metadata_file = backup_path / "metadata.json"
                atomic_write_json(metadata_file, {
                    **metadata,
                    "backup_name": backup_path.name,
                    "timestamp": datetime.utcnow().isoformat(),
                    "source_dir": str(self.source_dir)
                })
        
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, _copy_dir)
        
        logger.debug(f"Directory backup created: {backup_path}")
    
    def list_backups(self) -> List[Dict[str, Any]]:
        """
        List all available backups.
        
        Returns:
            List of backup information dictionaries
        """
        backups = []
        
        for item in sorted(self.backup_dir.iterdir()):
            if item.is_file() and item.suffix == ".gz":
                # Compressed backup
                stat = item.stat()
                backups.append({
                    "name": item.stem.replace(".tar", ""),
                    "path": str(item),
                    "size_mb": stat.st_size / (1024 ** 2),
                    "created": datetime.fromtimestamp(stat.st_mtime),
                    "compressed": True
                })
            elif item.is_dir() and not item.name.startswith("."):
                # Directory backup
                size = sum(f.stat().st_size for f in item.rglob("*") if f.is_file())
                stat = item.stat()
                backups.append({
                    "name": item.name,
                    "path": str(item),
                    "size_mb": size / (1024 ** 2),
                    "created": datetime.fromtimestamp(stat.st_mtime),
                    "compressed": False
                })
        
        # Sort by creation time (newest first)
        backups.sort(key=lambda x: x["created"], reverse=True)
        
        return backups
    
    async def restore_backup(
        self,
        backup_path: Path,
        target_dir: Optional[Path] = None,
        dry_run: bool = False
    ) -> bool:
        """
        Restore from a backup.
        
        Args:
            backup_path: Path to backup to restore
            target_dir: Target directory (defaults to source_dir)
            dry_run: If True, validate only without restoring
        
        Returns:
            True if restore successful
        
        Raises:
            MemoryError: If restore fails
        """
        backup_path = Path(backup_path)
        target_dir = Path(target_dir) if target_dir else self.source_dir
        
        with OperationContext(
            operation="backup_restore",
            backup=str(backup_path),
            target=str(target_dir),
            dry_run=dry_run
        ):
            try:
                if not backup_path.exists():
                    raise MemoryError(
                        f"Backup not found: {backup_path}",
                        operation="backup_restore",
                        context={"backup_path": str(backup_path)}
                    )
                
                if dry_run:
                    logger.info(f"Dry run: would restore {backup_path} to {target_dir}")
                    return True
                
                # Create target directory
                target_dir.mkdir(parents=True, exist_ok=True)
                
                if backup_path.suffix == ".gz":
                    await self._restore_compressed_backup(backup_path, target_dir)
                else:
                    await self._restore_directory_backup(backup_path, target_dir)
                
                logger.info(f"Backup restored successfully: {backup_path} -> {target_dir}")
                return True
            
            except Exception as e:
                logger.error(f"Failed to restore backup: {e}")
                raise MemoryError(
                    "Backup restore failed",
                    operation="backup_restore",
                    context={"error": str(e), "backup_path": str(backup_path)}
                )
    
    async def _restore_compressed_backup(self, backup_path: Path, target_dir: Path):
        """Restore from compressed backup."""
        def _extract_tar():
            with tarfile.open(backup_path, "r:gz") as tar:
                # Extract to temporary directory first (using secure temp directory)
                temp_dir = Path(tempfile.mkdtemp(prefix="sanctuary_restore_"))
                try:
                    tar.extractall(temp_dir)
                    
                    # Move contents to target
                    extracted = list(temp_dir.iterdir())
                    if len(extracted) == 1 and extracted[0].is_dir():
                        # Backup has a root directory
                        source = extracted[0]
                    else:
                        source = temp_dir
                    
                    # Remove target if exists
                    if target_dir.exists():
                        shutil.rmtree(target_dir)
                    
                    # Move to target
                    shutil.move(str(source), str(target_dir))
                finally:
                    # Cleanup temp
                    if temp_dir.exists():
                        shutil.rmtree(temp_dir)
        
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, _extract_tar)
        
        logger.debug(f"Restored compressed backup: {backup_path}")
    
    async def _restore_directory_backup(self, backup_path: Path, target_dir: Path):
        """Restore from directory backup."""
        def _copy_dir():
            if target_dir.exists():
                shutil.rmtree(target_dir)
            shutil.copytree(backup_path, target_dir)
        
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, _copy_dir)
        
        logger.debug(f"Restored directory backup: {backup_path}")
    
    async def cleanup_old_backups(self) -> int:
        """
        Remove backups older than retention period.

        Always retains the newest ``min_keep`` backups regardless of age:
        a retention policy must never delete the only remaining copies.

        Returns:
            Number of backups removed
        """
        cutoff_date = datetime.utcnow() - timedelta(days=self.retention_days)
        removed_count = 0

        with OperationContext(operation="backup_cleanup", retention_days=self.retention_days):
            try:
                # list_backups() returns newest first
                backups = self.list_backups()[self.min_keep:]

                for backup in backups:
                    if backup["created"] < cutoff_date:
                        backup_path = Path(backup["path"])
                        
                        if backup_path.is_file():
                            backup_path.unlink()
                        elif backup_path.is_dir():
                            shutil.rmtree(backup_path)
                        
                        removed_count += 1
                        logger.info(
                            f"Removed old backup: {backup['name']} "
                            f"(created: {backup['created']})"
                        )
                
                logger.info(f"Cleanup complete: removed {removed_count} old backups")
                return removed_count
            
            except Exception as e:
                logger.error(f"Error during backup cleanup: {e}")
                raise MemoryError(
                    "Backup cleanup failed",
                    operation="backup_cleanup",
                    context={"error": str(e)}
                )
    
    async def schedule_daily_backup(self):
        """
        Start daily backup scheduler.
        
        Runs backup creation once per day and cleanup after each backup.
        """
        logger.info("Starting daily backup scheduler")
        
        while True:
            try:
                # Create backup
                await self.create_backup(
                    metadata={
                        "type": "scheduled",
                        "retention_days": self.retention_days
                    }
                )
                
                # Cleanup old backups
                await self.cleanup_old_backups()
                
                # Wait 24 hours
                await asyncio.sleep(24 * 60 * 60)
            
            except asyncio.CancelledError:
                logger.info("Daily backup scheduler stopped")
                break
            except Exception as e:
                logger.error(f"Error in backup scheduler: {e}")
                # Wait before retry
                await asyncio.sleep(60 * 60)  # 1 hour


# Global backup manager instance
_global_backup_manager: Optional[BackupManager] = None


def get_global_backup_manager(
    source_dir: Optional[Path] = None,
    backup_dir: Optional[Path] = None
) -> BackupManager:
    """
    Get or create global backup manager.
    
    Args:
        source_dir: Source directory (only used on first call)
        backup_dir: Backup directory (only used on first call)
    
    Returns:
        Global BackupManager instance
    """
    global _global_backup_manager
    if _global_backup_manager is None:
        if source_dir is None:
            source_dir = Path("memories")
        if backup_dir is None:
            backup_dir = Path("backups")
        
        _global_backup_manager = BackupManager(
            source_dir=source_dir,
            backup_dir=backup_dir
        )
    
    return _global_backup_manager
