"""
Legacy Data Migration Script

Migrates existing journal entries from data/journal/*.json
to the new Sovereign Memory Architecture.

Usage:
    python migrate_legacy_data.py [--dry-run] [--journal-dir PATH] [--memory-dir PATH]

Author: Sanctuary Emergence Team
Date: November 23, 2025
"""

import asyncio
import argparse
import logging
import sys
from pathlib import Path
from datetime import datetime

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from mind.memory_manager import MemoryManager, MemoryConfig
from mind.legacy_parser import migrate_legacy_journals

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(f'migration_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
    ]
)
logger = logging.getLogger(__name__)


async def main():
    """Main migration function."""
    parser = argparse.ArgumentParser(
        description='Migrate legacy journal entries to Sovereign Memory Architecture'
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Parse entries but do not commit to new system (for testing)'
    )
    parser.add_argument(
        '--journal-dir',
        type=Path,
        default=Path('../data/journal'),
        help='Path to legacy journal directory (default: ../data/journal)'
    )
    parser.add_argument(
        '--memory-dir',
        type=Path,
        default=Path('../data/memories'),
        help='Path to new memory storage directory (default: ../data/memories)'
    )
    parser.add_argument(
        '--chroma-dir',
        type=Path,
        default=Path('../model_cache/chroma_db'),
        help='Path to ChromaDB storage (default: ../model_cache/chroma_db)'
    )
    parser.add_argument(
        '--backup',
        action='store_true',
        help='Create backup of existing memory directory before migration'
    )
    
    args = parser.parse_args()
    
    # Validate journal directory exists
    if not args.journal_dir.exists():
        logger.error(f"Journal directory not found: {args.journal_dir}")
        return 1
    
    # Count journal files
    json_files = list(args.journal_dir.glob("*.json"))
    json_files = [f for f in json_files if f.stem != "journal_index"]
    logger.info(f"Found {len(json_files)} journal files in {args.journal_dir}")
    
    if len(json_files) == 0:
        logger.warning("No journal files found to migrate")
        return 0
    
    # Backup existing memory directory if requested
    if args.backup and args.memory_dir.exists():
        import shutil
        backup_dir = args.memory_dir.parent / f"memories_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        logger.info(f"Creating backup: {backup_dir}")
        shutil.copytree(args.memory_dir, backup_dir)
        logger.info(f"Backup created successfully")
    
    # Initialize memory manager (skip if dry run)
    if not args.dry_run:
        logger.info(f"Initializing memory manager...")
        logger.info(f"  Memory directory: {args.memory_dir}")
        logger.info(f"  ChromaDB directory: {args.chroma_dir}")
        
        memory_manager = MemoryManager(
            base_dir=args.memory_dir,
            chroma_dir=args.chroma_dir,
        )
        
        # Get initial statistics
        initial_stats = await memory_manager.get_statistics()
        logger.info(f"Initial state:")
        logger.info(f"  Journal entries: {initial_stats.get('journal_entries', 0)}")
        logger.info(f"  Fact entries: {initial_stats.get('fact_entries', 0)}")
        logger.info(f"  Pivotal memories: {initial_stats.get('pivotal_memories', 0)}")
    else:
        memory_manager = None
        logger.info("DRY RUN MODE - No data will be committed")
    
    # Confirm migration
    if not args.dry_run:
        logger.info("\n" + "=" * 70)
        logger.info("MIGRATION READY")
        logger.info("=" * 70)
        logger.info(f"Source: {args.journal_dir}")
        logger.info(f"Destination: {args.memory_dir}")
        logger.info(f"Files to process: {len(json_files)}")
        logger.info("=" * 70)
        
        response = input("\nProceed with migration? (yes/no): ").strip().lower()
        if response != "yes":
            logger.info("Migration cancelled by user")
            return 0
    
    # Run migration
    logger.info("\nStarting migration...")
    start_time = datetime.now()
    
    try:
        total, successful, failed = await migrate_legacy_journals(
            journal_dir=args.journal_dir,
            memory_manager=memory_manager,
            dry_run=args.dry_run
        )
        
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        
        # Print summary
        logger.info("\n" + "=" * 70)
        logger.info("MIGRATION COMPLETE")
        logger.info("=" * 70)
        logger.info(f"Total entries processed: {total}")
        logger.info(f"Successfully migrated: {successful}")
        logger.info(f"Failed migrations: {failed}")
        logger.info(f"Success rate: {(successful/total*100) if total > 0 else 0:.1f}%")
        logger.info(f"Duration: {duration:.2f} seconds")
        logger.info(f"Throughput: {total/duration:.1f} entries/second")
        
        if not args.dry_run:
            # Get final statistics
            final_stats = await memory_manager.get_statistics()
            logger.info(f"\nFinal state:")
            logger.info(f"  Journal entries: {final_stats.get('journal_entries', 0)}")
            logger.info(f"  Fact entries: {final_stats.get('fact_entries', 0)}")
            logger.info(f"  Pivotal memories: {final_stats.get('pivotal_memories', 0)}")
            
            # Check manifest
            manifest = await memory_manager.load_manifest()
            if manifest:
                logger.info(f"\nManifest updated:")
                logger.info(f"  Core values: {len(manifest.core_values)}")
                logger.info(f"  Pivotal memories: {len(manifest.pivotal_memories)}")
                logger.info(f"  Current directives: {len(manifest.current_directives)}")
        
        logger.info("=" * 70)
        
        if failed > 0:
            logger.warning(f"\n⚠️  {failed} entries failed to migrate - check logs for details")
            return 1
        else:
            logger.info("\n✅ All entries migrated successfully!")
            return 0
            
    except Exception as e:
        logger.error(f"\n❌ Migration failed with error: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
