"""SpacePersistence -- saves and loads the digital space state.

Handles serialization of the space topology and entity-created rooms/objects
to disk. The space is saved to JSON files under data/environment/. Entity
creations are tracked separately so the system can distinguish between seed
topology and entity-authored content.

File layout:
    data/environment/space.json           -- full space topology
    data/environment/entity_creations.json -- rooms/objects created by the entity
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional

from sanctuary.core.atomic_io import atomic_write_json
from sanctuary.environment.room import EnvironmentObject
from sanctuary.environment.space import DigitalSpace

logger = logging.getLogger(__name__)

DEFAULT_DATA_DIR = Path("data/environment")


class SpacePersistence:
    """Saves and loads the digital space state to/from disk.

    Tracks the full space topology and separately tracks entity-created
    rooms and objects with timestamps.
    """

    def __init__(self, data_dir: str | Path = DEFAULT_DATA_DIR) -> None:
        self._data_dir = Path(data_dir)
        self._space_path = self._data_dir / "space.json"
        self._creations_path = self._data_dir / "entity_creations.json"
        self._last_saved: Optional[datetime] = None
        self._modifications: list[dict] = []

    @property
    def data_dir(self) -> Path:
        """The directory where space data is stored."""
        return self._data_dir

    @property
    def last_saved(self) -> Optional[datetime]:
        """When the space was last saved."""
        return self._last_saved

    # ------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------

    def save(self, space: DigitalSpace) -> None:
        """Save the full space topology to disk.

        Creates the data directory if it does not exist.
        """
        self._data_dir.mkdir(parents=True, exist_ok=True)

        # Save full topology
        space.save(self._space_path)

        # Save entity creations separately
        self._save_entity_creations(space)

        self._last_saved = datetime.now()
        logger.info("Space persisted to %s", self._data_dir)

    def _save_entity_creations(self, space: DigitalSpace) -> None:
        """Save entity-created rooms and objects to a separate file."""
        creations: dict = {
            "saved_at": datetime.now().isoformat(),
            "rooms": {},
            "objects": {},
        }

        for room_id, room in space.rooms.items():
            if room.created_by == "entity":
                creations["rooms"][room_id] = room.model_dump(mode="json")

            # Check objects in every room (entity can add objects to system rooms)
            for obj in room.objects:
                if obj.created_by == "entity":
                    key = f"{room_id}/{obj.id}"
                    creations["objects"][key] = {
                        "room_id": room_id,
                        "object": obj.model_dump(mode="json"),
                    }

        atomic_write_json(self._creations_path, creations, default=str)

    # ------------------------------------------------------------------
    # Load
    # ------------------------------------------------------------------

    def load(self) -> Optional[DigitalSpace]:
        """Load the space from disk.

        Returns:
            The loaded DigitalSpace, or None if no saved state exists.
        """
        if not self._space_path.exists():
            logger.info("No saved space found at %s", self._space_path)
            return None

        try:
            space = DigitalSpace.load(self._space_path)
            logger.info(
                "Loaded space with %d rooms from %s",
                len(space.room_ids),
                self._space_path,
            )
            return space
        except Exception as e:
            logger.error("Failed to load space from %s: %s", self._space_path, e)
            return None

    def load_or_create_default(self) -> DigitalSpace:
        """Load the space from disk, or create the default seed topology.

        This is the primary entry point for initialization. If a saved
        space exists, it is loaded. Otherwise, the default seed topology
        is created and saved.
        """
        space = self.load()
        if space is not None:
            return space

        logger.info("Creating default space topology")
        space = DigitalSpace.create_default_space()
        self.save(space)
        return space

    # ------------------------------------------------------------------
    # Modification tracking
    # ------------------------------------------------------------------

    def record_modification(self, description: str) -> None:
        """Record a modification to the space for tracking purposes."""
        self._modifications.append(
            {
                "description": description,
                "timestamp": datetime.now().isoformat(),
            }
        )

    @property
    def pending_modifications(self) -> list[dict]:
        """Modifications recorded since the last save."""
        return list(self._modifications)

    def clear_modifications(self) -> None:
        """Clear the pending modifications list (called after save)."""
        self._modifications.clear()

    # ------------------------------------------------------------------
    # Entity creations info
    # ------------------------------------------------------------------

    def load_entity_creations(self) -> Optional[dict]:
        """Load the entity creations file.

        Returns:
            A dict with 'rooms' and 'objects' keys, or None if not found.
        """
        if not self._creations_path.exists():
            return None

        try:
            with open(self._creations_path) as f:
                return json.load(f)
        except Exception as e:
            logger.error(
                "Failed to load entity creations from %s: %s",
                self._creations_path,
                e,
            )
            return None
