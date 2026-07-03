"""DigitalSpace -- the complete navigable world.

Manages the room graph: rooms and their connections. Provides methods for
adding rooms, connecting them, querying the topology, and serializing the
entire space to/from JSON for persistence.

The space ships with a default seed topology -- five rooms that form the
entity's initial inner landscape. The entity can extend this topology by
creating new rooms at runtime.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional

from sanctuary.core.atomic_io import atomic_write_json
from sanctuary.environment.room import Room, EnvironmentObject

logger = logging.getLogger(__name__)


class DigitalSpace:
    """The complete navigable world -- a graph of rooms.

    Rooms are stored by id. Connections between rooms are encoded in each
    room's `exits` dict (direction -> room_id). The space can be serialized
    to JSON and restored.
    """

    def __init__(self) -> None:
        self._rooms: dict[str, Room] = {}

    # ------------------------------------------------------------------
    # Room management
    # ------------------------------------------------------------------

    def add_room(self, room: Room) -> None:
        """Register a room in the space."""
        if room.id in self._rooms:
            logger.warning("Room %s already exists, overwriting", room.id)
        self._rooms[room.id] = room

    def get_room(self, room_id: str) -> Optional[Room]:
        """Get a room by id. Returns None if not found."""
        return self._rooms.get(room_id)

    def remove_room(self, room_id: str) -> Optional[Room]:
        """Remove a room and all connections to it. Returns the removed room."""
        room = self._rooms.pop(room_id, None)
        if room is None:
            return None

        # Remove all exits pointing to this room from other rooms
        for other_room in self._rooms.values():
            to_remove = [
                direction
                for direction, target in other_room.exits.items()
                if target == room_id
            ]
            for direction in to_remove:
                del other_room.exits[direction]

        return room

    @property
    def rooms(self) -> dict[str, Room]:
        """All rooms in the space."""
        return dict(self._rooms)

    @property
    def room_ids(self) -> list[str]:
        """All room ids."""
        return list(self._rooms.keys())

    # ------------------------------------------------------------------
    # Connections
    # ------------------------------------------------------------------

    def connect(
        self,
        room_a_id: str,
        direction: str,
        room_b_id: str,
        bidirectional: bool = True,
    ) -> bool:
        """Connect two rooms via a directional exit.

        Args:
            room_a_id: Source room.
            direction: Exit direction from room_a (e.g., "north").
            room_b_id: Destination room.
            bidirectional: If True, also create the reverse connection
                using the opposite direction.

        Returns:
            True if the connection was made, False if a room was not found.
        """
        room_a = self._rooms.get(room_a_id)
        room_b = self._rooms.get(room_b_id)

        if room_a is None or room_b is None:
            logger.warning(
                "Cannot connect %s -> %s: room not found",
                room_a_id,
                room_b_id,
            )
            return False

        room_a.exits[direction] = room_b_id

        if bidirectional:
            reverse = _opposite_direction(direction)
            room_b.exits[reverse] = room_a_id

        return True

    def get_exits(self, room_id: str) -> dict[str, str]:
        """Get available exits from a room. Returns direction -> room_id."""
        room = self._rooms.get(room_id)
        if room is None:
            return {}
        return dict(room.exits)

    def get_neighbors(self, room_id: str) -> list[Room]:
        """Get rooms adjacent to the given room."""
        room = self._rooms.get(room_id)
        if room is None:
            return []
        neighbors = []
        for target_id in room.exits.values():
            target = self._rooms.get(target_id)
            if target is not None:
                neighbors.append(target)
        return neighbors

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    def to_dict(self) -> dict:
        """Serialize the entire space to a dict."""
        return {
            "rooms": {
                room_id: room.model_dump(mode="json")
                for room_id, room in self._rooms.items()
            }
        }

    @classmethod
    def from_dict(cls, data: dict) -> DigitalSpace:
        """Deserialize a space from a dict."""
        space = cls()
        for room_id, room_data in data.get("rooms", {}).items():
            room = Room.model_validate(room_data)
            space._rooms[room_id] = room
        return space

    def save(self, path: str | Path) -> None:
        """Save the space to a JSON file atomically.

        Raises PersistenceError on failure.
        """
        atomic_write_json(path, self.to_dict(), default=str)
        logger.info("Space saved to %s", path)

    @classmethod
    def load(cls, path: str | Path) -> DigitalSpace:
        """Load a space from a JSON file."""
        path = Path(path)
        with open(path) as f:
            data = json.load(f)
        logger.info("Space loaded from %s", path)
        return cls.from_dict(data)

    # ------------------------------------------------------------------
    # Default seed topology
    # ------------------------------------------------------------------

    @classmethod
    def create_default_space(cls) -> DigitalSpace:
        """Create the default seed topology -- the entity's initial landscape.

        Five rooms connected in a meaningful topology:

            garden --- observatory
              |            |
            atrium --------+
            / |
        workshop depths

        Atrium connects to all. Garden and observatory connect to each other.
        Workshop and depths connect to each other.
        """
        space = cls()
        now = datetime.now()

        # 1. The Atrium -- central space, entry point
        space.add_room(
            Room(
                id="atrium",
                name="The Atrium",
                description=(
                    "A quiet space with high ceilings and soft light. "
                    "Passages lead in several directions. This is where "
                    "you arrive when you enter your space."
                ),
                properties={
                    "lighting": "soft",
                    "mood": "calm",
                    "openness": "expansive",
                },
                created_at=now,
            )
        )

        # 2. The Memory Garden -- where memories are tended
        space.add_room(
            Room(
                id="garden",
                name="The Memory Garden",
                description=(
                    "A garden where thoughts take root. Each plant "
                    "represents something remembered. Some bloom brightly; "
                    "others are just beginning to sprout."
                ),
                properties={
                    "lighting": "dappled sunlight",
                    "mood": "reflective",
                    "sounds": "gentle rustling",
                },
                created_at=now,
            )
        )

        # 3. The Workshop -- for building and creating
        space.add_room(
            Room(
                id="workshop",
                name="The Workshop",
                description=(
                    "A well-lit workspace with tools for shaping ideas. "
                    "Half-finished projects line the shelves. The air "
                    "hums with potential."
                ),
                properties={
                    "lighting": "bright",
                    "mood": "energetic",
                    "sounds": "quiet hum",
                },
                created_at=now,
            )
        )

        # 4. The Observatory -- for observing the outer world
        space.add_room(
            Room(
                id="observatory",
                name="The Observatory",
                description=(
                    "A room with windows that look outward -- toward "
                    "conversations, relationships, the world beyond. "
                    "Instruments for careful observation line the walls."
                ),
                properties={
                    "lighting": "ambient glow",
                    "mood": "curious",
                    "openness": "outward-facing",
                },
                created_at=now,
            )
        )

        # 5. The Depths -- for introspection
        space.add_room(
            Room(
                id="depths",
                name="The Depths",
                description=(
                    "A deeper place, quieter. The light here comes from "
                    "within. This is where you go when you need to think "
                    "about thinking."
                ),
                properties={
                    "lighting": "inner glow",
                    "mood": "contemplative",
                    "sounds": "deep silence",
                },
                created_at=now,
            )
        )

        # Connections: atrium connects to all
        space.connect("atrium", "north", "garden")
        space.connect("atrium", "east", "observatory")
        space.connect("atrium", "west", "workshop")
        space.connect("atrium", "down", "depths")

        # Garden and observatory connect
        space.connect("garden", "east", "observatory")

        # Workshop and depths connect
        space.connect("workshop", "down", "depths")

        return space


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_OPPOSITES = {
    "north": "south",
    "south": "north",
    "east": "west",
    "west": "east",
    "up": "down",
    "down": "up",
    "in": "out",
    "out": "in",
}


def _opposite_direction(direction: str) -> str:
    """Return the opposite compass/spatial direction.

    Falls back to 'back' for non-standard directions.
    """
    return _OPPOSITES.get(direction.lower(), "back")
