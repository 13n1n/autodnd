"""Hex map navigation utilities."""

from typing import Optional

from autodnd.models.world import HexCoordinate, HexMap, TerrainType


class HexNavigation:
    """Utilities for hex grid navigation and movement."""

    # Hex neighbor offsets in axial coordinates (q, r)
    # For pointy-topped hexagons: 6 neighbors
    HEX_DIRECTIONS = [
        (+1, 0),  # East
        (+1, -1),  # Northeast
        (0, -1),  # Northwest
        (-1, 0),  # West
        (-1, +1),  # Southwest
        (0, +1),  # Southeast
    ]

    # Direction names for easy reference
    DIRECTION_NAMES = ["east", "northeast", "northwest", "west", "southwest", "southeast"]
    DIRECTION_ABBREVIATIONS = ["E", "NE", "NW", "W", "SW", "SE"]

    @staticmethod
    def get_neighbor(coordinate: HexCoordinate, direction: int) -> HexCoordinate:
        """
        Get neighbor coordinate in given direction.

        Args:
            coordinate: Starting coordinate
            direction: Direction index (0-5) for the 6 hex directions

        Returns:
            Neighbor coordinate
        """
        if direction < 0 or direction >= len(HexNavigation.HEX_DIRECTIONS):
            raise ValueError(f"Direction must be between 0 and {len(HexNavigation.HEX_DIRECTIONS) - 1}")

        dq, dr = HexNavigation.HEX_DIRECTIONS[direction]
        return HexCoordinate(q=coordinate.q + dq, r=coordinate.r + dr)

    @staticmethod
    def get_all_neighbors(coordinate: HexCoordinate) -> list[HexCoordinate]:
        """Get all 6 neighbor coordinates."""
        return [HexNavigation.get_neighbor(coordinate, i) for i in range(6)]

    @staticmethod
    def get_direction_name(direction: int) -> str:
        """Get human-readable direction name."""
        if 0 <= direction < len(HexNavigation.DIRECTION_NAMES):
            return HexNavigation.DIRECTION_NAMES[direction]
        return "unknown"

    @staticmethod
    def get_direction_from_name(name: str) -> Optional[int]:
        """Get direction index from name or abbreviation."""
        name_lower = name.lower().strip()
        # Try full name
        if name_lower in HexNavigation.DIRECTION_NAMES:
            return HexNavigation.DIRECTION_NAMES.index(name_lower)
        # Try abbreviation
        name_upper = name.upper().strip()
        if name_upper in HexNavigation.DIRECTION_ABBREVIATIONS:
            return HexNavigation.DIRECTION_ABBREVIATIONS.index(name_upper)
        return None


    @staticmethod
    def is_valid_move(
        current: HexCoordinate, target: HexCoordinate, world_map: HexMap
    ) -> tuple[bool, Optional[str]]:
        """
        Check if a move from current to target is valid.

        Args:
            current: Current position
            target: Target position
            world_map: World map to check against

        Returns:
            Tuple of (is_valid, error_message)
        """
        print(abs(current.q - target.q), abs(current.r - target.r))
        if abs(current.q - target.q) <= 1 and abs(current.r - target.r) <= 1:
            return True, None
        return False, "Target is not adjacent"

    @staticmethod
    def get_movement_cost(
        from_cell: Optional[HexCoordinate], to_cell: HexCoordinate, world_map: HexMap
    ) -> int:
        """
        Get movement cost in half-days for moving to a cell.

        Args:
            from_cell: Source cell (optional, for future terrain-based costs)
            to_cell: Target cell
            world_map: World map

        Returns:
            Movement cost in half-days (default: 1 = HALF_DAY)
        """
        cell = world_map.get_cell(to_cell)
        if cell is None:
            # Default terrain (plains) - half day
            return 1

        # Terrain-based movement costs can be added here
        # For now, all terrain costs 1 half-day (as per design doc)
        terrain_costs = {
            TerrainType.PLAINS: 1,
            TerrainType.ROAD: 1,  # Roads might be faster in the future
            TerrainType.FOREST: 1,
            TerrainType.MOUNTAIN: 1,
            TerrainType.DESERT: 1,
            TerrainType.SWAMP: 1,
            TerrainType.CITY: 1,
            TerrainType.DUNGEON: 1,
            TerrainType.WATER: 1,  # Might require special handling
        }

        return terrain_costs.get(cell.terrain, 1)

