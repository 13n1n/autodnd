"""World map and coordinate models."""

import random
from enum import Enum
from typing import Optional

from pydantic import BaseModel, ConfigDict, Field, model_validator


class TerrainType(str, Enum):
    """Terrain types for hex cells."""

    PLAINS = "plains"
    FOREST = "forest"
    MOUNTAIN = "mountain"
    WATER = "water"
    DESERT = "desert"
    SWAMP = "swamp"
    ROAD = "road"
    CITY = "city"
    DUNGEON = "dungeon"


class HexCoordinate(BaseModel):
    """Hex grid coordinates (axial coordinate system)."""

    model_config = ConfigDict(frozen=True)  # Immutable model

    q: int = Field(description="Q coordinate (axial)")
    r: int = Field(description="R coordinate (axial)")


class HexCell(BaseModel):
    """Individual map cell."""

    model_config = ConfigDict(frozen=True)  # Immutable model

    coordinates: HexCoordinate = Field(description="Hex coordinates")
    terrain: TerrainType = Field(description="Terrain type")
    contents: list[str] = Field(
        default_factory=list, description="Contents of cell (NPCs, items, etc.)"
    )
    discovered: bool = Field(default=False, description="Whether cell has been discovered")
    description: Optional[str] = Field(default=None, description="Cell description")


class HexMap(BaseModel):
    """Complete world map with all cells. It does not changes at all during the game."""

    model_config = ConfigDict(frozen=True)  # Immutable model

    cells: dict[tuple[int, int], HexCell] = Field(
        default_factory=dict, description="Map cells indexed by (q, r) coordinates"
    )

    @staticmethod
    def _simple_noise(x: float, y: float, seed: int = 0) -> float:
        """Simple Perlin-like noise function using hash-based interpolation."""

        if abs(x) < 5 and abs(y) < 5:
            return 0.5 # Make sure that cells near origin are always flat

        # Use a simple hash function for pseudo-random values
        def hash_coord(xi: int, yi: int) -> int:
            n = xi * 374761393 + yi * 668265263 + seed
            n = (n ^ (n >> 13)) * 1274126177
            return (n ^ (n >> 16)) & 0x7FFFFFFF

        # Get integer coordinates
        xi, yi = int(x), int(y)
        xf, yf = x - xi, y - yi

        # Get noise values at corners
        n00 = hash_coord(xi, yi) / 2147483647.0
        n10 = hash_coord(xi + 1, yi) / 2147483647.0
        n01 = hash_coord(xi, yi + 1) / 2147483647.0
        n11 = hash_coord(xi + 1, yi + 1) / 2147483647.0

        # Smooth interpolation
        def smooth(t: float) -> float:
            return t * t * (3.0 - 2.0 * t)

        sx = smooth(xf)
        sy = smooth(yf)

        # Bilinear interpolation
        n0 = n00 * (1 - sx) + n10 * sx
        n1 = n01 * (1 - sx) + n11 * sx
        return n0 * (1 - sy) + n1 * sy

    @model_validator(mode='before')
    @classmethod
    def generate_map(cls, data: dict | None) -> dict:
        """
        Convert string keys in cells dict back to tuples when loading from JSON,
        and generate map if cells are missing.
        
        JSON doesn't support tuple keys, so they get serialized as strings like "50,-1".
        This validator converts them back to tuples (50, -1).
        """
        # Handle None or empty input
        if data is None:
            data = {}
        
        if not isinstance(data, dict):
            return data
        
        # Convert string keys to tuples if cells exist
        if 'cells' in data and isinstance(data['cells'], dict):
            cells = data['cells']
            converted_cells = {}
            
            for key, value in cells.items():
                if isinstance(key, str):
                    # Parse string key - handle multiple formats:
                    # "50,-1", "(50, -1)", "[50, -1]", etc.
                    try:
                        # Remove parentheses/brackets if present
                        cleaned = key.strip('()[]')
                        # Split by comma and strip whitespace
                        parts = [p.strip() for p in cleaned.split(',')]
                        if len(parts) == 2:
                            q = int(parts[0])
                            r = int(parts[1])
                            converted_cells[(q, r)] = value
                        else:
                            # Invalid format, skip
                            continue
                    except (ValueError, AttributeError):
                        # Invalid key format, skip
                        continue
                elif isinstance(key, (tuple, list)) and len(key) == 2:
                    # Already a tuple or list, ensure it's a tuple
                    converted_cells[tuple(key)] = value
                else:
                    # Keep as is (shouldn't happen, but be safe)
                    converted_cells[key] = value
            
            data['cells'] = converted_cells
            
            # If cells were successfully converted and not empty, return early
            if converted_cells:
                return data
        
        # If cells are missing or empty, generate new map

        # Generate map in a radius around origin (0, 0)
        map_radius = 50  # Generate cells within 50 hexes of origin
        cells: dict[tuple[int, int], HexCell] = {}

        # Random seed for reproducibility (can be made configurable)
        seed = random.randint(0, 1000000)
        random.seed(seed)

        # Flat terrain options (excluding water and mountain which use noise)
        flat_terrains = [
            TerrainType.PLAINS,
            TerrainType.FOREST,
            TerrainType.DESERT,
            TerrainType.SWAMP,
            TerrainType.PLAINS,
            TerrainType.FOREST,
            TerrainType.DESERT,
            TerrainType.SWAMP,
            TerrainType.PLAINS,
            TerrainType.FOREST,
            TerrainType.DESERT,
            TerrainType.SWAMP,
            TerrainType.PLAINS,
            TerrainType.FOREST,
            TerrainType.DESERT,
            TerrainType.SWAMP,
            TerrainType.PLAINS,
            TerrainType.FOREST,
            TerrainType.DESERT,
            TerrainType.SWAMP,
            TerrainType.PLAINS,
            TerrainType.FOREST,
            TerrainType.DESERT,
            TerrainType.SWAMP,
            TerrainType.PLAINS,
            TerrainType.FOREST,
            TerrainType.DESERT,
            TerrainType.SWAMP,
            TerrainType.CITY,
            TerrainType.DUNGEON,
        ]


        # Generate cells in a hexagon pattern
        for q in range(-map_radius, map_radius + 1):
            for r in range(-map_radius, map_radius + 1):
                # Check if within hexagon radius
                # Convert to cube coordinates for distance check
                s = -q - r
                if abs(q) + abs(r) + abs(s) > map_radius * 2:
                    continue

                coord = HexCoordinate(q=q, r=r)
                
                # Use noise for water and mountains
                # Scale coordinates for noise (smaller scale = larger features)
                noise_x = q * 0.1
                noise_y = r * 0.1
                
                # Water noise (lower values = water)
                water_noise = cls._simple_noise(noise_x, noise_y, seed)
                
                # Mountain noise (higher values = mountains)
                mountain_noise = cls._simple_noise(noise_x + 1000, noise_y + 1000, seed + 1000)
                
                # Determine terrain
                if water_noise < 0.25:  # ~25% chance for water
                    terrain = TerrainType.WATER
                elif mountain_noise > 0.8:  # ~30% chance for mountains
                    terrain = TerrainType.MOUNTAIN
                elif q == 0 and r == 0:
                    terrain = random.choice([
                        TerrainType.CITY,
                        TerrainType.DUNGEON,
                        TerrainType.FOREST,
                        TerrainType.DESERT
                    ])
                else:
                    # Randomly choose from flat terrains
                    terrain = random.choice(flat_terrains)

                cell = HexCell(
                    coordinates=coord,
                    terrain=terrain,
                    contents=[],
                    discovered=False,
                    description=None,
                )
                cells[(q, r)] = cell


        # TODO: Connect cells all cities and dungeons with network of roads


        return {'cells': cells}

    def set_cell(self, cell: HexCell) -> "HexMap":
        """Create new map with updated cell (immutable update)."""
        new_cells = self.cells.copy()
        new_cells[(cell.coordinates.q, cell.coordinates.r)] = cell
        return self.model_copy(update={"cells": new_cells})

    def get_cell(self, coordinate: HexCoordinate) -> Optional[HexCell]:
        """Get cell at given coordinate."""
        key = (coordinate.q, coordinate.r)
        return self.cells.get(key)

    def get_or_create_cell(self, coordinate: HexCoordinate, terrain: TerrainType = TerrainType.PLAINS) -> HexCell:
        """Get cell at coordinate, or create it if it doesn't exist."""
        cell = self.get_cell(coordinate)
        if cell is None:
            return HexCell(
                coordinates=coordinate,
                terrain=terrain,
                contents=[],
                discovered=False,
                description=None,
            )
        return cell

    def mark_discovered(self, coordinate: HexCoordinate) -> "HexMap":
        """Mark a cell as discovered (immutable update)."""
        cell = self.get_cell(coordinate)
        if cell is None:
            # Create cell if it doesn't exist
            new_cell = HexCell(
                coordinates=coordinate,
                terrain=TerrainType.PLAINS,
                contents=[],
                discovered=True,
                description=None,
            )
        else:
            new_cell = cell.model_copy(update={"discovered": True})
        return self.set_cell(new_cell)


class TimeState(BaseModel):
    """Game time tracking."""

    model_config = ConfigDict(frozen=True)  # Immutable model

    current_day: int = Field(ge=1, default=1, description="Current day number")
    half_day_increment: int = Field(ge=0, default=0, description="Half-day increments within day (0 or 1)")
    total_time_elapsed: int = Field(ge=0, default=0, description="Total half-days elapsed")

