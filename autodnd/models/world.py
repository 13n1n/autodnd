"""World map and coordinate models."""

from enum import Enum
from typing import Optional

from pydantic import BaseModel, ConfigDict, Field


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
    """Complete world map with all cells."""

    model_config = ConfigDict(frozen=True)  # Immutable model

    cells: dict[tuple[int, int], HexCell] = Field(
        default_factory=dict, description="Map cells indexed by (q, r) coordinates"
    )

    def get_cell(self, coordinate: HexCoordinate) -> Optional[HexCell]:
        """Get cell at given coordinate."""
        key = (coordinate.q, coordinate.r)
        return self.cells.get(key)

    def set_cell(self, cell: HexCell) -> "HexMap":
        """Create new map with updated cell (immutable update)."""
        new_cells = self.cells.copy()
        new_cells[(cell.coordinates.q, cell.coordinates.r)] = cell
        return self.model_copy(update={"cells": new_cells})

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

