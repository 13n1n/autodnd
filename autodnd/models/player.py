"""Player model."""

from typing import Optional

from pydantic import BaseModel, ConfigDict, Field

from autodnd.models.items import PlayerInventory
from autodnd.models.stats import PlayerStats
from autodnd.models.world import HexCoordinate


class Player(BaseModel):
    """Complete player information."""

    model_config = ConfigDict(frozen=True)  # Immutable model

    player_id: str = Field(description="Unique player identifier")
    name: str = Field(description="Player name")

    # ALL stats (base + current effective values)
    base_stats: PlayerStats = Field(description="Base stats (Health, Strength, Dexterity, Intelligence, Charisma)")
    current_stats: PlayerStats = Field(description="Effective stats (base + items + buffs/debuffs)")
    level: int = Field(ge=1, default=1, description="Player level")
    experience: int = Field(ge=0, default=0, description="Experience points")

    # Complete inventory - ALL items
    inventory: PlayerInventory = Field(
        default_factory=PlayerInventory, description="Complete inventory state"
    )

    # Position and movement
    position: HexCoordinate = Field(description="Current hex cell position")
    movement_history: list[HexCoordinate] = Field(
        default_factory=list, description="Path taken (optional, for debugging)"
    )

    # Status
    is_alive: bool = Field(default=True, description="Whether player is alive")
    status_conditions: list[str] = Field(
        default_factory=list, description="Status conditions (e.g., 'poisoned', 'stunned')"
    )

