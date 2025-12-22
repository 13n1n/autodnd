"""Game state and snapshot models."""

from datetime import datetime
from typing import Any, Optional

from pydantic import BaseModel, ConfigDict, Field

from autodnd.models.actions import Action, CombatState, Effect
from autodnd.models.messages import Message, MessageHistory
from autodnd.models.metadata import GameMetadata
from autodnd.models.player import Player
from autodnd.models.world import HexMap, TimeState


class GameState(BaseModel):
    """LARGE, comprehensive state model - immutable and self-contained."""

    model_config = ConfigDict(frozen=True)  # Immutable model

    # Game identification
    game_id: str = Field(description="Unique game identifier")
    state_version: int = Field(ge=0, default=0, description="Increments with each state change")
    created_at: datetime = Field(default_factory=datetime.now, description="State creation timestamp")

    # All players' complete information
    players: list[Player] = Field(default_factory=list, description="All players with complete stats, inventory, position")

    # World state
    world_map: HexMap = Field(default_factory=HexMap, description="Complete hex map with all cells and their contents")
    current_time: TimeState = Field(default_factory=TimeState, description="Day, half-day increments, total time elapsed")

    # Complete message history - ALL interactions
    message_history: MessageHistory = Field(
        default_factory=MessageHistory,
        description="ALL interactions: player actions, master responses, NPC dialogue, tool outputs",
    )

    # Combat state (if in combat)
    combat_state: Optional[CombatState] = Field(
        default=None, description="Current combat if active, None otherwise"
    )

    # Active effects (buffs/debuffs)
    active_effects: list[Effect] = Field(
        default_factory=list, description="All temporary stat modifiers, potion effects, etc."
    )

    # Game metadata
    metadata: GameMetadata = Field(
        default_factory=GameMetadata, description="Settings, difficulty, etc."
    )

    # Key-value storage for game master data (hidden objectives, notes, etc.)
    storage: dict[str, str] = Field(
        default_factory=dict, description="Key-value storage for persistent game data"
    )

    def model_dump_json(self, **kwargs) -> str:
        """Serialize to JSON string."""
        return super().model_dump_json(**kwargs)

    def model_dump(self, **kwargs) -> dict[str, Any]:
        """Serialize to dict."""
        return super().model_dump(**kwargs)


class StateSnapshot(BaseModel):
    """Historical state snapshot for reversion."""

    model_config = ConfigDict(frozen=True)  # Immutable model

    index: int = Field(ge=0, description="Sequential snapshot number")
    timestamp: datetime = Field(default_factory=datetime.now, description="When snapshot was created")
    state: GameState = Field(description="Complete, self-contained state (all players, items, messages, etc.)")
    metadata: dict[str, Any] = Field(
        default_factory=dict, description="Additional snapshot metadata (reason, tags, etc.)"
    )

    def model_dump_json(self, **kwargs) -> str:
        """Serialize to JSON string."""
        return super().model_dump_json(**kwargs)

    def model_dump(self, **kwargs) -> dict[str, Any]:
        """Serialize to dict."""
        return super().model_dump(**kwargs)

