"""Action and combat models."""

from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, Field


class ActionType(str, Enum):
    """Types of player actions."""

    MOVE = "move"
    ATTACK = "attack"
    USE_ITEM = "use_item"
    EQUIP_ITEM = "equip_item"
    UNEQUIP_ITEM = "unequip_item"
    TALK = "talk"
    SEARCH = "search"
    REST = "rest"
    CRAFT = "craft"
    TRADE = "trade"
    CAST_SPELL = "cast_spell"
    INTERACT = "interact"


class TimeCost(str, Enum):
    """Time cost for actions."""

    NO_TIME = "no_time"
    HALF_DAY = "half_day"
    WHOLE_DAY = "whole_day"


class Action(BaseModel):
    """Player action with validation."""

    action_id: str = Field(description="Unique action identifier")
    action_type: ActionType = Field(description="Type of action")
    parameters: dict[str, Any] = Field(
        default_factory=dict, description="Action parameters (target, direction, item_id, etc.)"
    )
    time_cost: TimeCost = Field(default=TimeCost.NO_TIME, description="Time cost of action")
    player_id: str = Field(description="Player performing action")

    class Config:
        frozen = True  # Immutable model


class CombatState(BaseModel):
    """Current combat information (if active)."""

    combat_id: str = Field(description="Unique combat identifier")
    participants: list[str] = Field(description="List of participant IDs (players and NPCs)")
    turn_order: list[str] = Field(description="Turn order (list of participant IDs)")
    current_turn: int = Field(ge=0, description="Current turn index in turn_order")
    combat_log: list[str] = Field(default_factory=list, description="Combat action log")
    is_active: bool = Field(default=True, description="Whether combat is currently active")

    class Config:
        frozen = True  # Immutable model


class Effect(BaseModel):
    """Active buff/debuff effect."""

    effect_id: str = Field(description="Unique effect identifier")
    effect_type: str = Field(description="Type of effect (buff/debuff)")
    duration: int = Field(ge=0, description="Duration in half-days remaining")
    stat_modifications: dict[str, int] = Field(
        default_factory=dict, description="Stat modifications (stat_name: modifier)"
    )
    source: str = Field(description="Source of effect (item_id, spell_id, etc.)")
    description: str = Field(description="Effect description")

    class Config:
        frozen = True  # Immutable model

