"""Game metadata model."""

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class Difficulty(str, Enum):
    """Game difficulty levels."""

    EASY = "easy"
    NORMAL = "normal"
    HARD = "hard"
    NIGHTMARE = "nightmare"


class RulesVariant(str, Enum):
    """DnD rules variant."""

    STANDARD = "standard"
    SIMPLIFIED = "simplified"
    CUSTOM = "custom"


class GameMetadata(BaseModel):
    """Game settings and metadata."""

    difficulty: Difficulty = Field(default=Difficulty.NORMAL, description="Game difficulty")
    rules_variant: RulesVariant = Field(
        default=RulesVariant.STANDARD, description="Rules variant"
    )
    settings: dict[str, Any] = Field(
        default_factory=dict, description="Additional game settings"
    )

    class Config:
        frozen = True  # Immutable model

