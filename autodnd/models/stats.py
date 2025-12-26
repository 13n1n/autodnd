"""Player statistics models."""

from pydantic import BaseModel, ConfigDict, Field


class PlayerStats(BaseModel):
    """Complete player statistics."""

    model_config = ConfigDict(frozen=True)  # Immutable model

    health: int = Field(ge=0, description="Current health points")
    max_health: int = Field(ge=1, description="Maximum health points")
    strength: int = Field(ge=0, description="Strength stat")
    dexterity: int = Field(ge=0, description="Dexterity stat")
    intelligence: int = Field(ge=0, description="Intelligence stat")
    charisma: int = Field(ge=0, description="Charisma stat")

