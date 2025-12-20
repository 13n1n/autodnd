"""Player statistics models."""

from pydantic import BaseModel, Field


class PlayerStats(BaseModel):
    """Complete player statistics."""

    health: int = Field(ge=0, description="Current health points")
    max_health: int = Field(ge=1, description="Maximum health points")
    strength: int = Field(ge=0, description="Strength stat")
    dexterity: int = Field(ge=0, description="Dexterity stat")
    intelligence: int = Field(ge=0, description="Intelligence stat")
    charisma: int = Field(ge=0, description="Charisma stat")

    class Config:
        frozen = True  # Immutable model

