"""Player model."""

import random
import uuid
from typing import Optional

from pydantic import BaseModel, ConfigDict, Field, model_validator

from autodnd.models.items import Bag, Item, ItemLocation, ItemTag, PlayerInventory
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

    @model_validator(mode="after")
    def generate_basic_bags(self) -> "Player":
        """Generate and equip random basic bag on player creation."""
        # Only generate if inventory is empty (new player)
        if self.inventory.all_items:
            return self

        # Generate basic bag
        bag = Bag(
            bag_id=str(uuid.uuid4()),
            size=random.randint(3, 12),
            items=[],
        )

        # Update inventory with equipped bag
        updated_inventory = self.inventory.model_copy(
            update={"bags": [bag], "all_items": []}
        )

        return self.model_copy(update={"inventory": updated_inventory})

    @model_validator(mode="after")
    def generate_basic_clothes(self) -> "Player":
        """Generate and equip random basic clothes on player creation."""
        # Only generate if inventory is empty (new player)
        if self.inventory.all_items:
            return self

        # Basic clothing names
        cloth_names = ["Simple Tunic", "Basic Shirt", "Plain Robe", "Worn Tunic"]
        hat_names = ["Simple Cap", "Basic Hat", "Plain Cap", "Worn Hat"]
        pants_names = ["Simple Pants", "Basic Trousers", "Plain Pants", "Worn Trousers"]

        # Generate basic cloth
        cloth = Item(
            item_id=str(uuid.uuid4()),
            name=random.choice(cloth_names),
            description="A basic piece of clothing, worn and simple.",
            tags=[ItemTag.CLOTH],
            location=ItemLocation.EQUIPPED,
            stat_modifiers={},  # Basic clothes have no stat modifiers
        )

        # Generate basic hat
        hat = Item(
            item_id=str(uuid.uuid4()),
            name=random.choice(hat_names),
            description="A basic hat, worn and simple.",
            tags=[ItemTag.HAT],
            location=ItemLocation.EQUIPPED,
            stat_modifiers={},  # Basic clothes have no stat modifiers
        )

        # Generate basic pants
        pants = Item(
            item_id=str(uuid.uuid4()),
            name=random.choice(pants_names),
            description="Basic pants, worn and simple.",
            tags=[ItemTag.PANTS],
            location=ItemLocation.EQUIPPED,
            stat_modifiers={},  # Basic clothes have no stat modifiers
        )

        # Update inventory with equipped clothes
        updated_inventory = self.inventory.model_copy(
            update={
                "cloth": cloth,
                "hat": hat,
                "pants": pants,
                "all_items": [cloth, hat, pants],
            }
        )


        return self.model_copy(update={"inventory": updated_inventory})

