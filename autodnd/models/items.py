"""Item and inventory models."""

from enum import Enum
from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, computed_field, model_validator


class ItemTag(str, Enum):
    """Item tags for categorization."""

    INSTANT = "instant"
    HEAVY = "heavy"
    LARGE = "large"
    WEAPON = "weapon"
    HAT = "hat"
    CLOTH = "cloth"
    PANTS = "pants"
    POTION = "potion"
    BOOK = "book"
    SCROLL = "scroll"


class ItemLocation(str, Enum):
    """Location of an item."""

    INVENTORY = "inventory"
    EQUIPPED = "equipped"
    BAG = "bag"


class Item(BaseModel):
    """Complete item information."""

    model_config = ConfigDict(frozen=True)  # Immutable model

    item_id: str = Field(description="Unique item identifier")
    name: str = Field(description="Item name")
    description: str = Field(description="Item description")

    # Item properties
    tags: list[ItemTag] = Field(default_factory=list, description="Item tags")
    slot_size: int = Field(ge=1, le=2, default=1, description="Slot size (1 for normal, 2 for heavy)")
    stat_modifiers: dict[str, int] = Field(
        default_factory=dict, description="Stat changes when equipped/used"
    )

    # Location tracking
    location: ItemLocation = Field(default=ItemLocation.INVENTORY, description="Item location")
    bag_index: Optional[int] = Field(default=None, ge=0, le=6, description="Bag index if in bag")
    slot_index: Optional[int] = Field(
        default=None, ge=0, description="Slot index within bag if in bag"
    )


class Bag(BaseModel):
    """Bag container with slots."""

    model_config = ConfigDict(frozen=True)  # Immutable model

    bag_id: str = Field(description="Unique bag identifier")
    size: int = Field(ge=3, le=12, description="Number of slots in bag")
    items: list[Optional[Item]] = Field(
        default_factory=list, description="Items in bag slots (None for empty slots)"
    )


class PlayerInventory(BaseModel):
    """Complete inventory state."""

    model_config = ConfigDict(frozen=True)  # Immutable model

    # All bags (up to 7)
    bags: list[Bag] = Field(default_factory=list, max_length=7, description="All bags")

    # Equipped items
    primary_weapon: Optional[Item] = Field(default=None, description="Primary weapon")
    secondary_weapon: Optional[Item] = Field(default=None, description="Secondary weapon")
    active_weapon: Literal["primary", "secondary"] = Field(
        default="primary", description="Which weapon is active"
    )
    cloth: Optional[Item] = Field(default=None, description="Equipped cloth")
    hat: Optional[Item] = Field(default=None, description="Equipped hat")
    pants: Optional[Item] = Field(default=None, description="Equipped pants")
    large_item: Optional[Item] = Field(
        default=None, description="Large item (takes 2 hex cells, e.g., horse)"
    )

    @computed_field
    def all_items(self) -> list[Item]:
        """Complete list of every item the player owns."""
        return [item for bag in self.bags for item in bag.items]
