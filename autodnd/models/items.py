"""Item and inventory models."""

from enum import Enum
from typing import Literal, Optional

from pydantic import BaseModel, Field, model_validator


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

    class Config:
        frozen = True  # Immutable model


class Bag(BaseModel):
    """Bag container with slots."""

    bag_id: str = Field(description="Unique bag identifier")
    size: int = Field(ge=3, le=12, description="Number of slots in bag")
    items: list[Optional[Item]] = Field(
        default_factory=list, description="Items in bag slots (None for empty slots)"
    )

    class Config:
        frozen = True  # Immutable model

    @model_validator(mode="after")
    def validate_items_size(self) -> "Bag":
        """Ensure items list matches bag size."""
        # Pad or truncate items list to match size
        if len(self.items) < self.size:
            padded_items = self.items + [None] * (self.size - len(self.items))
            return self.model_copy(update={"items": padded_items})
        elif len(self.items) > self.size:
            truncated_items = self.items[: self.size]
            return self.model_copy(update={"items": truncated_items})
        return self


class PlayerInventory(BaseModel):
    """Complete inventory state."""

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

    # All items with complete information
    all_items: list[Item] = Field(
        default_factory=list, description="Complete list of every item the player owns"
    )

    class Config:
        frozen = True  # Immutable model

