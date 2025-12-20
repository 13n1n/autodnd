"""Inventory management system."""

from autodnd.models.items import Bag, Item, ItemLocation, ItemTag, PlayerInventory


class InventoryManager:
    """Handles player inventory, bags, and equipment."""

    @staticmethod
    def can_place_item(
        inventory: PlayerInventory, item: Item, bag_index: int, slot_index: int
    ) -> bool:
        """
        Check if item can be placed in specified bag slot.

        Args:
            inventory: Player inventory
            item: Item to place
            bag_index: Index of bag
            slot_index: Index of slot within bag

        Returns:
            True if item can be placed, False otherwise
        """
        # Large items can't be placed in bags
        if ItemTag.LARGE in item.tags:
            return False

        # Check bag exists
        if bag_index >= len(inventory.bags):
            return False

        bag = inventory.bags[bag_index]

        # Check slot exists
        if slot_index >= bag.size:
            return False

        # Check slot is empty or can accommodate item
        if item.slot_size == 2:  # Heavy item needs 2 slots
            if slot_index >= bag.size - 1:
                return False  # Not enough space for 2-slot item
            if bag.items[slot_index] is not None or bag.items[slot_index + 1] is not None:
                return False  # Slots occupied
        else:
            if bag.items[slot_index] is not None:
                return False  # Slot occupied

        return True

    @staticmethod
    def place_item_in_bag(
        inventory: PlayerInventory, item: Item, bag_index: int, slot_index: int
    ) -> PlayerInventory:
        """
        Place item in bag slot.

        Args:
            inventory: Player inventory
            item: Item to place
            bag_index: Index of bag
            slot_index: Index of slot within bag

        Returns:
            New inventory with item placed
        """
        if not InventoryManager.can_place_item(inventory, item, bag_index, slot_index):
            raise ValueError(f"Cannot place item {item.item_id} in bag {bag_index}, slot {slot_index}")

        # Update item location
        updated_item = item.model_copy(
            update={
                "location": ItemLocation.BAG,
                "bag_index": bag_index,
                "slot_index": slot_index,
            }
        )

        # Update bag
        bag = inventory.bags[bag_index]
        new_items = list(bag.items)
        new_items[slot_index] = updated_item
        if item.slot_size == 2:
            new_items[slot_index + 1] = updated_item  # Heavy items occupy 2 slots

        new_bag = bag.model_copy(update={"items": new_items})

        # Update bags list
        new_bags = list(inventory.bags)
        new_bags[bag_index] = new_bag

        # Update all_items list
        new_all_items = [
            updated_item if old_item.item_id == item.item_id else old_item
            for old_item in inventory.all_items
        ]

        return inventory.model_copy(update={"bags": new_bags, "all_items": new_all_items})

    @staticmethod
    def equip_item(inventory: PlayerInventory, item: Item) -> PlayerInventory:
        """
        Equip an item (weapon, cloth, hat, pants, large_item).

        Args:
            inventory: Player inventory
            item: Item to equip

        Returns:
            New inventory with item equipped
        """
        # Determine equipment slot based on item tags
        updates = {}
        unequipped_item = None

        if ItemTag.WEAPON in item.tags:
            # Equip as primary or secondary weapon
            if inventory.primary_weapon is None:
                updates["primary_weapon"] = item
            elif inventory.secondary_weapon is None:
                updates["secondary_weapon"] = item
            else:
                # Replace primary weapon (unequip it first)
                unequipped_item = inventory.primary_weapon
                updates["primary_weapon"] = item
        elif ItemTag.CLOTH in item.tags:
            unequipped_item = inventory.cloth
            updates["cloth"] = item
        elif ItemTag.HAT in item.tags:
            unequipped_item = inventory.hat
            updates["hat"] = item
        elif ItemTag.PANTS in item.tags:
            unequipped_item = inventory.pants
            updates["pants"] = item
        elif ItemTag.LARGE in item.tags:
            unequipped_item = inventory.large_item
            updates["large_item"] = item
        else:
            raise ValueError(f"Item {item.item_id} cannot be equipped (no equipment tag)")

        # Update item location
        updated_item = item.model_copy(update={"location": ItemLocation.EQUIPPED})

        # Update all_items list
        new_all_items = [
            updated_item if old_item.item_id == item.item_id else old_item
            for old_item in inventory.all_items
        ]

        # If unequipped item exists, remove its equipment status
        if unequipped_item:
            unequipped_item = unequipped_item.model_copy(update={"location": ItemLocation.INVENTORY})

            new_all_items = [
                unequipped_item if old_item.item_id == unequipped_item.item_id else old_item
                for old_item in new_all_items
            ]

        # Remove item from bag if it was in one
        new_bags = inventory.bags
        if item.bag_index is not None and item.slot_index is not None:
            bag_index = item.bag_index
            slot_index = item.slot_index
            bag = inventory.bags[bag_index]
            new_bag_items = list(bag.items)
            new_bag_items[slot_index] = None
            if item.slot_size == 2:
                new_bag_items[slot_index + 1] = None
            new_bag = bag.model_copy(update={"items": new_bag_items})
            new_bags = list(inventory.bags)
            new_bags[bag_index] = new_bag

        return inventory.model_copy(update={**updates, "all_items": new_all_items})

