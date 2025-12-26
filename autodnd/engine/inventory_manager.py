"""Inventory management system."""

from autodnd.models.items import Bag, Item, ItemLocation, ItemTag, PlayerInventory


class InventoryManager:
    """Handles player inventory, bags, and equipment."""

    @staticmethod
    def place_item_in_bag(
        inventory: PlayerInventory, item: Item
    ) -> PlayerInventory:
        import copy

        for i, bag in enumerate(inventory.bags):
            if bag.size - len(bag.items) >= item.slot_size:
                break
        else:
            raise ValueError(f"No available bag slot for item {item.item_id}")

        new_bags = copy.deepcopy(inventory.bags)
        new_bags[i].items.append(item)

        return inventory.model_copy(update={"bags": new_bags})

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

