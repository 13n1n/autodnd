"""Main game engine for state management and action processing."""

import uuid
from datetime import datetime
from typing import Optional

from autodnd.agents.orchestrator import AgentOrchestrator
from autodnd.engine.action_validator import ActionValidator
from autodnd.engine.combat import CombatSystem
from autodnd.engine.hex_navigation import HexNavigation
from autodnd.engine.history import StateHistory
from autodnd.engine.inventory_manager import InventoryManager
from autodnd.engine.stat_calculator import StatCalculator
from autodnd.engine.time_manager import TimeManager
from autodnd.models.actions import Action, ActionType
from autodnd.models.items import ItemTag
from autodnd.models.messages import Message, MessageHistory, MessageSource, MessageType
from autodnd.models.state import GameState
from autodnd.models.world import HexCoordinate


class GameEngine:
    """Main state machine for game logic and progression."""

    def __init__(
        self, initial_state: Optional[GameState] = None, orchestrator: Optional[AgentOrchestrator] = None
    ) -> None:
        """
        Initialize game engine.

        Args:
            initial_state: Optional initial game state
            orchestrator: Optional agent orchestrator for agent interactions
        """
        if initial_state:
            self._state = initial_state
        else:
            self._state = self._create_initial_state()
        self._history = StateHistory()
        self._orchestrator = orchestrator
        # Create initial snapshot
        self._history.create_snapshot(self._state, {"reason": "initial_state"})

    def _create_initial_state(self) -> GameState:
        """Create initial game state."""
        return GameState(
            game_id=str(uuid.uuid4()),
            state_version=0,
            created_at=datetime.now(),
        )

    @property
    def state(self) -> GameState:
        """Get current game state."""
        return self._state

    @property
    def history(self) -> StateHistory:
        """Get state history."""
        return self._history

    @property
    def orchestrator(self) -> Optional[AgentOrchestrator]:
        """Get agent orchestrator."""
        return self._orchestrator

    def set_orchestrator(self, orchestrator: AgentOrchestrator) -> None:
        """Set agent orchestrator."""
        self._orchestrator = orchestrator

    def apply_action(self, action: Action) -> tuple[GameState, bool, str]:
        """
        Apply an action to the game state.

        Args:
            action: Action to apply

        Returns:
            Tuple of (new_state, success, error_message)
        """
        # Validate action
        is_valid, error_msg = ActionValidator.validate_action(action, self._state)
        if not is_valid:
            return self._state, False, error_msg

        # Determine time cost (pass state for USE_ITEM instant check)
        time_cost = ActionValidator.get_time_cost(action, self._state)

        # Apply action-specific state changes BEFORE agent processing
        # (e.g., movement, inventory changes)
        action_state_updates = self._apply_action_state_changes(action)

        # Create player action message
        player_message = Message(
            message_id=str(uuid.uuid4()),
            timestamp=datetime.now(),
            sequence_number=len(self._state.message_history.messages),
            source=MessageSource.PLAYER,
            source_id=action.player_id,
            content=f"Action: {action.action_type.value} {action.parameters}",
            message_type=MessageType.ACTION,
            action_id=action.action_id,
            metadata={"action_type": action.action_type.value, "parameters": action.parameters},
        )

        # Add player message to history
        new_message_history = self._state.message_history.add_message(player_message)

        # Update time
        new_time = TimeManager.advance_time(action_state_updates["current_time"], time_cost)

        # Apply action state updates to temp state for agent processing
        temp_state = action_state_updates["state"].model_copy(
            update={"message_history": new_message_history, "current_time": new_time}
        )

        # Process action with agent orchestrator if available
        agent_messages: list[Message] = []
        master_response = f"Master acknowledges action: {action.action_type.value}"

        if self._orchestrator:
            # Extract action text from parameters
            action_text = action.parameters.get("text", action.action_type.value)

            # Process through orchestrator
            master_response, agent_messages = self._orchestrator.process_player_action(
                action_text, temp_state, action.player_id
            )

            # Add all agent messages to history
            for msg in agent_messages:
                # Update sequence numbers and action_id
                updated_msg = msg.model_copy(
                    update={
                        "sequence_number": len(new_message_history.messages),
                        "action_id": action.action_id,
                    }
                )
                new_message_history = new_message_history.add_message(updated_msg)
        else:
            raise Exception("No orchestrator during applying action!")

        # Update players (recalculate stats if needed)
        updated_players = []
        for player in temp_state.players:
            if player.player_id == action.player_id:
                # Recalculate stats (in case equipment changed)
                new_current_stats = StatCalculator.calculate_current_stats(
                    player, temp_state.active_effects
                )
                updated_player = player.model_copy(update={"current_stats": new_current_stats})
                updated_players.append(updated_player)
            else:
                updated_players.append(player)

        # Create new state
        new_state = temp_state.model_copy(
            update={
                "state_version": self._state.state_version + 1,
                "message_history": new_message_history,
                "current_time": new_time,
                "players": updated_players,
            }
        )

        self._history.create_snapshot(
            new_state, {"reason": "action", "action_id": action.action_id}
        )

        self._state = new_state
        return new_state, True, ""

    def _apply_action_state_changes(self, action: Action) -> dict:
        """
        Apply action-specific state changes (movement, inventory, etc.).

        Returns:
            Dict with updated state and current_time (before time advancement)
        """
        from autodnd.models.world import TerrainType

        updated_state = self._state
        updated_time = self._state.current_time

        player = next((p for p in self._state.players if p.player_id == action.player_id), None)
        if not player:
            return {"state": updated_state, "current_time": updated_time}

        # Handle MOVE action
        if action.action_type == ActionType.MOVE:
            target_coordinate = None

            # Determine target coordinate
            if "target_coordinate" in action.parameters:
                coord_data = action.parameters["target_coordinate"]
                if isinstance(coord_data, dict):
                    target_coordinate = HexCoordinate(q=coord_data["q"], r=coord_data["r"])
            elif "direction" in action.parameters:
                direction_str = action.parameters["direction"]
                direction = HexNavigation.get_direction_from_name(direction_str)
                if direction is not None:
                    target_coordinate = HexNavigation.get_neighbor(player.position, direction)

            if target_coordinate:
                # Update player position
                new_movement_history = list(player.movement_history) + [player.position]
                updated_player = player.model_copy(
                    update={"position": target_coordinate, "movement_history": new_movement_history}
                )

                # Mark cell as discovered
                updated_map = updated_state.world_map.mark_discovered(target_coordinate)

                # Ensure cell exists in map
                cell = updated_map.get_cell(target_coordinate)
                if cell is None:
                    # Create cell if it doesn't exist
                    from autodnd.models.world import HexCell
                    new_cell = HexCell(
                        coordinates=target_coordinate,
                        terrain=TerrainType.PLAINS,
                        contents=[],
                        discovered=True,
                        description=None,
                    )
                    updated_map = updated_map.set_cell(new_cell)

                # Update players list
                updated_players = [
                    updated_player if p.player_id == action.player_id else p
                    for p in updated_state.players
                ]

                updated_state = updated_state.model_copy(
                    update={"players": updated_players, "world_map": updated_map}
                )

        # Handle EQUIP_ITEM action
        elif action.action_type == ActionType.EQUIP_ITEM:
            item_id = action.parameters.get("item_id")
            if item_id:
                item = next((i for i in player.inventory.all_items if i.item_id == item_id), None)
                if item:
                    try:
                        updated_inventory = InventoryManager.equip_item(player.inventory, item)
                        updated_player = player.model_copy(update={"inventory": updated_inventory})
                        updated_players = [
                            updated_player if p.player_id == action.player_id else p
                            for p in updated_state.players
                        ]
                        updated_state = updated_state.model_copy(update={"players": updated_players})
                    except ValueError:
                        # Equipment failed (validation already done, but handle gracefully)
                        pass

        # Handle UNEQUIP_ITEM action
        elif action.action_type == ActionType.UNEQUIP_ITEM:
            item_id = action.parameters.get("item_id")
            if item_id:
                # Find equipped item
                equipped_items = [
                    player.inventory.primary_weapon,
                    player.inventory.secondary_weapon,
                    player.inventory.cloth,
                    player.inventory.hat,
                    player.inventory.pants,
                    player.inventory.large_item,
                ]
                item = next((i for i in equipped_items if i and i.item_id == item_id), None)
                if item:
                    # Unequip by setting the appropriate slot to None
                    updates = {}
                    if player.inventory.primary_weapon and player.inventory.primary_weapon.item_id == item_id:
                        updates["primary_weapon"] = None
                    elif (
                        player.inventory.secondary_weapon
                        and player.inventory.secondary_weapon.item_id == item_id
                    ):
                        updates["secondary_weapon"] = None
                    elif player.inventory.cloth and player.inventory.cloth.item_id == item_id:
                        updates["cloth"] = None
                    elif player.inventory.hat and player.inventory.hat.item_id == item_id:
                        updates["hat"] = None
                    elif player.inventory.pants and player.inventory.pants.item_id == item_id:
                        updates["pants"] = None
                    elif (
                        player.inventory.large_item and player.inventory.large_item.item_id == item_id
                    ):
                        updates["large_item"] = None

                    if updates:
                        # Update item location to inventory
                        from autodnd.models.items import ItemLocation
                        updated_item = item.model_copy(update={"location": ItemLocation.INVENTORY})
                        new_all_items = [
                            updated_item if i.item_id == item_id else i
                            for i in player.inventory.all_items
                        ]
                        updated_inventory = player.inventory.model_copy(
                            update={**updates, "all_items": new_all_items}
                        )
                        updated_player = player.model_copy(update={"inventory": updated_inventory})
                        updated_players = [
                            updated_player if p.player_id == action.player_id else p
                            for p in updated_state.players
                        ]
                        updated_state = updated_state.model_copy(update={"players": updated_players})

        # Handle USE_ITEM action (basic implementation)
        elif action.action_type == ActionType.USE_ITEM:
            item_id = action.parameters.get("item_id")
            if item_id:
                item = next((i for i in player.inventory.all_items if i.item_id == item_id), None)
                if item:
                    # Handle instant items (potions, etc.)
                    if ItemTag.INSTANT in item.tags:
                        # Instant items are consumed/used immediately
                        # For potions, apply stat modifiers as temporary effects
                        if ItemTag.POTION in item.tags and item.stat_modifiers:
                            # Create effect from potion (simplified - could be enhanced)
                            # This would be better handled by the agent, but basic support here
                            pass
                        # Item could be consumed here (remove from inventory)
                        # For now, let the agent handle item consumption

        return {"state": updated_state, "current_time": updated_time}

    def revert_to(self, snapshot_index: int) -> tuple[bool, str]:
        """
        Revert game state to a specific snapshot.

        Args:
            snapshot_index: Index of snapshot to revert to

        Returns:
            Tuple of (success, error_message)
        """
        snapshot = self._history.get_snapshot(snapshot_index)
        if not snapshot:
            return False, f"Snapshot {snapshot_index} not found"

        # Restore state from snapshot
        self._state = snapshot.state

        # Truncate history after revert point
        self._history.truncate_after(snapshot_index)

        return True, ""

    def add_message(
        self,
        content: str,
        source: MessageSource,
        message_type: MessageType,
        source_id: Optional[str] = None,
        tool_name: Optional[str] = None,
        npc_id: Optional[str] = None,
        metadata: Optional[dict] = None,
    ) -> GameState:
        """
        Add a message to the game state (for agent interactions).

        Args:
            content: Message content
            source: Message source
            message_type: Message type
            source_id: Optional source ID
            tool_name: Optional tool name
            npc_id: Optional NPC ID
            metadata: Optional metadata

        Returns:
            New game state with message added
        """
        message = Message(
            message_id=str(uuid.uuid4()),
            timestamp=datetime.now(),
            sequence_number=len(self._state.message_history.messages),
            source=source,
            source_id=source_id,
            content=content,
            message_type=message_type,
            tool_name=tool_name,
            npc_id=npc_id,
            metadata=metadata or {},
        )

        new_message_history = self._state.message_history.add_message(message)
        new_state = self._state.model_copy(
            update={
                "state_version": self._state.state_version + 1,
                "message_history": new_message_history,
            }
        )
        self._state = new_state
        return new_state

    def update_storage(self, key: str, value: str) -> GameState:
        """
        Update storage in game state.

        Args:
            key: Storage key
            value: Storage value

        Returns:
            New game state with storage updated
        """
        new_storage = {**self._state.storage, key: value}
        new_state = self._state.model_copy(
            update={
                "state_version": self._state.state_version + 1,
                "storage": new_storage,
            }
        )
        self._state = new_state
        return new_state

    def take_item(
        self,
        item_id: str,
        item_name: Optional[str] = None,
        item_description: Optional[str] = None,
        tags: Optional[list[str]] = None,
        slot_size: int = 1,
        stat_modifiers: Optional[dict[str, int]] = None,
        player_id: Optional[str] = None,
        wear: bool = False,
    ) -> dict:
        """
        Take an item from a location and add it to player inventory.

        Args:
            item_id: Item ID
            item_name: Item name (required if creating new item)
            item_description: Item description (required if creating new item)
            tags: Item tags
            slot_size: Slot size (1 or 2)
            stat_modifiers: Stat modifiers when equipped
            player_id: Player ID
            wear: Whether to equip the item
            coordinate_q: Q coordinate where item is located
            coordinate_r: R coordinate where item is located

        Returns:
            Dict with success status and message
        """
        if tags is None:
            tags = []
        if stat_modifiers is None:
            stat_modifiers = {}

        # Get player
        player = next((p for p in self._state.players if p.player_id == player_id), None)
        if not player:
            return {"error": f"Player {player_id} not found"}


        # Get or create item
        from autodnd.models.items import Item, ItemTag, ItemLocation
        import uuid

        # Try to find existing item in player inventory first
        existing_item = next((i for i in player.inventory.all_items if i.item_id == item_id), None)
        
        if existing_item:
            # Item already in inventory
            return {"error": f"Item {item_id} already in player inventory"}

        # Check if item is a bag (bags are special - they're Bag objects, not Item objects)
        # We'll detect this by checking if item_name/description suggests it's a bag
        # or if there's a special indicator. For now, check if name contains "bag" (case-insensitive)
        is_bag = False
        if item_name:
            is_bag = "bag" in item_name.lower()

        # Create new item if needed
        if not is_bag and item_name and item_description:
            # Create new item
            item_tags = [ItemTag(tag) for tag in tags if tag in [t.value for t in ItemTag]]
            new_item = Item(
                item_id=item_id,
                name=item_name,
                description=item_description,
                tags=item_tags,
                slot_size=slot_size,
                stat_modifiers=stat_modifiers,
                location=ItemLocation.INVENTORY,
            )
        else:
            return {"error": "Item details not provided"}

        
        inventory = player.inventory
        bag_index = None
        slot_index = None
        
        # Handle bag items specially
        if is_bag:
            # Extract bag size from item description or use default
            # Try to extract size from description or use a default
            bag_size = stat_modifiers.get("bag_size", 6)  # Default bag size
            if item_description:
                # Try to extract number from description (e.g., "A bag with 8 slots")
                import re
                size_match = re.search(r'(\d+)\s*slot', item_description.lower())
                if size_match:
                    bag_size = int(size_match.group(1))
                    bag_size = max(3, min(12, bag_size))  # Clamp between 3 and 12
            
            # Check if we can add more bags (max 7)
            if len(inventory.bags) < 7:
                # Add new bag
                from autodnd.models.items import Bag
                new_bag = Bag(
                    bag_id=item_id,  # Use item_id as bag_id
                    size=bag_size,
                    items=[],
                )
                updated_bags = list(inventory.bags) + [new_bag]
                updated_inventory = inventory.model_copy(update={"bags": updated_bags})
            else:
                # Find first bag with lesser storage capacity
                replacement_bag_idx = None
                for b_idx, bag in enumerate(inventory.bags):
                    if bag.size < bag_size:
                        replacement_bag_idx = b_idx
                        break
                
                if replacement_bag_idx is None:
                    return {"error": "Cannot take bag: player already has 7 bags and no bag with lesser storage capacity to replace"}
                
                # Replace the bag
                from autodnd.models.items import Bag
                new_bag = Bag(
                    bag_id=item_id,
                    size=bag_size,
                    items=[],  # New bag starts empty
                )
                updated_bags = list(inventory.bags)
                updated_bags[replacement_bag_idx] = new_bag
                updated_inventory = inventory.model_copy(update={"bags": updated_bags})
                
                # Note: Items from replaced bag are lost (could be improved to try to preserve items)
                result_msg = f"Bag {item_name} ({item_id}) replaced bag at index {replacement_bag_idx}"
        else:
            # Regular item - find first available bag slot
            # Check if item is large (can't go in bag)
            if ItemTag.LARGE in new_item.tags:
                # Large items must be equipped or stored as large_item
                if wear:
                    try:
                        updated_inventory = InventoryManager.equip_item(inventory, new_item)
                    except ValueError as e:
                        return {"error": f"Cannot equip large item: {str(e)}"}
                else:
                    # Can't store large items in bags, must equip or return error
                    return {"error": "Large items cannot be stored in bags. Use wear=true to equip it."}
            else:
                # Place item in bag
                try:
                    updated_inventory = InventoryManager.place_item_in_bag(inventory, new_item)
                except ValueError as e:
                    return {"error": f"Cannot place item in bag: {str(e)}", "exception": str(e)}

            # If wear=true, equip the item
            if wear:
                try:
                    updated_inventory = InventoryManager.equip_item(updated_inventory, new_item)
                except ValueError as e:
                    # If equip fails, item is still in bag, which is fine
                    pass

        # For non-bag items, ensure item is in all_items
        if not is_bag:
            # Check if item is already in all_items (shouldn't be, but be safe)
            if not any(i.item_id == item_id for i in updated_inventory.all_items):
                updated_all_items = list(updated_inventory.all_items) + [new_item]
                updated_inventory = updated_inventory.model_copy(update={"all_items": updated_all_items})


        # Update player inventory
        updated_player = player.model_copy(update={"inventory": updated_inventory})
        updated_players = [
            updated_player if p.player_id == player_id else p
            for p in self._state.players
        ]

        # Create new state
        new_state = self._state.model_copy(
            update={
                "state_version": self._state.state_version + 1,
                "players": updated_players,
            }
        )
        self._state = new_state

        # Generate result message
        if is_bag:
            if len(inventory.bags) < 7:
                result_msg = f"Bag {item_name or item_id} ({item_id}) added to inventory"
            else:
                result_msg = f"Bag {item_name or item_id} ({item_id}) replaced existing bag"
        else:
            result_msg = f"Item {new_item.name} ({item_id}) added to inventory"
            if bag_index is not None:
                result_msg += f" in bag {bag_index}, slot {slot_index}"
            if wear:
                result_msg += " and equipped"

        return {"success": True, "message": result_msg, "item_id": item_id, "item_name": item_name or new_item.name}

    def get_storage(self, key: str) -> Optional[str]:
        """
        Get value from storage.

        Args:
            key: Storage key

        Returns:
            Storage value or None if not found
        """
        return self._state.storage.get(key)

