"""Action validation system."""

from autodnd.models.actions import Action, ActionType, TimeCost
from autodnd.models.state import GameState


class ActionValidator:
    """Validates player actions against game rules."""

    @staticmethod
    def validate_action(action: Action, state: GameState) -> tuple[bool, str]:
        """
        Validate an action against game state and rules.

        Args:
            action: Action to validate
            state: Current game state

        Returns:
            Tuple of (is_valid, error_message)
        """
        # Check player exists
        player = next((p for p in state.players if p.player_id == action.player_id), None)
        if not player:
            return False, f"Player {action.player_id} not found"

        if not player.is_alive:
            return False, f"Player {action.player_id} is not alive"

        # Type-specific validation
        if action.action_type == ActionType.MOVE:
            return ActionValidator._validate_move(action, state, player)
        elif action.action_type == ActionType.USE_ITEM:
            return ActionValidator._validate_use_item(action, state, player)
        elif action.action_type == ActionType.EQUIP_ITEM:
            return ActionValidator._validate_equip_item(action, state, player)
        elif action.action_type == ActionType.UNEQUIP_ITEM:
            return ActionValidator._validate_unequip_item(action, state, player)
        elif action.action_type == ActionType.ATTACK:
            return ActionValidator._validate_attack(action, state, player)
        # Add other action types as needed

        return True, ""

    @staticmethod
    def _validate_move(action: Action, state: GameState, player) -> tuple[bool, str]:
        """Validate move action."""
        if "direction" not in action.parameters and "target_coordinate" not in action.parameters:
            return False, "Move action requires 'direction' or 'target_coordinate' parameter"
        # Additional validation can be added here (check if target is reachable, etc.)
        return True, ""

    @staticmethod
    def _validate_use_item(action: Action, state: GameState, player) -> tuple[bool, str]:
        """Validate use item action."""
        if "item_id" not in action.parameters:
            return False, "Use item action requires 'item_id' parameter"

        item_id = action.parameters["item_id"]
        item = next((i for i in player.inventory.all_items if i.item_id == item_id), None)
        if not item:
            return False, f"Item {item_id} not found in player inventory"

        return True, ""

    @staticmethod
    def _validate_equip_item(action: Action, state: GameState, player) -> tuple[bool, str]:
        """Validate equip item action."""
        if "item_id" not in action.parameters:
            return False, "Equip item action requires 'item_id' parameter"

        item_id = action.parameters["item_id"]
        item = next((i for i in player.inventory.all_items if i.item_id == item_id), None)
        if not item:
            return False, f"Item {item_id} not found in player inventory"

        return True, ""

    @staticmethod
    def _validate_unequip_item(action: Action, state: GameState, player) -> tuple[bool, str]:
        """Validate unequip item action."""
        if "item_id" not in action.parameters:
            return False, "Unequip item action requires 'item_id' parameter"

        item_id = action.parameters["item_id"]
        # Check if item is equipped
        equipped_items = [
            player.inventory.primary_weapon,
            player.inventory.secondary_weapon,
            player.inventory.cloth,
            player.inventory.hat,
            player.inventory.pants,
            player.inventory.large_item,
        ]
        if not any(item and item.item_id == item_id for item in equipped_items):
            return False, f"Item {item_id} is not equipped"

        return True, ""

    @staticmethod
    def _validate_attack(action: Action, state: GameState, player) -> tuple[bool, str]:
        """Validate attack action."""
        # Check if player has a weapon equipped
        active_weapon = (
            player.inventory.primary_weapon
            if player.inventory.active_weapon == "primary"
            else player.inventory.secondary_weapon
        )
        if not active_weapon:
            return False, "Player must have a weapon equipped to attack"

        return True, ""

    @staticmethod
    def get_time_cost(action: Action) -> TimeCost:
        """
        Determine time cost for an action.

        Args:
            action: Action to determine time cost for

        Returns:
            TimeCost enum value
        """
        # Use action's time_cost if set, otherwise determine from action type
        if action.time_cost != TimeCost.NO_TIME:
            return action.time_cost

        # Default time costs based on action type
        if action.action_type in [ActionType.ATTACK, ActionType.TALK]:
            return TimeCost.NO_TIME
        elif action.action_type == ActionType.MOVE:
            return TimeCost.HALF_DAY
        elif action.action_type in [
            ActionType.REST,
            ActionType.SEARCH,
            ActionType.CRAFT,
            ActionType.INTERACT,
        ]:
            return TimeCost.HALF_DAY
        else:
            return TimeCost.NO_TIME

