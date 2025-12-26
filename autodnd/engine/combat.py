"""Combat system for turn-based combat."""

from autodnd.models.actions import CombatState


class CombatSystem:
    """Handles turn-based combat resolution."""

    @staticmethod
    def is_in_combat(state) -> bool:
        """Check if game is currently in combat."""
        return state.combat_state is not None and state.combat_state.is_active

    @staticmethod
    def get_current_turn_participant(combat_state: CombatState) -> str:
        """Get the participant ID whose turn it is."""
        if not combat_state.is_active or not combat_state.turn_order:
            raise ValueError("Combat is not active")
        return combat_state.turn_order[combat_state.current_turn % len(combat_state.turn_order)]

    @staticmethod
    def advance_turn(combat_state: CombatState) -> CombatState:
        """Advance to next turn in combat."""
        if not combat_state.is_active:
            return combat_state

        new_current_turn = (combat_state.current_turn + 1) % len(combat_state.turn_order)
        return combat_state.model_copy(update={"current_turn": new_current_turn})

