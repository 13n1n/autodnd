"""Main game engine for state management and action processing."""

import uuid
from datetime import datetime
from typing import Optional

from autodnd.agents.orchestrator import AgentOrchestrator
from autodnd.engine.action_validator import ActionValidator
from autodnd.engine.combat import CombatSystem
from autodnd.engine.history import StateHistory
from autodnd.engine.inventory_manager import InventoryManager
from autodnd.engine.stat_calculator import StatCalculator
from autodnd.engine.time_manager import TimeManager
from autodnd.models.actions import Action
from autodnd.models.messages import Message, MessageHistory, MessageSource, MessageType
from autodnd.models.state import GameState


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

        # Determine time cost
        time_cost = ActionValidator.get_time_cost(action)

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
        new_time = TimeManager.advance_time(self._state.current_time, time_cost)

        # Process action with agent orchestrator if available
        agent_messages: list[Message] = []
        master_response = f"Master acknowledges action: {action.action_type.value}"

        if self._orchestrator:
            # Extract action text from parameters
            action_text = action.parameters.get("text", action.action_type.value)
            
            # Create temporary state with player message for orchestrator
            temp_state = self._state.model_copy(update={"message_history": new_message_history})
            
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
            # Fallback: create placeholder master message
            master_message = Message(
                message_id=str(uuid.uuid4()),
                timestamp=datetime.now(),
                sequence_number=len(new_message_history.messages),
                source=MessageSource.MASTER,
                source_id=None,
                content=master_response,
                message_type=MessageType.RESPONSE,
                action_id=action.action_id,
            )
            new_message_history = new_message_history.add_message(master_message)

        # Update players (recalculate stats if needed)
        updated_players = []
        for player in self._state.players:
            if player.player_id == action.player_id:
                # Recalculate stats
                new_current_stats = StatCalculator.calculate_current_stats(
                    player, self._state.active_effects
                )
                updated_player = player.model_copy(update={"current_stats": new_current_stats})
                updated_players.append(updated_player)
            else:
                updated_players.append(player)

        # Create new state
        new_state = self._state.model_copy(
            update={
                "state_version": self._state.state_version + 1,
                "message_history": new_message_history,
                "current_time": new_time,
                "players": updated_players,
            }
        )

        # Create snapshot if needed (for now, snapshot every action - can be optimized later)
        should_snapshot = self._should_create_snapshot(action, new_state)
        if should_snapshot:
            self._history.create_snapshot(
                new_state, {"reason": "action", "action_id": action.action_id}
            )

        self._state = new_state
        return new_state, True, ""

    def _should_create_snapshot(self, action: Action, state: GameState) -> bool:
        """
        Determine if a snapshot should be created.

        Args:
            action: Action that was applied
            state: New state

        Returns:
            True if snapshot should be created
        """
        # Snapshot on combat start, level-up, or explicit save points
        # For now, snapshot on every action (can be optimized later)
        return True

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

