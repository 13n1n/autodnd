"""State serialization for agent consumption."""

from typing import Any, Optional

from autodnd.models.messages import MessageSource
from autodnd.models.state import GameState


class StateSerializer:
    """Serializes GameState to agent-readable format with subset extraction."""

    @staticmethod
    def serialize_full_state(state: GameState) -> dict[str, Any]:
        """
        Serialize complete game state to dict (for debugging/logging).

        Args:
            state: Game state to serialize

        Returns:
            Complete state as dict
        """
        return state.model_dump(mode="json")

    @staticmethod
    def extract_npc_context(
        state: GameState, npc_id: str, npc_location: Optional[tuple[int, int]] = None
    ) -> dict[str, Any]:
        """
        Extract context for NPC agent (principle of least privilege).

        NPC agent needs:
        - NPC location and nearby cells
        - Recent messages involving this NPC (last 10)
        - Player information if nearby
        - Current time

        Args:
            state: Game state
            npc_id: NPC identifier
            npc_location: Optional (q, r) tuple for NPC location

        Returns:
            Context dict for NPC agent
        """
        # Find NPC location if not provided
        if not npc_location:
            for (q, r), cell in state.world_map.cells.items():
                if npc_id in cell.contents:
                    npc_location = (q, r)
                    break

        # Get nearby cells
        nearby_cells = []
        nearby_players = []
        if npc_location:
            from autodnd.models.world import HexCoordinate

            coord = HexCoordinate(q=npc_location[0], r=npc_location[1])
            cell = state.world_map.get_cell(coord)
            if cell:
                nearby_cells.append(cell.model_dump(mode="json"))

            # Get cells within 1 hex distance
            for q_offset in range(-1, 2):
                for r_offset in range(-1, 2):
                    if q_offset == 0 and r_offset == 0:
                        continue
                    nearby_coord = HexCoordinate(q=npc_location[0] + q_offset, r=npc_location[1] + r_offset)
                    nearby_cell = state.world_map.get_cell(nearby_coord)
                    if nearby_cell:
                        nearby_cells.append(nearby_cell.model_dump(mode="json"))

            # Find players nearby
            for player in state.players:
                player_pos = player.position
                if abs(player_pos.q - npc_location[0]) <= 1 and abs(player_pos.r - npc_location[1]) <= 1:
                    nearby_players.append(player.model_dump(mode="json"))

        # Get recent messages involving this NPC
        npc_messages = []
        for msg in state.message_history.messages[-10:]:
            if msg.npc_id == npc_id or msg.source == MessageSource.NPC:
                npc_messages.append(
                    {
                        "source": msg.source.value,
                        "content": msg.content,
                        "message_type": msg.message_type.value,
                        "timestamp": msg.timestamp.isoformat(),
                    }
                )

        context = {
            "npc_id": npc_id,
            "npc_location": npc_location,
            "current_time": state.current_time.model_dump(mode="json"),
            "nearby_cells": nearby_cells,
            "nearby_players": nearby_players,
            "recent_npc_messages": npc_messages,
        }

        return context

    @staticmethod
    def extract_rag_context(state: GameState, query: str, player_id: Optional[str] = None) -> dict[str, Any]:
        """
        Extract context for RAG agent queries.

        RAG agent needs:
        - Query string
        - Current location/area context
        - Recent relevant messages (last 5)
        - Player stats if relevant to query

        Args:
            state: Game state
            query: Query string for RAG
            player_id: Optional player ID for context

        Returns:
            Context dict for RAG agent
        """
        # Get target player
        target_player = None
        if player_id:
            target_player = next((p for p in state.players if p.player_id == player_id), None)
        if not target_player and state.players:
            target_player = state.players[0]

        # Get recent messages
        recent_messages = state.message_history.messages[-5:] if state.message_history.messages else []

        context = {
            "query": query,
            "current_time": state.current_time.model_dump(mode="json"),
            "player_location": (
                target_player.position.model_dump(mode="json") if target_player else None
            ),
            "recent_messages": [
                {
                    "source": msg.source.value,
                    "content": msg.content[:200],  # Truncate for context
                    "timestamp": msg.timestamp.isoformat(),
                }
                for msg in recent_messages
            ],
            "player_stats": (
                target_player.current_stats.model_dump(mode="json") if target_player else None
            ),
        }

        return context
