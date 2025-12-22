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
    def extract_game_master_context(state: GameState, player_id: Optional[str] = None) -> dict[str, Any]:
        """
        Extract context for Game Master agent (principle of least privilege).

        Game Master needs:
        - Current player(s) stats and position
        - Recent message history (last 20 messages)
        - Current time
        - World map around player position
        - Combat state if active
        - Active effects

        Args:
            state: Game state
            player_id: Optional player ID to focus on (if None, uses first player)

        Returns:
            Context dict for Game Master
        """
        # Get target player
        target_player = None
        if player_id:
            target_player = next((p for p in state.players if p.player_id == player_id), None)
        if not target_player and state.players:
            target_player = state.players[0]

        # Get recent messages (last 20)
        recent_messages = state.message_history.messages[-20:] if state.message_history.messages else []

        # Get map cells around player (if player exists)
        nearby_cells = []
        if target_player:
            from autodnd.models.world import HexCoordinate

            player_pos = target_player.position
            # Get cells within 2 hex distance
            for q_offset in range(-2, 3):
                for r_offset in range(-2, 3):
                    coord = HexCoordinate(q=player_pos.q + q_offset, r=player_pos.r + r_offset)
                    cell = state.world_map.get_cell(coord)
                    if cell:
                        nearby_cells.append(cell.model_dump(mode="json"))

        context = {
            "current_time": state.current_time.model_dump(mode="json"),
            "players": [p.model_dump(mode="json") for p in state.players] if state.players else [],
            "target_player": target_player.model_dump(mode="json") if target_player else None,
            "recent_messages": [
                {
                    "source": msg.source.value,
                    "content": msg.content,
                    "message_type": msg.message_type.value,
                    "timestamp": msg.timestamp.isoformat(),
                    "source_id": msg.source_id,
                }
                for msg in recent_messages
            ],
            "nearby_cells": nearby_cells,
            "combat_state": state.combat_state.model_dump(mode="json") if state.combat_state else None,
            "active_effects": [effect.model_dump(mode="json") for effect in state.active_effects],
        }

        return context

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

    @staticmethod
    def format_context_for_agent(context: dict[str, Any], agent_type: str) -> str:
        """
        Format context dict as readable string for agent prompt.

        Args:
            context: Context dict
            agent_type: Type of agent ("game_master", "npc", "rag")

        Returns:
            Formatted string for agent consumption
        """
        if agent_type == "game_master":
            lines = ["=== Game Master Context ==="]
            if context.get("target_player"):
                player = context["target_player"]
                lines.append(f"Player: {player.get('name', 'Unknown')} (ID: {player.get('player_id')})")
                lines.append(f"Level: {player.get('level', 1)}")
                lines.append(f"Position: ({player.get('position', {}).get('q', 0)}, {player.get('position', {}).get('r', 0)})")
                stats = player.get("current_stats", {})
                lines.append(
                    f"Stats: HP={stats.get('health', 0)}/{stats.get('max_health', 0)}, "
                    f"STR={stats.get('strength', 0)}, DEX={stats.get('dexterity', 0)}, "
                    f"INT={stats.get('intelligence', 0)}, CHA={stats.get('charisma', 0)}"
                )
            lines.append(f"Current Time: Day {context.get('current_time', {}).get('day', 0)}, Half-day {context.get('current_time', {}).get('half_day', 0)}")
            if context.get("combat_state"):
                lines.append("⚠️ Combat is active!")
            if context.get("recent_messages"):
                lines.append("\nRecent Messages:")
                for msg in context["recent_messages"][-5:]:
                    lines.append(f"  [{msg.get('source', 'unknown')}] {msg.get('content', '')[:100]}")
            return "\n".join(lines)

        elif agent_type == "npc":
            lines = [f"=== NPC Context (ID: {context.get('npc_id', 'unknown')}) ==="]
            if context.get("npc_location"):
                lines.append(f"Location: ({context['npc_location'][0]}, {context['npc_location'][1]})")
            if context.get("nearby_players"):
                lines.append(f"Nearby Players: {len(context['nearby_players'])}")
            if context.get("recent_npc_messages"):
                lines.append("\nRecent NPC Messages:")
                for msg in context["recent_npc_messages"][-3:]:
                    lines.append(f"  [{msg.get('source', 'unknown')}] {msg.get('content', '')[:100]}")
            return "\n".join(lines)

        elif agent_type == "rag":
            lines = ["=== RAG Query Context ==="]
            lines.append(f"Query: {context.get('query', '')}")
            if context.get("player_location"):
                loc = context["player_location"]
                lines.append(f"Player Location: ({loc.get('q', 0)}, {loc.get('r', 0)})")
            return "\n".join(lines)

        # Fallback: return JSON-like string
        import json

        return json.dumps(context, indent=2)

