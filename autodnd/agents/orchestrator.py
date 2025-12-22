"""Agent orchestrator for coordinating multi-agent interactions."""

import logging
import uuid
from datetime import datetime
from typing import TYPE_CHECKING, Any, Callable, Optional

from autodnd.agents.game_master import GameMasterAgent
from autodnd.agents.npc_agent import NPCAgent
from autodnd.agents.rag_agent import RAGAgent
from autodnd.agents.state_serializer import StateSerializer
from autodnd.models.messages import Message, MessageSource, MessageType
from autodnd.models.state import GameState

if TYPE_CHECKING:
    from autodnd.engine.game_engine import GameEngine

logger = logging.getLogger(__name__)


class AgentOrchestrator:
    """Orchestrates agent interactions and state passing."""

    def __init__(
        self,
        game_master: GameMasterAgent,
        npc_agent: Optional[NPCAgent] = None,
        rag_agent: Optional[RAGAgent] = None,
        engine_getter: Optional["Callable[[], GameEngine]"] = None,
        max_retries: int = 2,
    ) -> None:
        """
        Initialize agent orchestrator.

        Args:
            game_master: Game Master agent instance
            npc_agent: Optional NPC agent instance
            rag_agent: Optional RAG agent instance
            engine_getter: Function to get current game engine instance
            max_retries: Maximum retry attempts for agent calls
        """
        self._game_master = game_master
        self._npc_agent = npc_agent
        self._rag_agent = rag_agent
        self._engine_getter = engine_getter
        self._max_retries = max_retries

    def process_player_action(
        self, action_text: str, state: GameState, player_id: Optional[str] = None
    ) -> tuple[str, list[Message]]:
        """
        Process player action through agent orchestration.

        Flow:
        1. Game Master processes action (may call RAG for lore)
        2. If NPC interaction detected, call NPC agent
        3. Log all agent interactions as Messages
        4. Return master response and all generated messages

        Args:
            action_text: Player action text
            state: Current game state
            player_id: Optional player ID

        Returns:
            Tuple of (master_response, list_of_messages)
        """
        messages: list[Message] = []
        sequence_start = len(state.message_history.messages)

        try:
            # Step 1: Get context for Game Master
            gm_context = StateSerializer.extract_game_master_context(state, player_id)
            context_str = StateSerializer.format_context_for_agent(gm_context, "game_master")

            # Step 2: Prepare message history for Game Master
            recent_messages = [
                {
                    "source": msg.source.value,
                    "content": msg.content,
                    "message_type": msg.message_type.value,
                }
                for msg in state.message_history.messages[-10:]
            ]

            # Step 3: Call Game Master (with retry logic) and capture tool outputs
            master_response, tool_outputs = self._call_game_master_with_tools(
                action_text, recent_messages, sequence_start + len(messages)
            )

            # Step 4: Log tool outputs first (if any)
            for tool_output in tool_outputs:
                messages.append(tool_output)

            # Step 5: Log Game Master response
            master_message = self._create_message(
                content=master_response,
                source=MessageSource.MASTER,
                message_type=MessageType.RESPONSE,
                sequence_number=sequence_start + len(messages),
            )
            messages.append(master_message)

            # Step 6: Check if NPC interaction is needed
            # (Simple heuristic: if action mentions "talk", "speak", "ask", or NPC name)
            npc_interaction = self._detect_npc_interaction(action_text, master_response, state)
            if npc_interaction and self._npc_agent:
                npc_response = self._handle_npc_interaction(
                    npc_interaction["npc_id"],
                    npc_interaction.get("npc_name", "NPC"),
                    npc_interaction.get("npc_personality", ""),
                    action_text,
                    state,
                    sequence_start + len(messages),
                )
                if npc_response:
                    messages.append(npc_response)

            # Step 7: Check if RAG query is needed
            # (If Game Master response suggests lore/world info needed)
            if self._rag_agent and self._should_query_rag(action_text, master_response):
                rag_messages = self._handle_rag_query(
                    action_text, state, player_id, sequence_start + len(messages)
                )
                messages.extend(rag_messages)

        except Exception as e:
            logger.error(f"Error in agent orchestration: {e}", exc_info=True)
            # Log error as system message
            error_message = self._create_message(
                content=f"Agent orchestration error: {str(e)}",
                source=MessageSource.SYSTEM,
                message_type=MessageType.SYSTEM,
                sequence_number=sequence_start + len(messages),
                metadata={"error": str(e), "error_type": type(e).__name__},
            )
            messages.append(error_message)
            master_response = "I encountered an error processing your action. Please try again."

        return master_response, messages

    def _call_game_master_with_tools(
        self, action_text: str, recent_messages: list[dict], sequence_start: int
    ) -> tuple[str, list[Message]]:
        """
        Call Game Master agent and capture tool outputs.

        Args:
            action_text: Player action text
            recent_messages: Recent message history
            sequence_start: Starting sequence number for messages

        Returns:
            Tuple of (master_response, list_of_tool_messages)
        """
        tool_messages: list[Message] = []
        sequence = sequence_start

        try:
            # Convert message history to LangChain message format
            messages = []
            for msg in recent_messages:
                if msg["source"] == "player":
                    messages.append({"role": "user", "content": msg["content"]})
                elif msg["source"] == "master":
                    messages.append({"role": "assistant", "content": msg["content"]})

            # Add current action
            messages.append({"role": "user", "content": action_text})

            # Invoke agent
            result = self._game_master._agent.invoke({"messages": messages})
            logger.info(f"Game Master result: {result}")

            # Extract tool calls and outputs from agent messages
            if result.get("messages"):
                for msg in result["messages"]:
                    # Handle both message objects and dicts
                    msg_dict = msg if isinstance(msg, dict) else {}
                    msg_obj = msg if not isinstance(msg, dict) else None

                    # Check if this is a tool call message (AIMessage with tool_calls)
                    tool_calls = None
                    if msg_obj and hasattr(msg_obj, "tool_calls") and msg_obj.tool_calls:
                        tool_calls = msg_obj.tool_calls
                    elif msg_dict.get("tool_calls"):
                        tool_calls = msg_dict["tool_calls"]

                    if tool_calls:
                        for tool_call in tool_calls:
                            # Handle both dict and object tool calls
                            if isinstance(tool_call, dict):
                                tool_name = tool_call.get("name", "unknown")
                                tool_args = tool_call.get("args", {})
                            else:
                                tool_name = getattr(tool_call, "name", "unknown")
                                tool_args = getattr(tool_call, "args", {})

                            # Create tool call message
                            tool_call_msg = self._create_message(
                                content=f"Tool call: {tool_name}({tool_args})",
                                source=MessageSource.TOOL,
                                message_type=MessageType.TOOL_OUTPUT,
                                sequence_number=sequence,
                                source_id=tool_name,
                                tool_name=tool_name,
                                metadata={"tool_call": str(tool_call), "tool_args": tool_args},
                            )
                            tool_messages.append(tool_call_msg)
                            sequence += 1

                    # Check if this is a tool result message (ToolMessage)
                    tool_name_attr = None
                    tool_content = None
                    if msg_obj:
                        tool_name_attr = getattr(msg_obj, "name", None)
                        tool_content = getattr(msg_obj, "content", None)
                    elif msg_dict:
                        tool_name_attr = msg_dict.get("name")
                        tool_content = msg_dict.get("content")

                    if tool_name_attr and tool_content:
                        # This is a tool result
                        content_str = str(tool_content)
                        tool_result_msg = self._create_message(
                            content=content_str,
                            source=MessageSource.TOOL,
                            message_type=MessageType.TOOL_OUTPUT,
                            sequence_number=sequence,
                            source_id=tool_name_attr,
                            tool_name=tool_name_attr,
                            metadata={"tool_result": True},
                        )
                        tool_messages.append(tool_result_msg)
                        sequence += 1

            # Extract final response
            master_response = self._extract_agent_response(result)

        except Exception as e:
            logger.error(f"Error calling Game Master with tools: {e}", exc_info=True)
            # Log error as tool message
            error_msg = self._create_message(
                content=f"Game Master tool error: {str(e)}",
                source=MessageSource.SYSTEM,
                message_type=MessageType.SYSTEM,
                sequence_number=sequence,
                metadata={"error": str(e)},
            )
            tool_messages.append(error_msg)
            master_response = "I encountered an error processing your action."

        return master_response, tool_messages

    def _extract_agent_response(self, result: dict[str, Any]) -> str:
        """
        Extract agent response from LangChain agent result.

        Args:
            result: Agent invocation result

        Returns:
            Agent response string
        """
        if result.get("messages"):
            last_message = result["messages"][-1]
            # Handle both AIMessage objects and dicts
            if hasattr(last_message, "content"):
                content = last_message.content
                if isinstance(content, str):
                    return content
                elif isinstance(content, list):
                    # Handle content blocks
                    return " ".join(str(item) for item in content)
            elif isinstance(last_message, dict):
                content = last_message.get("content", "")
                if isinstance(content, str):
                    return content

        # Fallback: try to get output directly
        return result.get("output", "Master acknowledges your action.")

    def _call_with_retry(self, func: Callable[[], str], agent_name: str) -> str:
        """
        Call agent function with retry logic.

        Args:
            func: Function to call
            agent_name: Name of agent (for logging)

        Returns:
            Agent response string

        Raises:
            Exception: If all retries fail
        """
        last_error = None
        for attempt in range(self._max_retries + 1):
            try:
                return func()
            except Exception as e:
                last_error = e
                logger.warning(f"{agent_name} call attempt {attempt + 1} failed: {e}")
                if attempt < self._max_retries:
                    continue
                raise last_error

        # Should not reach here, but satisfy type checker
        raise RuntimeError(f"{agent_name} call failed after {self._max_retries + 1} attempts")

    def _detect_npc_interaction(
        self, action_text: str, master_response: str, state: GameState
    ) -> Optional[dict[str, Any]]:
        """
        Detect if action involves NPC interaction.

        Args:
            action_text: Player action text
            master_response: Game Master response
            state: Current game state

        Returns:
            Dict with npc_id, npc_name, npc_personality if detected, None otherwise
        """
        # Simple heuristic: check for NPC-related keywords
        npc_keywords = ["talk", "speak", "ask", "tell", "say", "greet", "npc"]
        action_lower = action_text.lower()

        if not any(keyword in action_lower for keyword in npc_keywords):
            return None

        # Try to find NPC in nearby cells
        if state.players:
            player_pos = state.players[0].position
            # Check cells within 1 hex distance
            for q_offset in range(-1, 2):
                for r_offset in range(-1, 2):
                    from autodnd.models.world import HexCoordinate

                    coord = HexCoordinate(q=player_pos.q + q_offset, r=player_pos.r + r_offset)
                    cell = state.world_map.get_cell(coord)
                    if cell and cell.contents:
                        # Check if any content looks like an NPC ID
                        for content_id in cell.contents:
                            # Simple heuristic: if content_id contains "npc" or is a UUID-like string
                            if "npc" in content_id.lower() or len(content_id) > 20:
                                return {
                                    "npc_id": content_id,
                                    "npc_name": content_id.replace("_", " ").title(),
                                    "npc_personality": "A friendly NPC in the game world.",
                                }

        return None

    def _handle_npc_interaction(
        self,
        npc_id: str,
        npc_name: str,
        npc_personality: str,
        player_message: str,
        state: GameState,
        sequence_number: int,
    ) -> Optional[Message]:
        """
        Handle NPC interaction.

        Args:
            npc_id: NPC identifier
            npc_name: NPC name
            npc_personality: NPC personality description
            player_message: Player message
            state: Current game state
            sequence_number: Message sequence number

        Returns:
            NPC message or None if error
        """
        if not self._npc_agent:
            return None

        try:
            # Get NPC context
            npc_context = StateSerializer.extract_npc_context(state, npc_id)
            context_str = StateSerializer.format_context_for_agent(npc_context, "npc")

            # Prepare message history
            recent_messages = [
                {
                    "source": msg.source.value,
                    "content": msg.content,
                    "message_type": msg.message_type.value,
                }
                for msg in state.message_history.messages[-5:]
            ]

            # Call NPC agent
            npc_response = self._call_with_retry(
                lambda: self._npc_agent.generate_dialogue(
                    npc_id=npc_id,
                    npc_name=npc_name,
                    npc_personality=npc_personality,
                    player_message=player_message,
                    context=context_str,
                    message_history=recent_messages,
                ),
                "NPC Agent",
            )

            # Create NPC message
            return self._create_message(
                content=npc_response,
                source=MessageSource.NPC,
                message_type=MessageType.DIALOGUE,
                sequence_number=sequence_number,
                npc_id=npc_id,
            )

        except Exception as e:
            logger.error(f"Error in NPC interaction: {e}", exc_info=True)
            return None

    def _should_query_rag(self, action_text: str, master_response: str) -> bool:
        """
        Determine if RAG query is needed.

        Args:
            action_text: Player action text
            master_response: Game Master response

        Returns:
            True if RAG query should be made
        """
        # Simple heuristic: check for lore/world-related keywords
        rag_keywords = ["lore", "history", "information", "about", "what is", "tell me", "enemy", "creature"]
        action_lower = action_text.lower()
        response_lower = master_response.lower()

        # Check if action or response suggests need for world knowledge
        return any(keyword in action_lower or keyword in response_lower for keyword in rag_keywords)

    def _handle_rag_query(
        self, query: str, state: GameState, player_id: Optional[str], sequence_number: int
    ) -> list[Message]:
        """
        Handle RAG query.

        Args:
            query: Query string
            state: Current game state
            player_id: Optional player ID
            sequence_number: Starting sequence number

        Returns:
            List of messages (RAG query result and tool output)
        """
        messages: list[Message] = []

        if not self._rag_agent:
            return messages

        try:
            # Get RAG context
            rag_context = StateSerializer.extract_rag_context(state, query, player_id)

            # Call RAG agent (through its query_lore method if available)
            # Note: RAG agent's query_lore is typically called via tool, but we can call it directly
            # For now, we'll create a tool output message
            # In a full implementation, we'd call the RAG agent's query method

            # Create a placeholder message for RAG tool output
            # In practice, this would call the actual RAG retrieval
            rag_result = "RAG query executed (integration pending)"
            tool_message = self._create_message(
                content=rag_result,
                source=MessageSource.TOOL,
                message_type=MessageType.TOOL_OUTPUT,
                sequence_number=sequence_number,
                source_id="query_lore",
                tool_name="query_lore",
                metadata={"query": query, "context": rag_context},
            )
            messages.append(tool_message)

        except Exception as e:
            logger.error(f"Error in RAG query: {e}", exc_info=True)
            error_message = self._create_message(
                content=f"RAG query error: {str(e)}",
                source=MessageSource.SYSTEM,
                message_type=MessageType.SYSTEM,
                sequence_number=sequence_number,
                metadata={"error": str(e)},
            )
            messages.append(error_message)

        return messages

    def _create_message(
        self,
        content: str,
        source: MessageSource,
        message_type: MessageType,
        sequence_number: int,
        source_id: Optional[str] = None,
        tool_name: Optional[str] = None,
        npc_id: Optional[str] = None,
        metadata: Optional[dict[str, Any]] = None,
    ) -> Message:
        """
        Create a message object.

        Args:
            content: Message content
            source: Message source
            message_type: Message type
            sequence_number: Sequence number
            source_id: Optional source ID
            tool_name: Optional tool name
            npc_id: Optional NPC ID
            metadata: Optional metadata

        Returns:
            Message object
        """
        return Message(
            message_id=str(uuid.uuid4()),
            timestamp=datetime.now(),
            sequence_number=sequence_number,
            source=source,
            source_id=source_id,
            content=content,
            message_type=message_type,
            tool_name=tool_name,
            npc_id=npc_id,
            metadata=metadata or {},
        )

    def update_game_master(self, game_master: GameMasterAgent) -> None:
        """Update Game Master agent instance."""
        self._game_master = game_master

    def update_npc_agent(self, npc_agent: Optional[NPCAgent]) -> None:
        """Update NPC agent instance."""
        self._npc_agent = npc_agent

    def update_rag_agent(self, rag_agent: Optional[RAGAgent]) -> None:
        """Update RAG agent instance."""
        self._rag_agent = rag_agent

