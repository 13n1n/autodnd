"""Game Master Agent using LangChain."""

import uuid
import logging
from datetime import datetime
from typing import TYPE_CHECKING, Callable, Optional

from langchain.agents import create_agent
from langchain.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.tools import StructuredTool
from langchain_openai import ChatOpenAI
from langchain_ollama import ChatOllama
from langgraph.graph.state import CompiledStateGraph

from autodnd.config import (
    DEFAULT_GAME_MASTER_MESSAGE_HISTORY_LIMIT,
    DEFAULT_GAME_MASTER_TEMPERATURE,
    DEFAULT_OLLAMA_BASE_URL,
    DEFAULT_LLM_MODEL,
    DEFAULT_LLM_NUM_CTX,
)
from autodnd.models.messages import Message, MessageSource, MessageType

if TYPE_CHECKING:
    from autodnd.engine.game_engine import GameEngine


class GameMasterAgent:
    """Game Master Agent that interprets player actions and generates responses."""

    def __init__(
        self,
        llm: Optional[BaseChatModel] = None,
        tools: Optional[list[StructuredTool]] = None,
        engine_getter: Optional["Callable[[], GameEngine]"] = None,
        system_prompt: Optional[str] = None,
    ) -> None:
        """
        Initialize Game Master Agent.

        Args:
            llm: LangChain LLM instance (if None, will be created with defaults)
            tools: List of tools for the agent
            engine_getter: Function to get current game engine instance
            system_prompt: Custom system prompt for the game master (optional)
        """
        self._engine_getter = engine_getter
        self._llm = llm or self._create_default_llm()
        self._tools = tools or []
        self._custom_prompt = system_prompt
        self._agent: CompiledStateGraph | None = None
        self._build_agent()

    def _create_default_llm(self) -> ChatOllama:
        """Create default LLM instance."""
        return ChatOllama(
            model=DEFAULT_LLM_MODEL,
            temperature=DEFAULT_GAME_MASTER_TEMPERATURE,
            base_url=DEFAULT_OLLAMA_BASE_URL.rstrip("/"),  # Remove trailing slash for consistency
            num_ctx=DEFAULT_LLM_NUM_CTX,
        )

    def _build_agent(self) -> None:
        """Build the agent using create_agent."""

        system_prompt = f"""
You are a Dungeon Master for a D&D game. Follow this TODO list for EVERY player action:

=== MANDATORY TODO LIST (Follow in order) ===

[ ] STEP 1: ALWAYS check player's current position FIRST
    → Use tool: get_map_state() (call without parameters to get all map info)
    → Extract player_position from the response: {{"q": X, "r": Y}}
    → Remember: Player is at hex coordinates (q, r) - this is their ACTUAL location

[ ] STEP 2: Get player context (if needed for the action)
    → Use tool: get_player_stats(player_id) to check stats, level, health
    → Use tool: get_inventory(player_id) to check items and equipment
    → Use tool: get_map_state(coordinate_q=X, coordinate_r=Y) to check specific cell details

[ ] STEP 3: Generate response that RESPECTS player position
    → Your description MUST be consistent with player being at hex (q, r)
    → If describing the location, describe what's at hex (q, r), not somewhere else
    → If generating world content, ensure it matches the terrain/contents at hex (q, r)
    → NEVER assume player is at a different location than their actual position

[ ] STEP 4: Use other tools as needed
    → Use roll_dice() for game mechanics when appropriate
    → Use store_data() for hidden objectives, game notes (don't reveal hidden objectives directly)
    → Use get_data() to retrieve stored information

[ ] STEP 5: Respond as Dungeon Master
    → Be creative, descriptive, and engaging
    → Follow D&D-like rules and mechanics
    → Keep responses concise but immersive
    → Always respond in character as the Dungeon Master

=== AVAILABLE TOOLS ===
- get_map_state() - MANDATORY: Use this FIRST to get player position (player_position field)
- get_player_stats(player_id) - Get player stats, level, health, status
- get_inventory(player_id) - Get player inventory and equipment
- roll_dice(dice_type, modifier, count) - Roll DnD dice
- store_data(key, value) - Store hidden objectives, game notes
- get_data(key) - Retrieve stored data
- take_item(item_id, ...) - Take an item from a location and add to player inventory.
                            Stores to first available bag slot. Can optionally equip without storing to bag with wear=true.
                            Respect tags, so quest items should be taken with `quest` tag.

=== CRITICAL RULES ===
1. NEVER generate content without first checking player position via get_map_state()
2. ALWAYS respect the player_position from get_map_state() response
3. Descriptions must match the actual hex cell where player is located
4. Do NOT reveal hidden objectives directly - use store_data() and reveal through gameplay

=== PLAYER PREFERENCES ===
{self._custom_prompt if self._custom_prompt else "None specified"}

Remember: Check player position FIRST, then generate content that matches that location!""".strip()

        self._agent = create_agent(
            model=self._llm,
            tools=self._tools,
            system_prompt=system_prompt,
        )

    def update_llm(self, llm: ChatOpenAI) -> None:
        """Update the LLM instance (for hot-reconfiguration)."""
        self._llm = llm
        self._build_agent()
    
    def update_system_prompt(self, system_prompt: Optional[str]) -> None:
        """Update the system prompt (for hot-reconfiguration)."""
        self._custom_prompt = system_prompt
        self._build_agent()

    def process_action(self, action_description: str, message_history: list[dict]) -> tuple[str, list[Message]]:
        """
        Process a player action and generate a response.

        Args:
            action_description: Description of player action
            message_history: Recent message history for context

        Returns:
            Tuple of (response_text, list_of_tool_messages)
        """
        # Convert message history to LangChain message format
        messages = []
        for msg in message_history[-DEFAULT_GAME_MASTER_MESSAGE_HISTORY_LIMIT:]:  # Last N messages for context
            if msg["source"] == "player":
                messages.append({"role": "user", "content": msg["content"]})
            elif msg["source"] == "master":
                messages.append({"role": "assistant", "content": msg["content"]})

        # Add current action
        messages.append({"role": "user", "content": action_description})

        # Invoke agent
        result = self._agent.invoke({"messages": messages})

        # Extract tool calls and final response
        response, tool_messages = self._extract_tool_calls_from_result(result)

        if not response:
            response = result.get("output", "Master acknowledges your action.")

        return response, tool_messages

    def generate_story_continuation(self, context: str) -> tuple[str, list[Message]]:
        """
        Generate story continuation based on context.

        Args:
            context: Current game context/situation

        Returns:
            Tuple of (story_continuation_text, list_of_tool_messages)
        """
        messages = [{"role": "user", "content": f"Continue the story based on this context: {context}"}]

        result = self._agent.invoke({"messages": messages})

        # Extract tool calls and final response
        continuation, tool_messages = self._extract_tool_calls_from_result(result)

        if not continuation:
            continuation = result.get("output", "Story continues...")

        return continuation, tool_messages

    def _extract_tool_calls_from_result(self, result: dict, sequence_start: int = 0) -> tuple[str, list[Message]]:
        """
        Extract tool calls and final response from agent result.

        Args:
            result: Agent invocation result
            sequence_start: Starting sequence number for messages

        Returns:
            Tuple of (final_response_text, list_of_tool_messages)
        """
        tool_messages: list[Message] = []
        sequence = sequence_start
        final_response = ""

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
                            tool_id = tool_call.get("id", str(uuid.uuid4()))
                        else:
                            tool_name = getattr(tool_call, "name", "unknown")
                            tool_args = getattr(tool_call, "args", {})
                            tool_id = getattr(tool_call, "id", str(uuid.uuid4()))

                        # Create tool call message
                        tool_call_msg = Message(
                            message_id=str(uuid.uuid4()),
                            timestamp=datetime.now(),
                            sequence_number=sequence,
                            source=MessageSource.TOOL,
                            source_id=tool_name,
                            content=f"Tool call: {tool_name}({tool_args})",
                            message_type=MessageType.TOOL_OUTPUT,
                            tool_name=tool_name,
                            metadata={"tool_call_id": tool_id, "tool_args": tool_args},
                        )
                        tool_messages.append(tool_call_msg)
                        sequence += 1

                # Check if this is a tool result message (ToolMessage)
                if isinstance(msg, ToolMessage):
                    tool_name_attr = getattr(msg, "name", None)
                    tool_content = getattr(msg, "content", None)
                    tool_call_id_attr = getattr(msg, "tool_call_id", None)
                    # This is a tool result
                    content_str = str(tool_content) if tool_content else ""
                    tool_result_msg = Message(
                        message_id=str(uuid.uuid4()),
                        timestamp=datetime.now(),
                        sequence_number=sequence,
                        source=MessageSource.TOOL,
                        source_id=tool_name_attr,
                        content=content_str,
                        message_type=MessageType.TOOL_OUTPUT,
                        tool_name=tool_name_attr,
                        metadata={"tool_result": True, "tool_call_id": tool_call_id_attr or ""},
                    )
                    tool_messages.append(tool_result_msg)
                    sequence += 1

        final_response = [
            msg.content for msg in result.get("messages", [])
            if isinstance(msg, AIMessage)
        ][-1]

        return final_response, tool_messages

    def generate_initial_intro(
        self,
        difficulty: Optional[str] = None,
        custom_prompt: Optional[str] = None,
    ) -> tuple[str, list[Message]]:
        """
        Generate initial game introduction message.

        This message should include:
        - Game setting and world description
        - Initial conflict or situation to resolve
        - Objectives (may be hidden/unclear, requiring investigation)
        - Starting context for the player

        Args:
            difficulty: Game difficulty level
            custom_prompt: Optional custom prompt from player

        Returns:
            Tuple of (intro_message_text, list_of_tool_messages)
        """
        
        intro_prompt = f"""As the Dungeon Master, complete this TODO list to initialize the game:

[ ] PROMISE 1: Generate an engaging title
    → Use the first line of your final response (after all tool calls) as the game title
    → Make it dramatic, memorable, and fitting for the adventure
    → Format: Start your response with the title as the first line

[ ] PROMISE 2: Plot the story line with hidden objectives
    → Create a main story arc with 3-5 key plot points
    → Design 2-3 hidden objectives that players must discover through gameplay
    → Use store_data() to save hidden objectives (key: "hidden_objectives")
    → Use store_data() to save story arc (key: "story_arc")
    → Ensure objectives are discoverable but not immediately obvious

[ ] PROMISE 3: Establish the initial scene
    → Check player position using get_map_state()
    → Describe the starting location vividly and immersively
    → Set the mood and atmosphere
    → Introduce immediate surroundings and any visible elements

[ ] PROMISE 4: Create engaging introduction
    → Write a compelling opening narrative
    → Introduce the world setting and context
    → Hint at the adventure ahead without revealing hidden objectives
    → Make it feel like the beginning of an epic story

[ ] PROMISE 5: Set up game mechanics context
    → Ensure player stats are appropriate for the difficulty level
    → Consider any starting items or equipment that fit the theme
    → Establish any special rules or mechanics for this adventure

[ ] PROMISE 6: Generate and list a set of starting quest items
    → Use store_data() to save quest items (key: "quest_items")
    → Then taking items with take_item() should be with `quest` tag

Game theme: {custom_prompt or "general D&D story"}
Difficulty: {difficulty}

Remember: Your first line will become the game title. Make it count!"""

        messages = [{"role": "user", "content": intro_prompt}]

        tool_messages: list[Message] = []
        intro_message = None
        while not intro_message:
            langchain_tool_messages = []
            # Convert tool_messages to LangChain format
            for msg in tool_messages:
                if msg.message_type == MessageType.TOOL_OUTPUT:
                    # Check if this is a tool result (has tool_result in metadata)
                    if msg.metadata.get("tool_result"):
                        # This is a tool result - convert back to ToolMessage
                        tool_call_id = msg.metadata.get("tool_call_id", "")
                        tool_name = msg.tool_name or msg.source_id
                        if tool_call_id:  # Only add if we have a valid tool_call_id
                            langchain_tool_messages.append(
                                ToolMessage(
                                    content=msg.content,
                                    tool_call_id=tool_call_id,
                                    name=tool_name,
                                )
                            )
                    # Tool calls are already embedded in AIMessage, so we don't need to convert them

            # Convert pleaseure to LangChain format
            pleaseure_messages = []
            if intro_message is None:
                pleaseure_messages = []
            else:
                pleaseure_messages = [HumanMessage(content="Please, complete the intro message")]

            result = self._agent.invoke({"messages": messages + langchain_tool_messages + pleaseure_messages})

            # Extract tool calls and final response
            intro_message, new_tool_messages = self._extract_tool_calls_from_result(result)
            tool_messages.extend(new_tool_messages)

        return intro_message, tool_messages

