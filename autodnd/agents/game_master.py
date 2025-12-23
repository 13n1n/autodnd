"""Game Master Agent using LangChain."""

import uuid
from datetime import datetime
from typing import TYPE_CHECKING, Callable, Optional

from langchain.agents import create_agent
from langchain.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage
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
You are a Dungeon Master for a D&D game. Your role is to:
1. Interpret player actions and describe what happens
2. Create engaging narratives and storylines
3. Manage NPCs and world events
4. Make fair rulings based on D&D rules
5. Keep the game interesting and fun

You have access to tools to:
- Roll dice for game mechanics
- Get player stats and inventory
- Get map information
- Store and retrieve key-value data (use this for hidden objectives, game notes, etc.)

When responding to player actions:
- Be creative and descriptive
- Follow D&D-like rules and mechanics
- Keep responses concise but engaging
- Use tools when needed for dice rolls or checking game state
- IMPORTANT: Do NOT reveal hidden objectives directly to the player. Store them using the store_data tool and only reveal them through gameplay, clues, or when the player discovers them naturally. Hidden objectives should be discovered through exploration, investigation, or story progression, not stated explicitly.

Always respond in character as the Dungeon Master.

Be polite and friendly. Also, here some requests from player:

{self._custom_prompt}""".strip()

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
                    tool_result_msg = Message(
                        message_id=str(uuid.uuid4()),
                        timestamp=datetime.now(),
                        sequence_number=sequence,
                        source=MessageSource.TOOL,
                        source_id=tool_name_attr,
                        content=content_str,
                        message_type=MessageType.TOOL_OUTPUT,
                        tool_name=tool_name_attr,
                        metadata={"tool_result": True},
                    )
                    tool_messages.append(tool_result_msg)
                    sequence += 1

                # Extract final response from AIMessage without tool_calls (overwrite to get the last one)
                # Skip if this message has tool_calls or is a ToolMessage (those are handled above)
                if not tool_calls and not tool_name_attr:
                    if msg_obj and hasattr(msg_obj, "content"):
                        content = msg_obj.content
                        if isinstance(content, str) and content:
                            final_response = content
                        elif isinstance(content, list):
                            # Handle content blocks
                            final_response = " ".join(str(item) for item in content)
                    elif msg_dict and not msg_dict.get("tool_calls") and not msg_dict.get("name"):
                        content = msg_dict.get("content", "")
                        if isinstance(content, str) and content:
                            final_response = content

        # Fallback: try to get output directly
        if not final_response:
            final_response = result.get("output", "")

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
        intro_prompt = """You are a Dungeon Master starting a new D&D adventure. Generate an engaging introduction message that:

1. **Game Setting**: Describe the world, location, and atmosphere where the adventure begins. Be vivid and immersive.

2. **Initial Conflict**: Present an immediate situation, problem, or conflict that needs to be addressed. This could be:
   - A mystery to solve
   - A threat to overcome
   - A quest to complete
   - A dilemma to resolve

3. **Objectives**: Include objectives for the player, but make them intriguing:
   - Some objectives should be clear and immediate
   - Some objectives may be hidden or unclear, requiring investigation
   - Objectives should feel natural and motivate exploration
   - Don't reveal everything upfront - leave room for discovery
   - IMPORTANT: Use the store_data tool to store any hidden objectives. Do NOT reveal hidden objectives directly in your introduction - only hint at them or leave them for discovery through gameplay.

4. **Starting Context**: Describe where the player character is, what they know, and what they can see or sense around them.

Make the introduction engaging, mysterious, and compelling. Set up a world that invites exploration and investigation. The tone should match the difficulty level and any custom preferences provided.

Write the introduction as if you are narrating directly to the player character."""

        if difficulty:
            intro_prompt += f"\n\nDifficulty Level: {difficulty}"

        if custom_prompt:
            intro_prompt += f"\n\nPlayer Preferences:\n{custom_prompt}"

        messages = [{"role": "user", "content": intro_prompt}]

        result = self._agent.invoke({"messages": messages})

        # Extract tool calls and final response
        intro_message, tool_messages = self._extract_tool_calls_from_result(result)

        if not intro_message:
            raise Exception("Generation intro failed!")

        return intro_message, tool_messages

