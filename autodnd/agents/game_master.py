"""Game Master Agent using LangChain."""

from typing import Callable, Optional

from langchain.agents import create_agent
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.tools import StructuredTool
from langchain_openai import ChatOpenAI

from autodnd.engine.game_engine import GameEngine
from autodnd.models.messages import MessageSource, MessageType


class GameMasterAgent:
    """Game Master Agent that interprets player actions and generates responses."""

    def __init__(
        self,
        llm: Optional[ChatOpenAI] = None,
        tools: Optional[list[StructuredTool]] = None,
        engine_getter: Optional[Callable[[], GameEngine]] = None,
    ) -> None:
        """
        Initialize Game Master Agent.

        Args:
            llm: LangChain LLM instance (if None, will be created with defaults)
            tools: List of tools for the agent
            engine_getter: Function to get current game engine instance
        """
        self._engine_getter = engine_getter
        self._llm = llm or self._create_default_llm()
        self._tools = tools or []
        self._agent = None
        self._build_agent()

    def _create_default_llm(self) -> ChatOpenAI:
        """Create default LLM instance."""
        return ChatOpenAI(
            model="gpt-4o-mini",
            temperature=0.7,
        )

    def _build_agent(self) -> None:
        """Build the agent using create_agent."""
        system_prompt = """You are a Dungeon Master for a D&D game. Your role is to:
1. Interpret player actions and describe what happens
2. Create engaging narratives and storylines
3. Manage NPCs and world events
4. Make fair rulings based on D&D rules
5. Keep the game interesting and fun

You have access to tools to:
- Roll dice for game mechanics
- Get player stats and inventory
- Get map information

When responding to player actions:
- Be creative and descriptive
- Follow D&D rules and mechanics
- Keep responses concise but engaging
- Use tools when needed for dice rolls or checking game state

Always respond in character as the Dungeon Master."""

        self._agent = create_agent(
            model=self._llm,
            tools=self._tools,
            system_prompt=system_prompt,
        )

    def update_llm(self, llm: ChatOpenAI) -> None:
        """Update the LLM instance (for hot-reconfiguration)."""
        self._llm = llm
        self._build_agent()

    def process_action(self, action_description: str, message_history: list[dict]) -> str:
        """
        Process a player action and generate a response.

        Args:
            action_description: Description of player action
            message_history: Recent message history for context

        Returns:
            Game Master response text
        """
        # Convert message history to LangChain message format
        messages = []
        for msg in message_history[-10:]:  # Last 10 messages for context
            if msg["source"] == "player":
                messages.append({"role": "user", "content": msg["content"]})
            elif msg["source"] == "master":
                messages.append({"role": "assistant", "content": msg["content"]})

        # Add current action
        messages.append({"role": "user", "content": action_description})

        # Invoke agent
        result = self._agent.invoke({"messages": messages})

        # Extract the last message content from the response
        # The result contains messages array, get the last one (should be AIMessage)
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

    def generate_story_continuation(self, context: str) -> str:
        """
        Generate story continuation based on context.

        Args:
            context: Current game context/situation

        Returns:
            Story continuation text
        """
        messages = [{"role": "user", "content": f"Continue the story based on this context: {context}"}]

        result = self._agent.invoke({"messages": messages})

        # Extract the last message content from the response
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

        return result.get("output", "Story continues...")


