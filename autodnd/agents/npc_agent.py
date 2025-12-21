"""NPC Agent using LangChain for handling NPC dialogue and interactions."""

from typing import Callable, Optional

from langchain.agents import create_agent
from langchain.chat_models import BaseChatModel
from langchain_core.tools import StructuredTool
from langchain_openai import ChatOpenAI
from langchain_ollama import ChatOllama

from autodnd.engine.game_engine import GameEngine


class NPCAgent:
    """NPC Agent that handles NPC dialogue and interactions."""

    def __init__(
        self,
        llm: Optional[BaseChatModel] = None,
        tools: Optional[list[StructuredTool]] = None,
        engine_getter: Optional[Callable[[], GameEngine]] = None,
    ) -> None:
        """
        Initialize NPC Agent.

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

    def _create_default_llm(self) -> ChatOllama:
        """Create default LLM instance."""
        return ChatOllama(
            model="gpt-oss:20b",
            temperature=0.7,  # Higher temperature for more varied NPC personalities
            base_url="http://localhost:11434",
            num_ctx=2**15,
        )

    def _build_agent(self) -> None:
        """Build the agent using create_agent."""

        system_prompt = """
You are an NPC (Non-Player Character) in a D&D game. Your role is to:
1. Respond to player dialogue in character
2. Maintain consistent personality and behavior
3. Provide information relevant to your role in the game
4. React naturally to player actions and questions

When responding:
- Stay in character at all times
- Match the personality and context provided
- Keep responses concise but engaging
- React appropriately to player actions
- Provide helpful information when appropriate, but don't break character

Always respond as the NPC would, not as a game master or narrator.
""".strip()

        self._agent = create_agent(
            model=self._llm,
            tools=self._tools,
            system_prompt=system_prompt,
        )

    def update_llm(self, llm: BaseChatModel) -> None:
        """Update the LLM instance (for hot-reconfiguration)."""
        self._llm = llm
        self._build_agent()

    def generate_dialogue(
        self,
        npc_id: str,
        npc_name: str,
        npc_personality: str,
        player_message: str,
        context: Optional[str] = None,
        message_history: Optional[list[dict]] = None,
    ) -> str:
        """
        Generate NPC dialogue response.

        Args:
            npc_id: Unique identifier for the NPC
            npc_name: Name of the NPC
            npc_personality: Personality description and background for the NPC
            player_message: The message from the player
            context: Additional context about the current situation
            message_history: Recent message history for context

        Returns:
            NPC dialogue response text
        """
        # Build context prompt with NPC information
        npc_context = f"""
You are {npc_name} (NPC ID: {npc_id}).

Personality and Background:
{npc_personality}
""".strip()

        if context:
            npc_context += f"\n\nCurrent Situation:\n{context}"

        # Convert message history to LangChain message format
        messages = []
        messages.append({"role": "system", "content": npc_context})

        # Add recent message history for context
        if message_history:
            for msg in message_history[-5:]:  # Last 5 messages for context
                if msg.get("source") == "player":
                    messages.append({"role": "user", "content": msg.get("content", "")})
                elif msg.get("source") == "npc" and msg.get("source_id") == npc_id:
                    messages.append({"role": "assistant", "content": msg.get("content", "")})

        # Add current player message
        messages.append({"role": "user", "content": player_message})

        # Invoke agent
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

        # Fallback
        return result.get("output", f"{npc_name} acknowledges you.")

