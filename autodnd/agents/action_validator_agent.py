"""Action Validator Agent using LangChain for validating player actions."""

from typing import TYPE_CHECKING, Callable, Optional

from langchain.agents import create_agent
from langchain.chat_models import BaseChatModel
from langchain_core.tools import StructuredTool
from langchain_openai import ChatOpenAI
from langchain_ollama import ChatOllama
from pydantic import BaseModel, Field

from autodnd.engine.action_validator import ActionValidator
from autodnd.models.actions import Action

if TYPE_CHECKING:
    from autodnd.engine.game_engine import GameEngine


class ValidateActionInput(BaseModel):
    """Input for validate action tool."""

    action_type: str = Field(description="Type of action (e.g., 'move', 'attack', 'use_item')")
    parameters: dict = Field(description="Action parameters as a dictionary")
    player_id: str = Field(description="Player ID performing the action")


class ActionValidatorAgent:
    """Action Validator Agent that validates player actions against game rules."""

    def __init__(
        self,
        llm: Optional[BaseChatModel] = None,
        engine_getter: Optional["Callable[[], GameEngine]"] = None,
    ) -> None:
        """
        Initialize Action Validator Agent.

        Args:
            llm: LangChain LLM instance (if None, will be created with defaults)
            engine_getter: Function to get current game engine instance
        """
        self._engine_getter = engine_getter
        self._llm = llm or self._create_default_llm()
        self._action_validator = ActionValidator()
        self._tools = []
        self._agent = None
        self._build_agent()

    def _create_default_llm(self) -> ChatOllama:
        """Create default LLM instance."""
        return ChatOllama(
            model="gpt-oss:20b",
            temperature=0.1,  # Low temperature for consistent validation
            base_url="http://localhost:11434",
            num_ctx=2**15,
        )

    def _build_agent(self) -> None:
        """Build the agent using create_agent."""

        # Create validation tool
        def validate_action_tool(action_type: str, parameters: dict, player_id: str) -> dict:
            """Validate an action using the ActionValidator."""
            try:
                from autodnd.models.actions import ActionType

                # Convert string to ActionType enum
                action_type_enum = ActionType[action_type.upper()]
                action = Action(
                    action_type=action_type_enum,
                    parameters=parameters,
                    player_id=player_id,
                )

                if self._engine_getter:
                    engine = self._engine_getter()
                    state = engine.get_state()
                    is_valid, error_message = self._action_validator.validate_action(action, state)
                    return {
                        "is_valid": is_valid,
                        "error_message": error_message,
                        "action_type": action_type,
                        "player_id": player_id,
                    }
                else:
                    return {
                        "is_valid": False,
                        "error_message": "Engine getter not available",
                        "action_type": action_type,
                        "player_id": player_id,
                    }
            except (KeyError, ValueError) as e:
                return {
                    "is_valid": False,
                    "error_message": f"Invalid action type: {str(e)}",
                    "action_type": action_type,
                    "player_id": player_id,
                }

        validate_tool = StructuredTool.from_function(
            func=validate_action_tool,
            name="validate_action",
            description="Validate a player action against game rules. Returns is_valid (bool) and error_message (str if invalid)",
            args_schema=ValidateActionInput,
        )

        self._tools = [validate_tool]

        system_prompt = """
You are an Action Validator for a D&D game. Your role is to:
1. Validate player actions against game rules
2. Check if actions are allowed in the current game state
3. Provide clear error messages when actions are invalid
4. Ensure game rules are followed consistently

When validating actions:
- Use the validate_action tool to check actions
- Provide clear, concise validation results
- Explain why an action is invalid if it fails validation
- Be consistent with game rules

Always respond with validation results in a structured format.
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

    def validate_action(self, action: Action) -> tuple[bool, str]:
        """
        Validate an action using the agent.

        Args:
            action: Action to validate

        Returns:
            Tuple of (is_valid, error_message)
        """
        # First, use the direct ActionValidator for fast validation
        if self._engine_getter:
            engine = self._engine_getter()
            state = engine.get_state()
            is_valid, error_message = self._action_validator.validate_action(action, state)
            return is_valid, error_message

        return False, "Engine getter not available"

    def validate_with_explanation(self, action: Action, context: Optional[str] = None) -> dict:
        """
        Validate an action and get an explanation from the agent.

        Args:
            action: Action to validate
            context: Additional context about the action

        Returns:
            Dictionary with validation result and explanation
        """
        # Use direct validator first
        is_valid, error_message = self.validate_action(action)

        # Build prompt for agent explanation
        prompt = f"""
Validate this action and provide an explanation:

Action Type: {action.action_type.value}
Player ID: {action.player_id}
Parameters: {action.parameters}
""".strip()

        if context:
            prompt += f"\n\nContext: {context}"

        if not is_valid:
            prompt += f"\n\nNote: Direct validation failed with error: {error_message}"

        messages = [{"role": "user", "content": prompt}]

        # Invoke agent for explanation
        result = self._agent.invoke({"messages": messages})

        # Extract response
        explanation = ""
        if result.get("messages"):
            last_message = result["messages"][-1]
            if hasattr(last_message, "content"):
                content = last_message.content
                if isinstance(content, str):
                    explanation = content
                elif isinstance(content, list):
                    explanation = " ".join(str(item) for item in content)
            elif isinstance(last_message, dict):
                explanation = last_message.get("content", "")

        return {
            "is_valid": is_valid,
            "error_message": error_message if not is_valid else "",
            "explanation": explanation or "Action validated.",
        }

