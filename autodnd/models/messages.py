"""Message and message history models."""

from datetime import datetime
from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, ConfigDict, Field


class MessageSource(str, Enum):
    """Source of a message."""

    PLAYER = "player"
    MASTER = "master"
    NPC = "npc"
    TOOL = "tool"
    SYSTEM = "system"


class MessageType(str, Enum):
    """Type of message."""

    ACTION = "action"
    RESPONSE = "response"
    DIALOGUE = "dialogue"
    TOOL_OUTPUT = "tool_output"
    SYSTEM = "system"


class Message(BaseModel):
    """Complete message history entry."""

    model_config = ConfigDict(frozen=True)  # Immutable model

    message_id: str = Field(description="Unique message identifier")
    timestamp: datetime = Field(description="Message timestamp")
    sequence_number: int = Field(ge=0, description="Order of messages")

    # Message source
    source: MessageSource = Field(description="Message source")
    source_id: Optional[str] = Field(default=None, description="player_id, npc_id, tool_name, etc.")

    # Message content
    content: str = Field(description="The actual message text")
    message_type: MessageType = Field(description="Message type")

    # Context
    action_id: Optional[str] = Field(default=None, description="If related to a specific action")
    tool_name: Optional[str] = Field(default=None, description="If from a tool")
    npc_id: Optional[str] = Field(default=None, description="If from NPC")

    # Metadata
    metadata: dict[str, Any] = Field(
        default_factory=dict, description="Additional context (dice rolls, validation results, etc.)"
    )


class MessageHistory(BaseModel):
    """Ordered list of all messages."""

    model_config = ConfigDict(frozen=True)  # Immutable model

    messages: list[Message] = Field(default_factory=list, description="All messages in order")

    def add_message(self, message: Message) -> "MessageHistory":
        """Create new history with added message (immutable update)."""
        new_messages = self.messages + [message]
        return self.model_copy(update={"messages": new_messages})

