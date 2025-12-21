"""RAG Agent using LangChain for retrieving world lore and game information."""

from typing import Callable, Optional

from langchain.agents import create_agent
from langchain.chat_models import BaseChatModel
from langchain_core.tools import StructuredTool
from langchain_ollama import ChatOllama
from pydantic import BaseModel, Field

from autodnd.engine.game_engine import GameEngine


class QueryLoreInput(BaseModel):
    """Input for query lore tool."""

    query: str = Field(description="Query string to search for in world lore")
    category: Optional[str] = Field(
        default=None, description="Category filter (e.g., 'enemy', 'location', 'item', 'lore')"
    )


class RAGAgent:
    """RAG Agent that retrieves world lore, enemy info, and setting details from vector store."""

    def __init__(
        self,
        llm: Optional[BaseChatModel] = None,
        engine_getter: Optional[Callable[[], GameEngine]] = None,
        vector_store: Optional[object] = None,  # Will be Chroma/FAISS vector store
    ) -> None:
        """
        Initialize RAG Agent.

        Args:
            llm: LangChain LLM instance (if None, will be created with defaults)
            engine_getter: Function to get current game engine instance
            vector_store: Vector store for RAG retrieval (optional, can be added later)
        """
        self._engine_getter = engine_getter
        self._llm = llm or self._create_default_llm()
        self._vector_store = vector_store
        self._tools = []
        self._agent = None
        self._build_agent()

    def _create_default_llm(self) -> ChatOllama:
        """Create default LLM instance."""
        return ChatOllama(
            model="gpt-oss:20b",
            temperature=0.3,
            base_url="http://localhost:11434",
            num_ctx=2**15,
        )

    def _build_agent(self) -> None:
        """Build the agent using create_agent."""

        # Create query lore tool (basic implementation, can be extended with vector store)
        def query_lore_tool(query: str, category: Optional[str] = None) -> dict:
            """Query world lore and game information."""
            # TODO: Implement actual vector store retrieval
            # For now, return a placeholder response
            if self._vector_store is None:
                return {
                    "results": [],
                    "message": "Vector store not initialized. RAG functionality not available.",
                    "query": query,
                    "category": category,
                }

            # Placeholder for future vector store integration
            # When implemented, this will:
            # 1. Query the vector store with the query string
            # 2. Filter by category if provided
            # 3. Return relevant documents/chunks
            return {
                "results": [],
                "message": "Vector store integration pending",
                "query": query,
                "category": category,
            }

        query_tool = StructuredTool.from_function(
            func=query_lore_tool,
            name="query_lore",
            description="Query world lore, enemy information, setting details, or game content. "
            "Optionally filter by category (enemy, location, item, lore).",
            args_schema=QueryLoreInput,
        )

        self._tools = [query_tool]

        system_prompt = """
You are a RAG (Retrieval-Augmented Generation) Agent for a D&D game. Your role is to:
1. Retrieve relevant world lore and game information
2. Answer questions about game content (enemies, locations, items, lore)
3. Provide context from the game's knowledge base
4. Help other agents access game information

When querying:
- Use the query_lore tool to search for information
- Be specific with queries to get relevant results
- Filter by category when appropriate
- Provide clear, concise summaries of retrieved information

Always use the query_lore tool to access game information.
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

    def set_vector_store(self, vector_store: object) -> None:
        """Set the vector store for RAG retrieval."""
        self._vector_store = vector_store
        self._build_agent()  # Rebuild agent to update tool

    def query(self, query: str, category: Optional[str] = None) -> dict:
        """
        Query world lore and game information.

        Args:
            query: Query string to search for
            category: Optional category filter

        Returns:
            Dictionary with query results
        """
        messages = [{"role": "user", "content": f"Query: {query}" + (f" (Category: {category})" if category else "")}]

        result = self._agent.invoke({"messages": messages})

        # Extract response
        if result.get("messages"):
            last_message = result["messages"][-1]
            if hasattr(last_message, "content"):
                content = last_message.content
                if isinstance(content, str):
                    return {"query": query, "category": category, "response": content}
                elif isinstance(content, list):
                    return {"query": query, "category": category, "response": " ".join(str(item) for item in content)}
            elif isinstance(last_message, dict):
                content = last_message.get("content", "")
                return {"query": query, "category": category, "response": content}

        return {"query": query, "category": category, "response": result.get("output", "No results found.")}

