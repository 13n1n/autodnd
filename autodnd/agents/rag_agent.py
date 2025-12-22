"""RAG Agent using LangChain for retrieving world lore and game information."""

import logging
from typing import TYPE_CHECKING, Any, Callable, Optional

import chromadb
from chromadb.config import Settings
from langchain.agents import create_agent
from langchain.chat_models import BaseChatModel
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_core.tools import StructuredTool
from langchain_ollama import ChatOllama
from pydantic import BaseModel, Field

from autodnd.config import (
    DEFAULT_CHROMADB_ALLOW_RESET,
    DEFAULT_CHROMADB_ANONYMIZED_TELEMETRY,
    DEFAULT_EMBEDDING_FALLBACK_MODEL,
    DEFAULT_EMBEDDING_MODEL,
    DEFAULT_OLLAMA_BASE_URL,
    DEFAULT_LLM_MODEL,
    DEFAULT_LLM_NUM_CTX,
    DEFAULT_RAG_AGENT_TEMPERATURE,
    DEFAULT_RAG_ALLOWED_CATEGORIES,
    DEFAULT_RAG_COLLECTION_NAME,
    DEFAULT_RAG_SIMILARITY_SEARCH_K,
)

# Try to import OllamaEmbeddings from langchain_ollama first, fallback to langchain_community
try:
    from langchain_ollama import OllamaEmbeddings  # type: ignore
except ImportError:
    from langchain_community.embeddings import OllamaEmbeddings  # type: ignore

if TYPE_CHECKING:
    from autodnd.engine.game_engine import GameEngine

logger = logging.getLogger(__name__)


class QueryLoreInput(BaseModel):
    """Input for query lore tool."""

    query: str = Field(description="Query string to search for in world lore")
    category: Optional[str] = Field(
        default=None, description="Category filter (e.g., 'enemy', 'location', 'item', 'lore')"
    )


class RAGAgent:
    """RAG Agent that retrieves world lore, enemy info, and setting details from vector store."""

    # Allowed categories for metadata filtering (security)
    ALLOWED_CATEGORIES = DEFAULT_RAG_ALLOWED_CATEGORIES

    def __init__(
        self,
        llm: Optional[BaseChatModel] = None,
        engine_getter: Optional["Callable[[], GameEngine]"] = None,
        vector_store: Optional[Chroma] = None,
        embedding_model: Optional[Any] = None,
        persist_directory: Optional[str] = None,
    ) -> None:
        """
        Initialize RAG Agent.

        Args:
            llm: LangChain LLM instance (if None, will be created with defaults)
            engine_getter: Function to get current game engine instance
            vector_store: Chroma vector store for RAG retrieval (optional, will be created if None)
            embedding_model: Embedding model instance (optional, will use Ollama if None)
            persist_directory: Directory to persist ChromaDB data (optional, in-memory if None)
        """
        self._engine_getter = engine_getter
        self._llm = llm or self._create_default_llm()

        # Initialize embedding model
        self._embedding_model = embedding_model or self._create_default_embedding()

        # Initialize vector store
        if vector_store is None:
            self._vector_store = self._create_vector_store(persist_directory)
        else:
            self._vector_store = vector_store

        self._tools = []
        self._agent = None
        self._build_agent()

    def _create_default_llm(self) -> ChatOllama:
        """Create default LLM instance."""
        return ChatOllama(
            model=DEFAULT_LLM_MODEL,
            temperature=DEFAULT_RAG_AGENT_TEMPERATURE,
            base_url=DEFAULT_OLLAMA_BASE_URL.rstrip("/"),  # Remove trailing slash for consistency
            num_ctx=DEFAULT_LLM_NUM_CTX,
        )

    def _create_default_embedding(self) -> OllamaEmbeddings:
        """Create default embedding model (Ollama embeddings)."""
        try:
            return OllamaEmbeddings(
                model=DEFAULT_EMBEDDING_MODEL,
                base_url=DEFAULT_OLLAMA_BASE_URL.rstrip("/"),  # Remove trailing slash for consistency
            )
        except Exception as e:
            logger.warning(f"Failed to create Ollama embeddings: {e}. Using fallback.")
            # Fallback: try with a different model or use a simple embedding
            try:
                return OllamaEmbeddings(
                    model=DEFAULT_EMBEDDING_FALLBACK_MODEL,
                    base_url=DEFAULT_OLLAMA_BASE_URL.rstrip("/"),  # Remove trailing slash for consistency
                )
            except Exception:
                logger.error("Failed to create embedding model. RAG functionality may be limited.")
                raise

    def _create_vector_store(self, persist_directory: Optional[str] = None) -> Chroma:
        """Create ChromaDB vector store with embeddings."""
        collection_name = DEFAULT_RAG_COLLECTION_NAME

        if persist_directory:
            # Persistent storage
            chroma_client = chromadb.PersistentClient(
                path=persist_directory,
                settings=Settings(
                    anonymized_telemetry=DEFAULT_CHROMADB_ANONYMIZED_TELEMETRY,
                    allow_reset=DEFAULT_CHROMADB_ALLOW_RESET,
                ),
            )
        else:
            # In-memory storage
            chroma_client = chromadb.Client(
                settings=Settings(
                    anonymized_telemetry=DEFAULT_CHROMADB_ANONYMIZED_TELEMETRY,
                    allow_reset=DEFAULT_CHROMADB_ALLOW_RESET,
                )
            )

        # Create Chroma vector store using LangChain integration
        vector_store = Chroma(
            client=chroma_client,
            collection_name=collection_name,
            embedding_function=self._embedding_model,
        )

        return vector_store

    def _build_agent(self) -> None:
        """Build the agent using create_agent."""

        # Create query lore tool with full vector store integration
        def query_lore_tool(query: str, category: Optional[str] = None) -> dict:
            """Query world lore and game information from vector store."""
            try:
                # Security: Validate category if provided
                if category and category.lower() not in self.ALLOWED_CATEGORIES:
                    logger.warning(f"Invalid category '{category}' filtered out for security.")
                    category = None

                # Prepare metadata filter for security
                where_filter: Optional[dict[str, Any]] = None
                if category:
                    where_filter = {"category": category.lower()}

                # Perform similarity search with metadata filtering
                if self._vector_store:
                    # Retrieve top N most relevant documents
                    docs = self._vector_store.similarity_search_with_score(
                        query, k=DEFAULT_RAG_SIMILARITY_SEARCH_K, filter=where_filter
                    )

                    if docs:
                        results = []
                        for doc, score in docs:
                            doc_metadata = doc.metadata or {}
                            results.append(
                                {
                                    "content": doc.page_content,
                                    "category": doc_metadata.get("category", "unknown"),
                                    "source": doc_metadata.get("source", "unknown"),
                                    "score": float(score),
                                }
                            )

                        return {
                            "results": results,
                            "query": query,
                            "category": category,
                            "count": len(results),
                        }
                    else:
                        return {
                            "results": [],
                            "query": query,
                            "category": category,
                            "message": "No relevant documents found.",
                        }
                else:
                    return {
                        "results": [],
                        "query": query,
                        "category": category,
                        "message": "Vector store not initialized.",
                    }
            except Exception as e:
                logger.error(f"Error querying vector store: {e}", exc_info=True)
                return {
                    "results": [],
                    "query": query,
                    "category": category,
                    "error": str(e),
                }

        query_tool = StructuredTool.from_function(
            func=query_lore_tool,
            name="query_lore",
            description="Query world lore, enemy information, setting details, or game content. "
            "Optionally filter by category (enemy, location, item, lore, npc, creature, spell). "
            "Returns relevant documents with content and metadata.",
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
- Always cite sources when providing information

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

    def set_embedding_model(self, embedding_model: Any) -> None:
        """Set the embedding model (requires vector store rebuild)."""
        self._embedding_model = embedding_model
        # Recreate vector store with new embedding model
        self._vector_store = self._create_vector_store()

    def add_document(
        self,
        content: str,
        category: str,
        source: Optional[str] = None,
        metadata: Optional[dict[str, Any]] = None,
    ) -> None:
        """
        Add a document to the vector store.

        Args:
            content: Document content/text
            category: Category of the document (must be in ALLOWED_CATEGORIES for security)
            source: Source identifier (e.g., "core_rulebook", "campaign_notes")
            metadata: Additional metadata dictionary
        """
        # Security: Validate category
        if category.lower() not in self.ALLOWED_CATEGORIES:
            logger.warning(f"Invalid category '{category}' rejected for security.")
            return

        try:
            # Prepare document metadata with security filtering
            doc_metadata: dict[str, Any] = {
                "category": category.lower(),
            }
            if source:
                # Sanitize source to prevent injection
                doc_metadata["source"] = str(source).strip()[:100]  # Limit length

            if metadata:
                # Only include safe metadata fields
                safe_keys = {"name", "type", "level", "description"}
                for key in safe_keys:
                    if key in metadata:
                        value = metadata[key]
                        # Convert to string and limit length for security
                        doc_metadata[key] = str(value).strip()[:500]

            # Create document
            document = Document(page_content=content, metadata=doc_metadata)

            # Add to vector store
            self._vector_store.add_documents([document])
            logger.info(f"Added document to vector store: category={category}, source={source}")

        except Exception as e:
            logger.error(f"Error adding document to vector store: {e}", exc_info=True)

    def add_documents(self, documents: list[Document]) -> None:
        """
        Add multiple documents to the vector store.

        Args:
            documents: List of Document objects to add
        """
        try:
            # Filter documents to ensure security
            safe_documents = []
            for doc in documents:
                metadata = doc.metadata or {}
                category = metadata.get("category", "").lower()

                # Validate category
                if category and category in self.ALLOWED_CATEGORIES:
                    # Ensure category is set correctly
                    doc.metadata["category"] = category
                    safe_documents.append(doc)
                else:
                    logger.warning(f"Document with invalid category '{category}' skipped.")

            if safe_documents:
                self._vector_store.add_documents(safe_documents)
                logger.info(f"Added {len(safe_documents)} documents to vector store")
        except Exception as e:
            logger.error(f"Error adding documents to vector store: {e}", exc_info=True)

    def populate_initial_content(self) -> None:
        """Populate vector store with initial game content examples."""
        initial_content = [
            {
                "content": "Goblin: Small, green-skinned humanoids. Typically found in groups. "
                "Weak in combat but dangerous in numbers. Often use crude weapons and tactics. "
                "Common in caves and ruins.",
                "category": "enemy",
                "source": "core_monster_manual",
                "metadata": {"name": "Goblin", "type": "humanoid", "level": "1"},
            },
            {
                "content": "Dragon: Massive, intelligent, ancient reptiles with breath weapons. "
                "Come in many varieties (red, blue, green, etc.) each with different abilities. "
                "Extremely powerful and hoard treasure.",
                "category": "creature",
                "source": "core_monster_manual",
                "metadata": {"name": "Dragon", "type": "dragon", "level": "10+"},
            },
            {
                "content": "Fireball Spell: A classic evocation spell that creates a massive "
                "explosion of fire. Does significant area damage. Requires careful positioning "
                "to avoid harming allies.",
                "category": "spell",
                "source": "core_spellbook",
                "metadata": {"name": "Fireball", "type": "evocation", "level": "3"},
            },
            {
                "content": "The Forgotten Temple: An ancient temple hidden in the mountains. "
                "Rumored to contain powerful artifacts. Guarded by ancient traps and undead. "
                "Last visited decades ago.",
                "category": "location",
                "source": "campaign_lore",
                "metadata": {"name": "Forgotten Temple", "type": "dungeon"},
            },
            {
                "content": "Health Potion: A common magical item that restores health when consumed. "
                "Red liquid in a glass vial. Instant effect. Essential for adventurers.",
                "category": "item",
                "source": "core_item_list",
                "metadata": {"name": "Health Potion", "type": "potion"},
            },
            {
                "content": "The Age of Heroes: A legendary era when powerful heroes roamed the land. "
                "Many artifacts and locations date back to this time. Stories passed down through "
                "generations inspire modern adventurers.",
                "category": "lore",
                "source": "world_history",
                "metadata": {"name": "Age of Heroes", "type": "historical_period"},
            },
        ]

        for item in initial_content:
            self.add_document(
                content=item["content"],
                category=item["category"],
                source=item.get("source"),
                metadata=item.get("metadata"),
            )

        logger.info("Populated vector store with initial game content")

    def query(self, query: str, category: Optional[str] = None) -> dict:
        """
        Query world lore and game information.

        Args:
            query: Query string to search for
            category: Optional category filter

        Returns:
            Dictionary with query results
        """
        messages = [
            {"role": "user", "content": f"Query: {query}" + (f" (Category: {category})" if category else "")}
        ]

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

    def get_vector_store(self) -> Optional[Chroma]:
        """Get the current vector store instance."""
        return self._vector_store
