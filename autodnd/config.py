"""Central configuration defaults and constants for AutoDnD."""

# LLM Provider Defaults
DEFAULT_LLM_PROVIDER = "ollama"
DEFAULT_LLM_BASE_URL = "http://localhost:11434/"
DEFAULT_LLM_MODEL = "gpt-oss:20b"
DEFAULT_LLM_TEMPERATURE = 0.7
DEFAULT_LLM_TIMEOUT = 60
DEFAULT_LLM_NUM_CTX = 2**14  # 16384 tokens context window

# Agent-Specific LLM Defaults
# Different agents may need different temperatures for their roles
DEFAULT_GAME_MASTER_TEMPERATURE = 0.4  # Balanced creativity and consistency
DEFAULT_NPC_AGENT_TEMPERATURE = 0.7  # Higher for varied NPC personalities
DEFAULT_RAG_AGENT_TEMPERATURE = 0.3  # Lower for factual retrieval
DEFAULT_ACTION_VALIDATOR_TEMPERATURE = 0.1  # Very low for consistent validation

# Security Configuration Defaults
DEFAULT_SECURITY_ENABLED = True
DEFAULT_SECURITY_USE_LLM_VALIDATION = True
DEFAULT_SECURITY_MAX_INPUT_LENGTH = 1000
DEFAULT_SECURITY_LLM_MODEL = "qwen3:8b"  # Cheaper model for security validation
DEFAULT_SECURITY_LLM_TEMPERATURE = 0.1  # Low temperature for consistent validation
DEFAULT_SECURITY_LLM_TIMEOUT = 30  # Shorter timeout for security checks

# RAG Configuration Defaults
DEFAULT_EMBEDDING_MODEL = "nomic-embed-text"
DEFAULT_EMBEDDING_FALLBACK_MODEL = "llama3"  # Fallback if primary embedding model fails
DEFAULT_RAG_COLLECTION_NAME = "autodnd_lore"
DEFAULT_RAG_SIMILARITY_SEARCH_K = 3  # Number of documents to retrieve
DEFAULT_RAG_ALLOWED_CATEGORIES = {
    "enemy",
    "location",
    "item",
    "lore",
    "npc",
    "creature",
    "spell",
}

# Message History Context Defaults
DEFAULT_GAME_MASTER_MESSAGE_HISTORY_LIMIT = 10  # Last N messages for context
DEFAULT_NPC_AGENT_MESSAGE_HISTORY_LIMIT = 5  # Last N messages for context

# ChromaDB Configuration
DEFAULT_CHROMADB_ANONYMIZED_TELEMETRY = False
DEFAULT_CHROMADB_ALLOW_RESET = True

# Ollama API Key (Ollama doesn't require real API key, but some libraries expect it)
DEFAULT_OLLAMA_API_KEY = "ollama"

