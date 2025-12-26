"""Central configuration defaults and constants for AutoDnD."""

import os

# LLM Provider Defaults
DEFAULT_LLM_PROVIDER = os.getenv("AUTODND_LLM_PROVIDER", "openai")
DEFAULT_OLLAMA_BASE_URL = os.getenv("AUTODND_OLLAMA_BASE_URL", "http://localhost:11434/")
DEFAULT_OPENAI_BASE_URL = os.getenv("AUTODND_OPENAI_BASE_URL", "https://bothub.chat/api/v2/openai/v1")
DEFAULT_OPENAI_API_KEY = os.getenv("AUTODND_OPENAI_API_KEY", "")
DEFAULT_LLM_MODEL = os.getenv("AUTODND_LLM_MODEL", "gpt-oss-120b")
DEFAULT_LLM_TEMPERATURE = float(os.getenv("AUTODND_LLM_TEMPERATURE", "0.7"))
DEFAULT_LLM_TIMEOUT = int(os.getenv("AUTODND_LLM_TIMEOUT", "60"))
DEFAULT_LLM_NUM_CTX = int(os.getenv("AUTODND_LLM_NUM_CTX", str(2**14)))  # 16384 tokens context window

# Agent-Specific LLM Defaults
# Different agents may need different temperatures for their roles
DEFAULT_GAME_MASTER_TEMPERATURE = float(os.getenv("AUTODND_GAME_MASTER_TEMPERATURE", "0.4"))  # Balanced creativity and consistency
DEFAULT_NPC_AGENT_TEMPERATURE = float(os.getenv("AUTODND_NPC_AGENT_TEMPERATURE", "0.7"))  # Higher for varied NPC personalities
DEFAULT_RAG_AGENT_TEMPERATURE = float(os.getenv("AUTODND_RAG_AGENT_TEMPERATURE", "0.3"))  # Lower for factual retrieval
DEFAULT_ACTION_VALIDATOR_TEMPERATURE = float(os.getenv("AUTODND_ACTION_VALIDATOR_TEMPERATURE", "0.1"))  # Very low for consistent validation

# Security Configuration Defaults
DEFAULT_SECURITY_ENABLED = os.getenv("AUTODND_SECURITY_ENABLED", "true").lower() in ("true", "1", "yes", "on")
DEFAULT_SECURITY_USE_LLM_VALIDATION = os.getenv("AUTODND_SECURITY_USE_LLM_VALIDATION", "true").lower() in ("true", "1", "yes", "on")
DEFAULT_SECURITY_MAX_INPUT_LENGTH = int(os.getenv("AUTODND_SECURITY_MAX_INPUT_LENGTH", "1000"))
DEFAULT_SECURITY_LLM_MODEL = os.getenv("AUTODND_SECURITY_LLM_MODEL", "qwen3:8b")  # Cheaper model for security validation
DEFAULT_SECURITY_LLM_TEMPERATURE = float(os.getenv("AUTODND_SECURITY_LLM_TEMPERATURE", "0.1"))  # Low temperature for consistent validation
DEFAULT_SECURITY_LLM_TIMEOUT = int(os.getenv("AUTODND_SECURITY_LLM_TIMEOUT", "30"))  # Shorter timeout for security checks

# RAG Configuration Defaults
DEFAULT_EMBEDDING_MODEL = os.getenv("AUTODND_EMBEDDING_MODEL", "nomic-embed-text")
DEFAULT_EMBEDDING_FALLBACK_MODEL = os.getenv("AUTODND_EMBEDDING_FALLBACK_MODEL", "llama3")  # Fallback if primary embedding model fails
DEFAULT_RAG_COLLECTION_NAME = os.getenv("AUTODND_RAG_COLLECTION_NAME", "autodnd_lore")
DEFAULT_RAG_SIMILARITY_SEARCH_K = int(os.getenv("AUTODND_RAG_SIMILARITY_SEARCH_K", "3"))  # Number of documents to retrieve
# RAG allowed categories - parse from comma-separated env var or use default set
_rag_categories_env = os.getenv("AUTODND_RAG_ALLOWED_CATEGORIES")
DEFAULT_RAG_ALLOWED_CATEGORIES = (
    set(_rag_categories_env.split(",")) if _rag_categories_env
    else {
        "enemy",
        "location",
        "item",
        "lore",
        "npc",
        "creature",
        "spell",
    }
)

# Message History Context Defaults
DEFAULT_GAME_MASTER_MESSAGE_HISTORY_LIMIT = int(os.getenv("AUTODND_GAME_MASTER_MESSAGE_HISTORY_LIMIT", "10"))  # Last N messages for context
DEFAULT_NPC_AGENT_MESSAGE_HISTORY_LIMIT = int(os.getenv("AUTODND_NPC_AGENT_MESSAGE_HISTORY_LIMIT", "5"))  # Last N messages for context

# ChromaDB Configuration
DEFAULT_CHROMADB_ANONYMIZED_TELEMETRY = os.getenv("AUTODND_CHROMADB_ANONYMIZED_TELEMETRY", "false").lower() in ("true", "1", "yes", "on")
DEFAULT_CHROMADB_ALLOW_RESET = os.getenv("AUTODND_CHROMADB_ALLOW_RESET", "true").lower() in ("true", "1", "yes", "on")

# Ollama API Key (Ollama doesn't require real API key, but some libraries expect it)
DEFAULT_OLLAMA_API_KEY = os.getenv("AUTODND_OLLAMA_API_KEY", "ollama")

