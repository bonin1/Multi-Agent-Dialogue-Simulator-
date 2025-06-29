# Updated configuration with improved model settings

import os
from pathlib import Path

# Base paths
BASE_DIR = Path(__file__).parent.parent.parent
DATA_DIR = BASE_DIR / "data"
LOGS_DIR = BASE_DIR / "logs"
CACHE_DIR = BASE_DIR / "cache"

# Create directories if they don't exist
DATA_DIR.mkdir(exist_ok=True)
LOGS_DIR.mkdir(exist_ok=True)
CACHE_DIR.mkdir(exist_ok=True)

# Model configuration
MODEL_CONFIG = {
    "name": "teknium/OpenHermes-2.5-Mistral-7B",
    "cache_dir": str(CACHE_DIR),
    "device": "auto",
    "quantization_bits": 4,
    "max_memory": {0: "7GB"},
    "use_auth_token": None,  # Set to None instead of False
    "trust_remote_code": True,
    "generation_config": {
        "max_new_tokens": 512,          # Increased for better responses
        "min_new_tokens": 10,           # Minimum tokens to generate
        "max_length": None,             # Use max_new_tokens instead
        "temperature": 0.7,
        "top_p": 0.9,
        "top_k": 50,
        "do_sample": True,
        "pad_token_id": None,           # Will be set automatically
        "eos_token_id": None,           # Will be set automatically
        "repetition_penalty": 1.1,
        "length_penalty": 1.0,
        "truncation": True,
        "padding": True
    }
}

# Memory configuration
MEMORY_CONFIG = {
    "provider": "chroma",
    "chroma_persist_directory": str(DATA_DIR / "chroma_db"),
    "collection_name": "dialogue_memory",
    "embedding_model": "sentence-transformers/all-MiniLM-L6-v2"
}

# Agent configuration
AGENT_CONFIG = {
    "max_agents": 10,
    "default_personality_strength": 0.7,
    "response_timeout": 30,
    "max_conversation_turns": 20
}

# Streamlit configuration
STREAMLIT_CONFIG = {
    "page_title": "Multi-Agent Dialogue Simulator",
    "page_icon": "🤖",
    "layout": "wide",
    "initial_sidebar_state": "expanded"
}

# Logging configuration
LOGGING_CONFIG = {
    "level": "INFO",
    "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    "handlers": ["console", "file"],
    "file_path": str(LOGS_DIR / "dialogue_simulator.log")
}

# Environment variables
DEBUG = os.getenv("DEBUG", "false").lower() == "true"
API_KEY = os.getenv("API_KEY", None)
