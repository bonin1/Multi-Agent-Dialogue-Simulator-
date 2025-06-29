"""
Configuration settings for the Multi-Agent Dialogue Simulator
"""
import os
from pathlib import Path
from typing import Dict, Any, List

# Base directories
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
LOGS_DIR = PROJECT_ROOT / "logs"

# Create directories if they don't exist
for dir_path in [DATA_DIR, MODELS_DIR, LOGS_DIR]:
    dir_path.mkdir(exist_ok=True)

# Model Configuration
MODEL_CONFIG = {
    "name": "teknium/OpenHermes-2.5-Mistral-7B",
    "device_map": "auto",
    "torch_dtype": "float16",
    "load_in_4bit": True,
    "max_length": 2048,
    "temperature": 0.7,
    "top_p": 0.9,
    "do_sample": True,
    "pad_token_id": 2,  # EOS token for Mistral
    "cache_dir": str(MODELS_DIR),
    "use_auth_token": False,
}

# Memory Configuration
MEMORY_CONFIG = {
    "chromadb_path": str(DATA_DIR / "agent_memories"),
    "collection_name": "agent_conversations",
    "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
    "max_memory_items": 1000,
    "similarity_threshold": 0.7,
    "reflection_frequency": 5,  # Reflect every 5 turns
}

# Agent Configuration
AGENT_CONFIG = {
    "max_agents": 5,
    "default_personalities": {
        "doctor": {
            "openness": 0.8,
            "conscientiousness": 0.9,
            "extraversion": 0.6,
            "agreeableness": 0.8,
            "neuroticism": 0.3,
            "traits": ["empathetic", "analytical", "patient", "caring"],
            "background": "Medical professional with 10+ years experience"
        },
        "engineer": {
            "openness": 0.9,
            "conscientiousness": 0.8,
            "extraversion": 0.4,
            "agreeableness": 0.6,
            "neuroticism": 0.2,
            "traits": ["logical", "methodical", "innovative", "precise"],
            "background": "Software engineer specializing in complex systems"
        },
        "spy": {
            "openness": 0.7,
            "conscientiousness": 0.7,
            "extraversion": 0.5,
            "agreeableness": 0.3,
            "neuroticism": 0.4,
            "traits": ["secretive", "observant", "cunning", "adaptive"],
            "background": "Intelligence operative with covert operations experience"
        },
        "rebel": {
            "openness": 0.9,
            "conscientiousness": 0.4,
            "extraversion": 0.8,
            "agreeableness": 0.3,
            "neuroticism": 0.6,
            "traits": ["passionate", "idealistic", "confrontational", "charismatic"],
            "background": "Social activist fighting for systemic change"
        },
        "diplomat": {
            "openness": 0.8,
            "conscientiousness": 0.8,
            "extraversion": 0.7,
            "agreeableness": 0.9,
            "neuroticism": 0.2,
            "traits": ["tactful", "patient", "persuasive", "balanced"],
            "background": "International relations expert with negotiation expertise"
        }
    }
}

# Emotion Configuration
EMOTION_CONFIG = {
    "base_emotions": ["joy", "sadness", "anger", "fear", "surprise", "disgust"],
    "emotion_decay_rate": 0.1,  # How quickly emotions fade
    "emotion_intensity_range": (0.0, 1.0),
    "emotion_influence_threshold": 0.3,  # Minimum emotion level to affect behavior
}

# Simulation Configuration
SIMULATION_CONFIG = {
    "max_conversation_turns": 50,
    "turn_timeout": 30,  # seconds
    "context_window": 10,  # number of previous messages to consider
    "enable_interruptions": True,
    "interruption_probability": 0.1,
    "topics_drift_probability": 0.15,
}

# Scenarios
SCENARIOS = {
    "political_negotiation": {
        "description": "A tense political negotiation between opposing parties",
        "context": "The country faces a critical decision about environmental policy. Stakeholders with different interests must find common ground.",
        "goals": ["Find compromise", "Protect interests", "Maintain relationships"],
        "tension_level": 0.8
    },
    "team_building": {
        "description": "A corporate team building exercise",
        "context": "New team members from different departments need to work together on an important project.",
        "goals": ["Build trust", "Establish roles", "Create synergy"],
        "tension_level": 0.3
    },
    "crisis_management": {
        "description": "Emergency response to a developing crisis",
        "context": "A natural disaster has struck and the response team must coordinate rescue and relief efforts.",
        "goals": ["Save lives", "Coordinate resources", "Manage information"],
        "tension_level": 0.9
    },
    "scientific_collaboration": {
        "description": "Researchers collaborating on a breakthrough discovery",
        "context": "Scientists from different fields must combine their expertise to solve a complex problem.",
        "goals": ["Share knowledge", "Validate theories", "Plan experiments"],
        "tension_level": 0.4
    },
    "social_gathering": {
        "description": "Casual social interaction at a community event",
        "context": "People from different backgrounds meet at a neighborhood gathering.",
        "goals": ["Build connections", "Share stories", "Have fun"],
        "tension_level": 0.2
    }
}

# Logging Configuration
LOGGING_CONFIG = {
    "level": "INFO",
    "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    "file_handler": True,
    "console_handler": True,
    "log_file": str(LOGS_DIR / "simulation.log"),
    "max_file_size": 10 * 1024 * 1024,  # 10MB
    "backup_count": 5,
}

# API Configuration (for future web interface)
API_CONFIG = {
    "host": "localhost",
    "port": 8000,
    "reload": True,
    "workers": 1,
}

def get_config(section: str = None) -> Dict[str, Any]:
    """Get configuration for a specific section or all configurations"""
    all_configs = {
        "model": MODEL_CONFIG,
        "memory": MEMORY_CONFIG,
        "agent": AGENT_CONFIG,
        "emotion": EMOTION_CONFIG,
        "simulation": SIMULATION_CONFIG,
        "scenarios": SCENARIOS,
        "logging": LOGGING_CONFIG,
        "api": API_CONFIG,
    }
    
    if section:
        return all_configs.get(section, {})
    return all_configs

def update_config(section: str, updates: Dict[str, Any]) -> None:
    """Update configuration for a specific section"""
    config_map = {
        "model": MODEL_CONFIG,
        "memory": MEMORY_CONFIG,
        "agent": AGENT_CONFIG,
        "emotion": EMOTION_CONFIG,
        "simulation": SIMULATION_CONFIG,
        "scenarios": SCENARIOS,
        "logging": LOGGING_CONFIG,
        "api": API_CONFIG,
    }
    
    if section in config_map:
        config_map[section].update(updates)
    else:
        raise ValueError(f"Unknown configuration section: {section}")
