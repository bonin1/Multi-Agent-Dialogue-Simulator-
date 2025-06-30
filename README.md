# Autonomous Multi-Agent Dialogue Simulator

A sophisticated multi-agent system where AI agents with distinct personalities, roles, and memory systems engage in dynamic conversations with emotional and contextual awareness.

## Features

- **Multi-Agent System**: 2-5 AI agents with unique personalities and roles
- **Role-Specific Memory**: Each agent maintains private memory using ChromaDB
- **Emotional States**: Agents track and express emotions during conversations
- **Perception Modules**: Context interpretation for images, events, and dialogue
- **Long-Term Reflection**: Agents reflect on conversations and update their beliefs
- **Interactive UI**: Streamlit-based interface for real-time interaction

## Installation

1. Clone the repository:
```bash
git clone https://github.com/your-username/Multi-Agent-Dialogue-Simulator.git
cd Multi-Agent-Dialogue-Simulator
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run the application:
```bash
streamlit run main.py
```

## Usage

1. Select scenario type (political negotiation, team building, etc.)
2. Choose agents and their roles
3. Start the simulation
4. Watch agents interact autonomously
5. Intervene or add context as needed

## Architecture

- `agents/`: Agent definitions and personalities
- `memory/`: Memory management and retrieval systems
- `perception/`: Context interpretation modules
- `ui/`: Streamlit interface components
- `scenarios/`: Pre-defined simulation scenarios
- `models/`: Model management and loading

## Supported Models

- teknium/OpenHermes-2.5-Mistral-7B (default)
- Any compatible Hugging Face transformer model

## License

MIT License
