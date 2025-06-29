# Autonomous Multi-Agent Dialogue Simulator

A sophisticated multi-agent system where AI agents with distinct personalities, roles, and memory engage in dynamic conversations with emotional intelligence and long-term reflection capabilities.

## 🌟 Features

- **2-5 AI Agents** with unique personalities and roles
- **Role-Specific Memory** using ChromaDB vector storage
- **Emotional States** and intent tracking
- **Perception Modules** for context interpretation
- **Long-term Reflection** every few conversation turns
- **Automatic Model Download** (Mistral-7B-Instruct)
- **Relationship Modeling** using NetworkX
- **Real-time Conversation Simulation**

## 🧩 Agent Roles

- **Doctor**: Medical expert with empathetic personality
- **Engineer**: Technical problem-solver with analytical mindset
- **Spy**: Secretive information gatherer with cunning traits
- **Rebel**: Revolutionary with passionate ideals
- **Diplomat**: Peaceful negotiator with balanced approach

## 🛠 Technology Stack

- `transformers` - AI model integration (Mistral-7B-Instruct)
- `langchain` - Memory and tool management
- `chromadb` - Vector database for agent memories
- `pydantic` - Data validation and modeling
- `networkx` - Relationship and social network modeling
- `torch` - PyTorch for model inference
- `sentence-transformers` - Text embeddings

## 🚀 Quick Start

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Run a basic simulation:
```python
python main.py
```

3. Run with custom scenario:
```python
python main.py --scenario "political_negotiation" --agents 4
```

### 🌐 Web Interface (Streamlit)

Launch the interactive web interface:

```bash
# Option 1: Use the launcher script
python launch_ui.py

# Option 2: Direct streamlit command
streamlit run streamlit_app.py
```

The web interface provides:
- 🎛️ **Interactive Configuration**: Choose scenarios, number of agents, conversation length
- 📱 **Real-time Visualization**: Watch conversations unfold in real-time
- 👥 **Agent Profiles**: View agent personalities and roles
- 📊 **Analytics**: Conversation statistics and metrics
- 💾 **Export**: Download conversations as JSON files
- 🎨 **Modern UI**: Clean, responsive design with live updates

## 📁 Project Structure

```
├── src/
│   ├── agents/
│   │   ├── base_agent.py          # Base agent class
│   │   ├── personality.py         # Personality traits system
│   │   └── roles/                 # Role-specific implementations
│   ├── memory/
│   │   ├── vector_memory.py       # ChromaDB integration
│   │   └── reflection.py          # Long-term reflection system
│   ├── perception/
│   │   └── context_interpreter.py # Context perception module
│   ├── simulation/
│   │   ├── dialogue_manager.py    # Conversation orchestration
│   │   └── scenarios.py           # Pre-defined scenarios
│   └── utils/
│       ├── model_manager.py       # Model loading and management
│       └── relationship_graph.py  # NetworkX relationship modeling
├── config/
│   └── settings.py               # Configuration settings
├── examples/                     # Example conversations and scenarios
├── main.py                      # Main entry point
└── requirements.txt             # Dependencies
```

## 🎮 Usage Examples

### Basic Conversation
```python
from src.simulation.dialogue_manager import DialogueManager

# Create simulation with 3 agents
manager = DialogueManager(num_agents=3, scenario="team_building")
conversation = manager.simulate_conversation(max_turns=20)
```

### Custom Agent Configuration
```python
from src.agents.base_agent import Agent
from src.agents.personality import Personality

# Create custom agent
personality = Personality(
    openness=0.8,
    conscientiousness=0.7,
    extraversion=0.6,
    agreeableness=0.4,
    neuroticism=0.3
)

agent = Agent(
    name="Custom Agent",
    role="consultant",
    personality=personality
)
```

## 🧠 Memory & Reflection

Agents maintain:
- **Short-term memory**: Recent conversation context
- **Long-term memory**: Important experiences stored in ChromaDB
- **Reflective memory**: Insights generated every 5-10 turns
- **Relationship memory**: Dynamic relationship states with other agents

## 🎯 Scenarios

Built-in scenarios include:
- Political negotiation
- Team building exercise
- Crisis management
- Scientific collaboration
- Social gathering

## 📊 Monitoring

The system provides real-time monitoring of:
- Agent emotional states
- Memory retrieval patterns
- Relationship dynamics
- Conversation flow and engagement

## 🔧 Configuration

Customize the simulation through `config/settings.py`:
- Model parameters
- Memory retention settings
- Reflection frequency
- Emotional response sensitivity

