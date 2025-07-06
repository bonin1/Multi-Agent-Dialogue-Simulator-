# 🤖 Autonomous Multi-Agent Dialogue Simulator

A sophisticated multi-agent conversation system where AI agents with distinct personalities, memories, and emotions engage in realistic, human-like conversations. Create custom agents, watch them interact naturally, and explore complex social dynamics through AI simulation.

## ✨ Key Features

### 🗣️ **Human-Like Conversations**
- **Natural speech patterns** with contractions, interruptions, and emotional reactions
- **Brief, realistic responses** - agents don't give speeches
- **Spontaneous reactions** to what was just said
- **Emotional authenticity** with genuine surprise, confusion, excitement, and disagreement

### 🛠️ **Custom Agent Creation**
- **Visual Agent Creator** with step-by-step wizard
- **Pre-built templates**: Tech Entrepreneur, Environmental Activist, Philosophy Professor, Social Media Influencer
- **Advanced personality control** using Big Five personality model
- **Custom instructions** and behavioral patterns
- **Import/Export** agent packs for sharing

### 🧠 **Advanced AI Behavior**
- **Modular prompt system** with role-specific personas
- **Emotional state tracking** and realistic emotional responses
- **Memory systems** with ChromaDB for persistent agent memory
- **Reflection capabilities** for agent self-awareness and growth

### 🎭 **Rich Simulation Modes**
- **Continuous auto-play** mode for autonomous conversations
- **Manual turn-by-turn** control for detailed observation
- **Multiple scenarios**: Political negotiations, crisis management, team building, corporate espionage
- **Real-time analytics** showing conversation patterns and agent states

### 📊 **Analytics & Insights**
- **Conversation flow analysis** with message distribution
- **Emotional state tracking** for all agents
- **Agent relationship dynamics** with trust levels
- **Export capabilities** for conversation logs

## 🚀 Quick Start

### Installation

1. **Clone the repository:**
```bash
git clone https://github.com/bonin1/Multi-Agent-Dialogue-Simulator.git
cd Multi-Agent-Dialogue-Simulator
```

2. **Install dependencies:**
```bash
pip install -r requirements.txt
```

3. **Run the application:**
```bash
streamlit run main.py
```

### First Simulation

1. **Load the AI model** (first time takes a few minutes)
2. **Choose a scenario** from the dropdown (e.g., "Team Building")
3. **Select agents** from predefined characters or create your own
4. **Start simulation** and watch agents interact
5. **Switch between continuous and manual modes** as desired

## 🎨 Creating Custom Agents

### Using Templates (Fastest)
```python
from agents.agent_creator import AgentCreator

creator = AgentCreator()

# Create from template
config = creator.create_agent_from_template(
    template_name="tech_entrepreneur",
    agent_name="Alex Innovation",
    custom_instructions="Always thinks about scalability"
)

creator.save_agent_config(config)
```

### Using the Visual Creator
1. Go to the **"Create Agents"** tab
2. Follow the **6-step wizard**:
   - Basic Information (name, role, description)
   - Personality Traits (Big Five model)
   - Background & Goals
   - Behavior Patterns & Speech
   - Psychology & Emotions
   - Advanced Settings
3. **Preview and create** your agent
4. **Test in conversations** and refine as needed

### Advanced Customization
```python
from agents.agent_creator import CustomAgentConfig

config = CustomAgentConfig(
    name="Dr. Luna Research",
    role="scientist",
    description="A marine biologist passionate about ocean conservation",
    
    personality_traits={
        "openness": 0.9,        # Very open to new ideas
        "conscientiousness": 0.8, # Highly organized
        "extraversion": 0.4,     # Somewhat introverted
        "agreeableness": 0.7,    # Cooperative
        "neuroticism": 0.3       # Emotionally stable
    },
    
    speech_patterns=[
        "The data shows...",
        "In my research, I've found...",
        "That's fascinating because..."
    ],
    
    emotional_triggers=[
        "Ocean pollution",
        "Climate change denial",
        "Overfishing"
    ],
    
    response_style="detailed",
    conflict_style="diplomatic"
)
```

## 🎪 Example Scenarios

### **Political Negotiation**
Watch agents with different political views negotiate environmental policy. Agents bring their backgrounds, biases, and goals to create realistic political dynamics.

### **Crisis Management** 
A natural disaster strikes. Agents must coordinate rescue efforts while managing different priorities, resources, and personalities under pressure.

### **Corporate Espionage**
Agents from different organizations work together while protecting their own interests. Trust is scarce and everyone has hidden agendas.

### **Innovation Workshop**
Diverse experts brainstorm solutions to global problems, balancing creativity with practical constraints and different perspectives.

## 🏗️ Architecture

```
├── agents/
│   ├── agent.py              # Core agent logic
│   ├── agent_creator.py      # Custom agent creation system
│   └── ...
├── models/
│   ├── model_manager.py      # AI model handling
│   ├── agent_models.py       # Data models and enums
│   └── ...
├── prompts/
│   └── agent_prompts.py      # Advanced prompt engineering
├── memory/
│   └── memory_system.py      # ChromaDB memory management
├── scenarios/
│   └── scenario_manager.py   # Simulation scenarios
├── ui/
│   ├── agent_creator_ui.py   # Agent creation interface
│   └── ...
├── main.py                   # Main Streamlit application
└── requirements.txt
```

### Example Agent Templates
- **Tech Entrepreneur**: Innovation-focused, market-oriented thinking
- **Environmental Activist**: Passionate about climate action and justice
- **Philosophy Professor**: Questions assumptions, thinks deeply
- **Social Media Influencer**: Trend-focused, engagement-oriented

## 🔧 Supported Models

- **teknium/OpenHermes-2.5-Mistral-7B** (default, recommended)
- **Any compatible Hugging Face transformer model**
- **Optimized for conversational AI** with enhanced creativity settings

## 🔬 Advanced Features

### **Emotional Intelligence**
- Agents track and express complex emotional states
- Emotional reactions influence future responses
- Relationship dynamics evolve based on interactions

### **Memory & Learning**
- Persistent memory using ChromaDB vector database
- Agents remember past conversations and experiences
- Contextual retrieval for relevant memories

### **Natural Language Processing**
- Advanced prompt engineering for human-like responses
- Role-specific language patterns and vocabularies
- Dynamic response length based on emotional state

### **Simulation Control**
- Pause/resume conversations at any time
- Adjust auto-play speed (1-10 seconds between turns)
- Manual intervention and context injection

## 🚦 Requirements

- **Python 3.8+**
- **CUDA-compatible GPU** (recommended for best performance)
- **8GB+ RAM** (16GB+ recommended)
- **5GB+ storage** for models and data

## 🤝 Contributing

We welcome contributions! Areas where help is needed:

1. **New agent templates** for different professions and personalities
2. **Additional scenarios** for various simulation contexts
3. **UI improvements** and user experience enhancements
4. **Performance optimizations** for faster model inference
5. **Documentation** and tutorial improvements

## 🙏 Acknowledgments

- **Hugging Face** for transformer models and infrastructure
- **ChromaDB** for vector database capabilities
- **Streamlit** for the interactive web interface
- **OpenAI** and **Anthropic** for inspiration in AI conversation design

---

**Ready to create your own AI agents and watch them come to life? Get started now!** 🚀
