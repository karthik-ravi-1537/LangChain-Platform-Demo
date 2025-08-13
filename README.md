# LangChain Platform Demo

This project demonstrates the integration and usage of **LangChain**, **LangGraph**, and **LangSmith** - showcasing how
these tools work together to build powerful AI applications with monitoring and observability.

## 🚀 Features

- **LangChain Integration**: Chains, agents, tools, and memory management
- **LangGraph Workflows**: Complex workflow orchestration (coming soon)
- **LangSmith Monitoring**: Real-time tracing, debugging, and analytics
- **Modular Architecture**: Clean, extensible project structure
- **Production Ready**: Environment configuration and error handling

## 📁 Project Structure

```
LangChain-Platform-Demo/
├── README.md                    # This file
├── environment.yml              # Conda environment configuration
├── requirements.txt             # Alternative pip dependencies
├── .env.example                 # Environment variables template
├── config/
│   └── settings.py             # Application configuration
├── src/
│   ├── chains/
│   │   └── basic_chain.py      # LangChain chain examples
│   ├── tools/
│   │   └── calculator.py       # Custom tools for agents
│   ├── langsmith/
│   │   └── setup.py            # LangSmith configuration
│   └── utils/
├── examples/
│   └── basic_usage.py          # Complete usage demonstration
└── tests/                      # Unit tests (coming soon)
```

## 🛠️ Setup Instructions

### Prerequisites
- Python 3.11+ installed
- **For conda users**: [Anaconda](https://www.anaconda.com/products/distribution) or [Miniconda](https://docs.conda.io/en/latest/miniconda.html)
- **For venv users**: Python's built-in venv module (included with Python 3.3+)

### 1. Create and Activate Conda Environment

**Option A: Using environment.yml (Recommended)**
```bash
# Create environment from file
conda env create -f environment.yml

# Activate the environment
conda activate langchain-platform-demo
```

**Option B: Manual setup with pip**
```bash
# Create a new conda environment
conda create -n langchain-platform-demo python=3.11 -y

# Activate the environment
conda activate langchain-platform-demo

# Install dependencies
pip install -r requirements.txt
```

**Option C: Using Python venv (Alternative)**
```bash
# Create a virtual environment
python -m venv venv-langchain-platform-demo

# Activate the environment
# On macOS/Linux:
source venv-langchain-platform-demo/bin/activate
# On Windows:
# venv-langchain-platform-demo\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Configure Environment Variables

1. Copy the environment template:
   ```bash
   cp .env.example .env
   ```

2. Edit `.env` and add your API keys:
   ```env
   # Required
   OPENAI_API_KEY=your_openai_api_key_here
   
   # Optional (for LangSmith tracing)
   LANGSMITH_TRACING=true
   LANGSMITH_API_KEY=your_langsmith_api_key_here
   LANGSMITH_PROJECT=langchain-demo
   ```

### 3. Get API Keys

- **OpenAI API Key**: Get from [OpenAI Platform](https://platform.openai.com/api-keys)
- **LangSmith API Key**: Get from [LangSmith](https://smith.langchain.com/) (optional but recommended)

## 🚦 Quick Start

### Run the Basic Demo

```bash
# Make sure your environment is activated
# For conda:
conda activate langchain-platform-demo
# For venv:
# source venv-langchain-platform-demo/bin/activate  # macOS/Linux
# venv-langchain-platform-demo\Scripts\activate     # Windows

# Run the demo
python examples/basic_usage.py
```

This will demonstrate:

- ✅ Basic LangChain chains
- ✅ Custom tools (calculator)
- ✅ Conversational agents with memory
- ✅ LangSmith tracing (if configured)

### Test Individual Components

```bash
# Test basic chain
python src/chains/basic_chain.py

# Test calculator tools
python src/tools/calculator.py
```

## 📊 LangSmith Integration

LangSmith provides powerful monitoring and debugging capabilities:

### Features Demonstrated:

- **Automatic Tracing**: All LangChain operations are automatically traced
- **Performance Monitoring**: Track latency, token usage, and costs
- **Debugging**: Step-by-step execution visualization
- **Analytics**: Usage patterns and performance metrics

### Viewing Traces:

1. Ensure `LANGSMITH_TRACING=true` in your `.env`
2. Add your LangSmith API key
3. Run any example
4. View traces at [smith.langchain.com](https://smith.langchain.com)

## 🔧 Components Overview

### LangChain Components

#### Basic Chain (`src/chains/basic_chain.py`)

- Simple fact generation
- List generation with custom output parsers
- Prompt template management
- Error handling

#### Calculator Tools (`src/tools/calculator.py`)

- Basic mathematical operations
- Advanced functions (sqrt, sin, cos, etc.)
- Input validation and error handling
- Async support

### LangSmith Components

#### Setup (`src/langsmith/setup.py`)

- Automatic tracing configuration
- Dataset creation and management
- Feedback logging
- Client initialization

### Configuration (`config/settings.py`)

- Environment variable management
- API key validation
- Centralized settings

## 🎯 Usage Examples

### Basic Chain Usage

```python
from src.chains.basic_chain import BasicChain

chain = BasicChain()
fact = chain.get_simple_fact("artificial intelligence")
print(fact)
```

### Agent with Tools

```python
from langchain.agents import initialize_agent
from src.tools.calculator import calculator_tool

agent = initialize_agent([calculator_tool], llm, verbose=True)
result = agent.invoke("What is 25 * 4?")
```

### LangSmith Tracing

All operations are automatically traced when LangSmith is configured. No additional code needed!

## 📈 What's Next

This project will be extended with:

- **LangGraph Workflows**: Complex multi-step workflows
- **RAG Implementation**: Document-based question answering
- **Web Search Tools**: Integration with search APIs
- **Advanced Agents**: Multi-agent systems
- **Custom Evaluators**: LangSmith evaluation metrics
- **Unit Tests**: Comprehensive test coverage

## 🐛 Troubleshooting

### Common Issues

1. **Missing API Keys**
   ```
   Error: Missing API keys: OPENAI_API_KEY
   ```
   Solution: Add your OpenAI API key to the `.env` file

2. **LangSmith Not Working**
   ```
   LangSmith tracing disabled
   ```
   Solution: Add `LANGSMITH_API_KEY` and set `LANGSMITH_TRACING=true`

3. **Import Errors**
   ```
   ModuleNotFoundError: No module named 'langchain'
   ```
   Solution: Ensure your environment is activated and dependencies are installed:
   ```bash
   # For conda:
   conda activate langchain-platform-demo
   conda env update -f environment.yml  # or pip install -r requirements.txt
   
   # For venv:
   source venv-langchain-platform-demo/bin/activate  # macOS/Linux
   # venv-langchain-platform-demo\Scripts\activate   # Windows
   pip install -r requirements.txt
   ```

4. **Environment Not Found**
   ```
   CondaEnvironmentError: cannot locate environment
   ```
   Solution: Create the environment first:
   ```bash
   conda env create -f environment.yml
   ```

## 📚 Resources

- [LangChain Documentation](https://python.langchain.com/)
- [LangGraph Documentation](https://langchain-ai.github.io/langgraph/)
- [LangSmith Documentation](https://docs.smith.langchain.com/)
- [OpenAI API Documentation](https://platform.openai.com/docs)

## 🤝 Contributing

Feel free to extend this project with additional examples, tools, or workflows. The modular structure makes it easy to
add new components.

---

**Happy Building! 🎉**
