# LangChain Platform Demo

Integration and usage demonstration of **LangChain**, **LangGraph**, and **LangSmith** for building AI applications with monitoring and observability.

## Features

- **LangChain Integration**: Chains, agents, tools, and memory management
- **LangGraph Workflows**: Complex workflow orchestration
- **LangSmith Monitoring**: Real-time tracing, debugging, and analytics
- **Modular Architecture**: Clean, extensible project structure

## Installation

### Prerequisites
- [Homebrew](https://brew.sh/) for installing uv
- [Anaconda](https://www.anaconda.com/products/distribution) or [Miniconda](https://docs.conda.io/en/latest/miniconda.html)

### Method 1: uv

```bash
brew install uv

git clone https://github.com/karthik-ravi-1537/LangChain-Platform-Demo.git
cd LangChain-Platform-Demo

uv sync

source .venv/bin/activate
```

### Method 2: conda

```bash
git clone https://github.com/karthik-ravi-1537/LangChain-Platform-Demo.git
cd LangChain-Platform-Demo

conda env create -f environment.yml
conda activate langchain-platform-demo
```

### Configure Environment Variables

```bash
cp .env.example .env
```

Edit `.env` and add your API keys:
```env
OPENAI_API_KEY=your_openai_api_key_here
LANGSMITH_API_KEY=your_langsmith_api_key_here
```

### Verification
```bash
python test_setup.py
```

## Quick Start

```bash
python examples/basic_usage.py
```

This demonstrates:
- Basic LangChain chains
- Custom tools (calculator)
- Conversational agents with memory
- LangSmith tracing (if configured)

## Project Structure

```
LangChain-Platform-Demo/
├── config/              # Application configuration
├── src/
│   ├── chains/         # LangChain chain examples
│   ├── tools/          # Custom tools for agents
│   ├── langsmith/      # LangSmith configuration
│   └── utils/          # Utility modules
├── examples/           # Complete usage demonstrations
└── tests/              # Unit tests
```

## Components

### LangChain Components
- **Basic Chain**: Fact generation with output parsers
- **Calculator Tools**: Mathematical operations with validation
- **Agents**: Conversational agents with tool integration

### LangSmith Integration
- **Automatic Tracing**: All operations traced automatically
- **Performance Monitoring**: Latency, token usage, and costs
- **Debugging**: Step-by-step execution visualization

## Testing

```bash
# Test individual components
python src/chains/basic_chain.py
python src/tools/calculator.py

# Run verification
python test_setup.py
```

## Contributing

### Development Setup

```bash
# Clone and install with dev dependencies
uv sync --dev
source .venv/bin/activate

# Install development tools
uv pip install -e ".[dev]"

# Set up pre-commit hooks
pre-commit install

# Run formatting and linting
pre-commit run --all-files
```

### Development Tools

- **ruff**: Fast Python linter and formatter
- **black**: Code formatter  
- **pre-commit**: Git hooks for code quality
- **pytest**: Testing framework