# My Dynamic Multi-Agent System

A Python-based dynamic multi-agent system.

## Overview

This project aims to create a flexible and adaptable multi-agent system where agents, tools, and even flow control can be dynamically defined and composed at runtime.

## Getting Started

1.  Clone the repository.
2.  Install dependencies using Poetry: `poetry install`
3.  ... (more instructions to come)


### Folder Structure

```
my_dynamic_mas/
├── pyproject.toml         # Poetry project configuration
├── poetry.lock           # Poetry lock file
├── src/
│   └── dynamic_mas/
│       ├── __init__.py
│       ├── core/             # Core system components
│       │   ├── __init__.py
│       │   ├── agent_manager.py  # Manages agent instances and lifecycle
│       │   ├── task_orchestrator.py # Orchestrates the flow of tasks
│       │   ├── flow_controller.py # Interprets and executes flow control structures
│       │   ├── function_registry.py # Holds dynamically created/registered functions (tools, conditions, actions)
│       │   ├── data_models/      # Pydantic models for core data structures
│       │   │   ├── __init__.py
│       │   │   ├── task.py
│       │   │   ├── subtask.py
│       │   │   ├── flow_control.py
│       │   │   └── ...
│       ├── agents/           # Implementations of different agents (as functions or classes)
│       │   ├── __init__.py
│       │   ├── ceo_agent.py
│       │   ├── product_creator_agent.py
│       │   ├── product_finder_agent.py
│       │   ├── order_processor_agent.py
│       │   └── ...
│       ├── tools/            # Implementations of reusable tools (API callers, etc.)
│       │   ├── __init__.py
│       │   ├── api_utils.py
│       │   ├── web_search_utils.py
│       │   ├── database_utils.py
│       │   └── ...
│       ├── flow_control_elements/ # Implementations of condition and action functions
│       │   ├── __init__.py
│       │   ├── conditions/
│       │   │   ├── __init__.py
│       │   │   ├── inventory_conditions.py
│       │   │   ├── payment_conditions.py
│       │   │   └── ...
│       │   ├── actions/
│       │   │   ├── __init__.py
│       │   │   ├── order_actions.py
│       │   │   ├── product_actions.py
│       │   │   └── ...
│       ├── dynamic_creation/   # Modules for runtime creation of components
│       │   ├── __init__.py
│       │   ├── function_factory.py # For dynamically creating functions
│       │   ├── model_factory.py    # For dynamically creating Pydantic models
│       │   └── flow_factory.py     # For dynamically creating flow control structures
│       ├── config/           # Configuration files (e.g., API keys)
│       │   ├── __init__.py
│       │   └── config.yaml
│       ├── main.py           # Entry point of the application
│       └── ...
├── tests/
│   ├── __init__.py
│   ├── core/
│   ├── agents/
│   ├── tools/
│   └── ...
├── README.md
└── .gitignore
```