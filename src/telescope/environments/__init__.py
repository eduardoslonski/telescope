"""
Environments module for RL tasks.

Each environment lives in its own subfolder and defines:
- Dataset loading and preprocessing
- Prompt templates
- Reward functions

Environments are auto-discovered from subfolders of this package.
To use one, just specify the folder name in config::

    ENVIRONMENTS = "countdown"
    ENVIRONMENTS = {"name": "hendrycks_math", "kwargs": {...}}
    ENVIRONMENTS = [
        {"name": "hendrycks_math", "weight": 2.0},
        {"name": "countdown", "weight": 1.0},
    ]

To add a new environment, create a subfolder with an ``environment.py``
that defines a concrete ``Environment`` subclass.  No registration or
``__init__.py`` needed.
"""
from telescope.environments.base import (
    # Base classes
    Environment,
    SingleTurnEnvironment,
    MultiTurnEnvironment,
    # Data types
    Sample,
    RewardResult,
    EvalMetricsResult,
    TrajectoryStep,
    RolloutState,
    # Type aliases
    ChatMessage,
    Messages,
)
from telescope.environments.tool_env import (
    # Tool environment base class
    ToolEnvironment,
    # Tool data types
    ToolCall,
    ToolResult,
    # Tool utilities
    func_to_tool_schema,
    parse_xml_tool_calls,
    parse_function_call_syntax,
)
from telescope.environments.parsers import (
    extract_boxed_answer,
    extract_answer_tags,
    extract_code_block,
    extract_xml_tag,
    strip_think_tags,
)
from telescope.environments.registry import get_environment, list_environments, check_environments

__all__ = [
    # Base classes
    "Environment",
    "SingleTurnEnvironment",
    "MultiTurnEnvironment",
    "ToolEnvironment",
    # Data types
    "Sample",
    "RewardResult",
    "EvalMetricsResult",
    "TrajectoryStep",
    "RolloutState",
    "ChatMessage",
    "Messages",
    # Tool types
    "ToolCall",
    "ToolResult",
    # Parsers
    "extract_boxed_answer",
    "extract_answer_tags",
    "extract_code_block",
    "extract_xml_tag",
    "strip_think_tags",
    # Tool utilities
    "func_to_tool_schema",
    "parse_xml_tool_calls",
    "parse_function_call_syntax",
    # Loader
    "get_environment",
    "list_environments",
    "check_environments",
]
