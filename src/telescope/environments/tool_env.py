"""
Tool Environment - Multi-turn environment with explicit tool calling support.

Extends MultiTurnEnvironment to provide:
- Automatic function-to-OpenAI-tool-schema conversion
- Flexible tool call parsing (XML tags by default, customizable)
- Tool execution with error handling
- Tool usage metrics tracking

Example usage:

    def get_weather(city: str) -> str:
        '''Get the current weather for a city.'''
        return f"The weather in {city} is sunny, 72°F"

    def search(query: str) -> str:
        '''Search for information.'''
        return f"Results for '{query}': ..."

    class MyToolEnv(ToolEnvironment):
        def __init__(self, **kwargs):
            super().__init__(
                tools=[get_weather, search],
                max_turns=5,
                **kwargs
            )
        
        def load_dataset(self, num_samples: int = -1) -> list[Sample]:
            # Load your dataset
            ...
        
        def compute_reward(self, state: RolloutState, eos_token: str = "") -> RewardResult:
            # Compute reward based on final answer
            ...

Tool Call Formats:

The default format uses XML tags that are easy for models to learn:

    <tool_call>
    {"name": "get_weather", "arguments": {"city": "San Francisco"}}
    </tool_call>

Override `parse_tool_calls()` for custom formats (JSON, function syntax, etc.)
"""
import ast
import inspect
import json
import logging
import re
from dataclasses import dataclass
from collections.abc import Callable
from typing import Any, get_type_hints

from telescope.environments.base import (
    MultiTurnEnvironment,
    RolloutState,
    RewardResult,
    Sample,
    ChatMessage,
)


logger = logging.getLogger(__name__)


# =============================================================================
# Tool Schema Utilities
# =============================================================================

def func_to_tool_schema(func: Callable) -> dict:
    """
    Convert a Python function to OpenAI tool schema format.
    
    Uses type hints and docstrings to generate the schema.
    
    Args:
        func: Python function with type hints and docstring
        
    Returns:
        OpenAI tool schema dict
    """
    # Get function signature and type hints
    sig = inspect.signature(func)
    hints = {}
    try:
        hints = get_type_hints(func)
    except Exception:
        pass  # Some functions may not have valid hints
    
    # Build parameters schema
    properties = {}
    required = []
    
    for name, param in sig.parameters.items():
        # Skip self for methods
        if name == "self":
            continue
            
        # Determine JSON schema type from Python type hint
        prop = _python_type_to_json_schema(hints.get(name))
        
        # Add description from docstring if available (basic parsing)
        # Could be enhanced with more sophisticated docstring parsing
        properties[name] = prop
        
        # Required if no default value
        if param.default == inspect.Parameter.empty:
            required.append(name)
    
    # Build the full tool schema
    return {
        "type": "function",
        "function": {
            "name": func.__name__,
            "description": (func.__doc__ or "").strip(),
            "parameters": {
                "type": "object",
                "properties": properties,
                "required": required,
            }
        }
    }


def _python_type_to_json_schema(python_type: Any) -> dict:
    """Convert Python type hint to JSON Schema property."""
    if python_type is None:
        return {"type": "string"}  # Default to string
    
    # Handle basic types
    if python_type == str:
        return {"type": "string"}
    elif python_type == int:
        return {"type": "integer"}
    elif python_type == float:
        return {"type": "number"}
    elif python_type == bool:
        return {"type": "boolean"}
    elif python_type == list or (hasattr(python_type, "__origin__") and python_type.__origin__ == list):
        # Handle List[X]
        args = getattr(python_type, "__args__", None)
        if args:
            return {"type": "array", "items": _python_type_to_json_schema(args[0])}
        return {"type": "array"}
    elif python_type == dict or (hasattr(python_type, "__origin__") and python_type.__origin__ == dict):
        return {"type": "object"}
    else:
        # Default to string for unknown types
        return {"type": "string"}


# =============================================================================
# Tool Call Parsing
# =============================================================================

@dataclass
class ToolCall:
    """Represents a parsed tool call from model output."""
    name: str
    arguments: dict[str, Any]
    raw_text: str = ""  # Original text that was parsed
    
    
@dataclass  
class ToolResult:
    """Result of executing a tool."""
    tool_name: str
    result: str
    success: bool = True
    error: str | None = None


def parse_xml_tool_calls(text: str) -> list[ToolCall]:
    """
    Parse tool calls from XML-tagged format.
    
    Expected format:
        <tool_call>
        {"name": "tool_name", "arguments": {"arg1": "value1"}}
        </tool_call>
    
    Also supports multiple tool calls and whitespace variations.
    """
    pattern = r"<tool_call>\s*(.*?)\s*</tool_call>"
    matches = re.findall(pattern, text, re.DOTALL | re.IGNORECASE)
    
    tool_calls = []
    for match in matches:
        try:
            # Parse JSON content
            data = json.loads(match.strip())
            
            name = data.get("name", "")
            args = data.get("arguments", {})
            
            if name:
                tool_calls.append(ToolCall(
                    name=name,
                    arguments=args,
                    raw_text=match
                ))
        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse tool call JSON: {e}")
            continue
    
    return tool_calls


def parse_function_call_syntax(
    text: str,
    tool_names: set[str] | list[str] | None = None,
) -> list[ToolCall]:
    """
    Parse tool calls from function-call-like syntax using Python's ast module.

    Expected format:
        tool_name(arg1="value1", arg2=123)

    Supports all Python literal argument types: strings, numbers, booleans,
    None, lists, and dicts. Uses ast.parse() for safe, correct parsing
    (no eval).

    Args:
        text: Model completion text to parse
        tool_names: If provided, only match function names in this set.
            This prevents false positives from natural language or code
            snippets that happen to look like function calls.

    Returns:
        List of parsed ToolCall objects
    """
    if tool_names is not None:
        tool_names = set(tool_names)

    # Find function-call-like patterns: word(...)
    # We use a two-phase approach:
    #   1. Find candidates via balanced-paren extraction
    #   2. Parse each candidate safely with ast.parse()
    tool_calls = []
    i = 0
    while i < len(text):
        # Find next identifier followed by '('
        match = re.search(r'\b([a-zA-Z_]\w*)\s*\(', text[i:])
        if not match:
            break

        func_name = match.group(1)
        paren_start = i + match.end() - 1  # index of '('

        # Extract balanced parenthesized expression
        paren_content = _extract_balanced_parens(text, paren_start)
        if paren_content is None:
            # No balanced parens found, skip past this match
            i = paren_start + 1
            continue

        full_call = f"{func_name}{paren_content}"
        i = paren_start + len(paren_content)

        # Filter by known tool names if provided
        if tool_names is not None and func_name not in tool_names:
            continue

        # Parse with ast for safe, correct argument extraction
        try:
            args = _parse_call_with_ast(full_call)
            tool_calls.append(ToolCall(
                name=func_name,
                arguments=args,
                raw_text=full_call,
            ))
        except Exception as e:
            logger.warning(f"Failed to parse function call '{func_name}': {e}")
            continue

    return tool_calls


def _extract_balanced_parens(text: str, start: int) -> str | None:
    """
    Extract a balanced parenthesized expression starting at text[start].

    Handles nested parens, brackets, braces, and string literals.
    Returns the full substring from '(' to matching ')' inclusive,
    or None if no balanced expression is found.
    """
    if start >= len(text) or text[start] != '(':
        return None

    depth = 0
    in_string = None  # None, '"', or "'"
    i = start
    while i < len(text):
        char = text[i]

        # Handle string literals
        if in_string is not None:
            if char == '\\' and i + 1 < len(text):
                i += 2  # Skip escaped character
                continue
            if char == in_string:
                in_string = None
            i += 1
            continue

        if char in ('"', "'"):
            # Check for triple-quoted strings
            if text[i:i+3] in ('"""', "'''"):
                # Triple-quoted: find closing triple
                quote = text[i:i+3]
                end = text.find(quote, i + 3)
                if end == -1:
                    return None  # Unterminated triple-quoted string
                i = end + 3
                continue
            in_string = char
            i += 1
            continue

        if char in ('(', '[', '{'):
            depth += 1
        elif char in (')', ']', '}'):
            depth -= 1
            if depth == 0:
                return text[start:i + 1]
            if depth < 0:
                return None

        i += 1

    return None  # Unbalanced


def _ast_value_to_python(node: ast.expr) -> Any:
    """Recursively convert an AST expression node to a Python value.

    Only allows literal values (constants, dicts, lists, sets, unary ops
    like negative numbers). Raises ValueError for non-literal expressions.
    """
    if isinstance(node, ast.Constant):
        return node.value
    elif isinstance(node, ast.Dict):
        return {
            _ast_value_to_python(k): _ast_value_to_python(v)
            for k, v in zip(node.keys, node.values)
        }
    elif isinstance(node, ast.List):
        return [_ast_value_to_python(el) for el in node.elts]
    elif isinstance(node, ast.Set):
        return {_ast_value_to_python(el) for el in node.elts}  # rare but valid
    elif isinstance(node, ast.Tuple):
        return tuple(_ast_value_to_python(el) for el in node.elts)
    elif isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.USub, ast.UAdd)):
        # Handles negative/positive numbers: -5, +3.14
        operand = _ast_value_to_python(node.operand)
        if isinstance(node.op, ast.USub):
            return -operand
        return +operand
    elif isinstance(node, ast.Name):
        # Handle True, False, None as identifiers (older Python AST compat)
        if node.id == 'True':
            return True
        elif node.id == 'False':
            return False
        elif node.id == 'None':
            return None
        raise ValueError(f"Non-literal identifier: {node.id}")
    else:
        raise ValueError(f"Unsupported AST node type: {type(node).__name__}")


def _parse_call_with_ast(call_str: str) -> dict:
    """Parse a function call string into a dict of keyword arguments using ast.parse().

    Args:
        call_str: e.g. 'tool_name(arg1="hello", arg2=42, data={"k": [1,2]})'

    Returns:
        Dict of argument name → value. Positional arguments are stored
        under numeric string keys ("0", "1", ...).
    """
    tree = ast.parse(call_str, mode='eval')
    if not isinstance(tree.body, ast.Call):
        raise ValueError(f"Not a function call: {call_str}")

    call_node = tree.body
    args = {}

    # Handle positional arguments
    for idx, arg_node in enumerate(call_node.args):
        args[str(idx)] = _ast_value_to_python(arg_node)

    # Handle keyword arguments
    for kw in call_node.keywords:
        if kw.arg is None:
            # **kwargs expansion — not expected in tool calls, skip
            continue
        args[kw.arg] = _ast_value_to_python(kw.value)

    return args


# =============================================================================
# Tool Environment Base Class
# =============================================================================

class ToolEnvironment(MultiTurnEnvironment):
    """
    Multi-turn environment with explicit tool calling support.
    
    Tools are Python functions that are:
    1. Converted to OpenAI tool schema for prompt context
    2. Parsed from model responses using configurable format
    3. Executed with error handling
    4. Results fed back as environment response
    
    Subclasses must implement:
    - load_dataset(): Load your dataset
    - compute_reward(): Compute reward for completed trajectory
    
    Optionally override:
    - parse_tool_calls(): Custom tool call parsing format
    - format_tool_result(): Custom tool result formatting
    - is_final_answer(): Detect when model gives final answer (stops tool use)
    
    Args:
        tools: List of Python functions to expose as tools
        max_turns: Maximum number of turns (tool calls + responses)
        tool_call_format: "xml" (default) or "function" for parsing style
        include_tool_schemas_in_prompt: Whether to add tool descriptions to system prompt
        error_formatter: Function to format tool execution errors
        system_prompt: Base system prompt (tool descriptions appended if enabled)
    """
    
    def __init__(
        self,
        tools: list[Callable] | None = None,
        max_turns: int = 10,
        tool_call_format: str = "xml",  # "xml" or "function"
        include_tool_schemas_in_prompt: bool = True,
        error_formatter: Callable[[Exception], str] | None = None,
        system_prompt: str | None = None,
        **kwargs
    ):
        super().__init__(max_turns=max_turns, system_prompt=system_prompt, **kwargs)
        
        self.tools = tools or []
        self.tool_call_format = tool_call_format
        self.include_tool_schemas_in_prompt = include_tool_schemas_in_prompt
        self.error_formatter = error_formatter or self._default_error_formatter
        
        # Build tool map: name -> function
        self.tool_map: dict[str, Callable] = {}
        for tool in self.tools:
            name = getattr(tool, "__name__", str(tool))
            self.tool_map[name] = tool
        
        # Generate OpenAI-style tool schemas
        self.tool_schemas = [func_to_tool_schema(t) for t in self.tools]
        
        # Build system prompt with tool descriptions
        if self.include_tool_schemas_in_prompt and self.tools:
            tools_section = self._build_tools_prompt_section()
            if self.system_prompt:
                self.system_prompt = f"{self.system_prompt}\n\n{tools_section}"
            else:
                self.system_prompt = tools_section
    
    def _default_error_formatter(self, error: Exception) -> str:
        """Default error formatting for tool execution errors."""
        return f"{type(error).__name__}: {str(error)}"
    
    def _build_tools_prompt_section(self) -> str:
        """
        Build the tools section for the system prompt.
        
        Override this to customize how tools are presented to the model.
        """
        lines = ["# Available Tools", ""]
        lines.append("You have access to the following tools:")
        lines.append("")
        
        for schema in self.tool_schemas:
            func = schema["function"]
            name = func["name"]
            desc = func.get("description", "")
            params = func.get("parameters", {}).get("properties", {})
            required = func.get("parameters", {}).get("required", [])
            
            lines.append(f"## {name}")
            if desc:
                lines.append(f"{desc}")
            lines.append("")
            
            if params:
                lines.append("**Parameters:**")
                for param_name, param_info in params.items():
                    param_type = param_info.get("type", "any")
                    req_marker = " (required)" if param_name in required else ""
                    lines.append(f"- `{param_name}` ({param_type}){req_marker}")
                lines.append("")
        
        lines.append("# Tool Call Format")
        lines.append("")
        
        if self.tool_call_format == "xml":
            lines.append("To use a tool, respond with:")
            lines.append("```")
            lines.append("<tool_call>")
            lines.append('{"name": "tool_name", "arguments": {"arg1": "value1"}}')
            lines.append("</tool_call>")
            lines.append("```")
        else:  # function format
            lines.append("To use a tool, respond with:")
            lines.append("```")
            lines.append('tool_name(arg1="value1", arg2=123)')
            lines.append("```")
        
        lines.append("")
        lines.append("When you have the final answer, respond normally without tool calls.")
        
        return "\n".join(lines)
    
    # =========================================================================
    # Tool Call Parsing (override for custom formats)
    # =========================================================================
    
    def parse_tool_calls(self, completion: str) -> list[ToolCall]:
        """
        Parse tool calls from model completion.
        
        Override this method for custom parsing formats.
        
        Args:
            completion: Model's completion text
            
        Returns:
            List of parsed ToolCall objects
        """
        if self.tool_call_format == "xml":
            return parse_xml_tool_calls(completion)
        elif self.tool_call_format == "function":
            return parse_function_call_syntax(
                completion, tool_names=set(self.tool_map.keys())
            )
        else:
            # Default to XML
            return parse_xml_tool_calls(completion)
    
    def is_final_answer(self, completion: str, state: RolloutState) -> bool:
        """
        Check if the model is providing a final answer (not using tools).
        
        Override this for task-specific final answer detection.
        Default: Returns True if no tool calls were parsed.
        
        Args:
            completion: Model's completion text
            state: Current rollout state
            
        Returns:
            True if this is a final answer, False if tool calls should be processed
        """
        tool_calls = self.parse_tool_calls(completion)
        return len(tool_calls) == 0
    
    # =========================================================================
    # Tool Execution
    # =========================================================================
    
    async def execute_tool(self, tool_call: ToolCall) -> ToolResult:
        """
        Execute a single tool call.

        Override for custom execution logic (e.g., async tools, sandboxing).
        Supports both sync and async tool functions.

        Args:
            tool_call: Parsed tool call to execute

        Returns:
            ToolResult with execution outcome
        """
        tool_name = tool_call.name

        if tool_name not in self.tool_map:
            return ToolResult(
                tool_name=tool_name,
                result=f"Unknown tool: '{tool_name}'. Available tools: {list(self.tool_map.keys())}",
                success=False,
                error="unknown_tool"
            )

        tool_func = self.tool_map[tool_name]

        try:
            import asyncio
            if asyncio.iscoroutinefunction(tool_func):
                result = await tool_func(**tool_call.arguments)
            else:
                result = tool_func(**tool_call.arguments)
            return ToolResult(
                tool_name=tool_name,
                result=str(result),
                success=True
            )
        except TypeError as e:
            # Likely wrong arguments
            return ToolResult(
                tool_name=tool_name,
                result=self.error_formatter(e),
                success=False,
                error="argument_error"
            )
        except Exception as e:
            return ToolResult(
                tool_name=tool_name,
                result=self.error_formatter(e),
                success=False,
                error=str(type(e).__name__)
            )
    
    def format_tool_result(self, result: ToolResult) -> str:
        """
        Format a tool result for the model.
        
        Override for custom result formatting.
        
        Args:
            result: Tool execution result
            
        Returns:
            Formatted string to include in next prompt
        """
        if result.success:
            return f"[{result.tool_name}] {result.result}"
        else:
            return f"[{result.tool_name}] Error: {result.result}"
    
    def format_multiple_tool_results(self, results: list[ToolResult]) -> str:
        """
        Format multiple tool results into a single message.
        
        Override for custom multi-result formatting.
        """
        if len(results) == 1:
            return self.format_tool_result(results[0])
        
        formatted = []
        for i, result in enumerate(results, 1):
            formatted.append(f"Tool {i}: {self.format_tool_result(result)}")
        
        return "\n".join(formatted)
    
    # =========================================================================
    # MultiTurnEnvironment Interface Implementation
    # =========================================================================
    
    def create_initial_state(self, sample: Sample) -> RolloutState:
        """Initialize state with tool tracking."""
        state = super().create_initial_state(sample)
        
        # Initialize tool tracking
        state.custom["tool_calls"] = []  # List of all tool calls made
        state.custom["tool_results"] = []  # List of all tool results
        state.custom["tool_success_count"] = 0
        state.custom["tool_error_count"] = 0
        state.custom["tools_by_name"] = {}  # Count per tool name
        
        return state
    
    async def env_response(
        self,
        messages: list[ChatMessage],
        state: RolloutState,
    ) -> list[ChatMessage]:
        """
        Parse tool calls, execute them, and return results.

        If no tool calls are detected (final answer), returns empty list
        to signal end of rollout.
        """
        # Get the last assistant message
        last_message = messages[-1]
        completion = last_message.get("content", "")

        # Check if this is a final answer (no tool calls)
        if self.is_final_answer(completion, state):
            return []

        # Parse tool calls
        tool_calls = self.parse_tool_calls(completion)

        if not tool_calls:
            # No tool calls found - treat as final answer
            return []

        # Execute each tool call
        results: list[ToolResult] = []
        for call in tool_calls:
            result = await self.execute_tool(call)
            results.append(result)
            
            # Track tool usage
            state.custom["tool_calls"].append({
                "name": call.name,
                "arguments": call.arguments,
                "raw": call.raw_text,
            })
            state.custom["tool_results"].append({
                "name": result.tool_name,
                "result": result.result,
                "success": result.success,
                "error": result.error,
            })
            
            if result.success:
                state.custom["tool_success_count"] += 1
            else:
                state.custom["tool_error_count"] += 1
            
            # Per-tool count
            if result.tool_name not in state.custom["tools_by_name"]:
                state.custom["tools_by_name"][result.tool_name] = 0
            state.custom["tools_by_name"][result.tool_name] += 1
        
        # Format results for next turn
        formatted_results = self.format_multiple_tool_results(results)
        
        return [{
            "role": "user",
            "content": formatted_results,
            "turn_type": "tool_result",
        }]
    
    def is_done(self, state: RolloutState) -> tuple[bool, str | None]:
        """
        Check if rollout should terminate.

        Terminates when max turns reached. Final-answer detection (no tool
        calls in last completion) is handled by env_response returning [].
        """
        if state.num_turns >= self.max_turns:
            return True, "max_turns_reached"

        return False, None
    
    # =========================================================================
    # Reward Computation Helpers
    # =========================================================================
    
    def get_tool_metrics(self, state: RolloutState) -> dict[str, float]:
        """
        Get tool usage metrics from state for reward computation.
        
        Returns:
            Dict with tool usage metrics:
            - total_tool_calls: Total number of tool calls
            - tool_success_rate: Proportion of successful calls
            - unique_tools_used: Number of different tools used
            - per-tool counts: {tool_name}_calls for each tool
        """
        total_calls = len(state.custom.get("tool_calls", []))
        success_count = state.custom.get("tool_success_count", 0)
        tools_by_name = state.custom.get("tools_by_name", {})
        
        metrics = {
            "total_tool_calls": float(total_calls),
            "tool_success_count": float(success_count),
            "tool_error_count": float(state.custom.get("tool_error_count", 0)),
            "tool_success_rate": success_count / total_calls if total_calls > 0 else 0.0,
            "unique_tools_used": float(len(tools_by_name)),
        }
        
        # Add per-tool counts
        for tool_name, count in tools_by_name.items():
            metrics[f"{tool_name}_calls"] = float(count)
        
        return metrics
    
    async def compute_reward(
        self,
        state: RolloutState,
        eos_token: str = "",
    ) -> RewardResult:
        """
        Compute reward for completed trajectory.

        This is an abstract method - subclasses MUST implement task-specific
        reward computation.

        Tool metrics are available via self.get_tool_metrics(state).

        Args:
            state: Completed rollout state with full trajectory
            eos_token: EOS token to strip from completions

        Returns:
            RewardResult with total reward and component breakdown
        """
        raise NotImplementedError(
            "Subclasses must implement compute_reward(). "
            "Use self.get_tool_metrics(state) to access tool usage metrics."
        )
    
    def load_dataset(self, num_samples: int = -1, **kwargs) -> list[Sample]:
        """
        Load and preprocess the dataset.
        
        Subclasses must implement this to provide task-specific data.
        """
        raise NotImplementedError("Subclasses must implement load_dataset()")

