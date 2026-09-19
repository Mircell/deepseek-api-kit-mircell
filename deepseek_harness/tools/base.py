"""Tool framework core: result type, registry, and the ``Tool`` abstraction.

This module provides the plumbing that every DeepSeek Harness tool plugs into:

* :class:`ToolResult` -- a normalized result (text + error flag + optional meta).
* :class:`Tool` -- the protocol each tool implements (``name``, ``schema``,
  ``run``).
* :class:`ToolRegistry` -- a name -> tool map that can emit OpenAI-style
  ``tools`` schemas and dispatch calls.

The goal is that any layer (the proxy, a standalone agent loop, tests) can ask
the registry for the tool schemas and then execute a returned tool call by name.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, List, Optional


@dataclass
class ToolResult:
    """Normalized result returned by every tool.

    Attributes:
        content: Human/LLM-readable text output.
        is_error: True when the tool failed.
        meta: Optional structured data (e.g. parsed matches, file bytes read).
    """

    content: str = ""
    is_error: bool = False
    meta: Dict[str, Any] = field(default_factory=dict)

    def to_message_content(self) -> str:
        """Render the result as the text a tool message carries back to a model."""
        return self.content

    @classmethod
    def ok(cls, content: str = "", **meta: Any) -> "ToolResult":
        return cls(content=content, is_error=False, meta=dict(meta))

    @classmethod
    def error(cls, content: str, **meta: Any) -> "ToolResult":
        return cls(content=content, is_error=True, meta=dict(meta))


class ToolError(Exception):
    """Raised by a tool to signal a handled, user-facing failure."""


class Tool:
    """Base class for a single tool.

    Subclasses set :attr:`name`, :attr:`description`, and :attr:`parameters`
    (a JSON-schema ``properties`` mapping plus optional ``required`` list), and
    implement :meth:`run`.
    """

    name: str = ""
    description: str = ""
    # JSON schema for the tool's arguments (OpenAI ``parameters`` object).
    parameters: Dict[str, Any] = {"type": "object", "properties": {}}

    def run(self, **kwargs: Any) -> ToolResult:  # pragma: no cover - abstract
        raise NotImplementedError

    # -- schema helpers -------------------------------------------------
    def openai_schema(self) -> Dict[str, Any]:
        """Return the OpenAI ``tools[]`` entry for this tool."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }

    def required_params(self) -> List[str]:
        return list(self.parameters.get("required") or [])

    def call(self, arguments: Any) -> ToolResult:
        """Validate and dispatch a call.

        ``arguments`` may be a dict or a JSON string (as produced by models).
        Unknown arguments are ignored; missing required arguments raise a
        handled error result.
        """
        if isinstance(arguments, str):
            try:
                arguments = json.loads(arguments or "{}")
            except (json.JSONDecodeError, ValueError):
                return ToolResult.error(
                    f"Invalid JSON arguments for tool '{self.name}': {arguments!r}"
                )
        if not isinstance(arguments, dict):
            return ToolResult.error(
                f"Tool '{self.name}' expects an object of arguments, got {type(arguments).__name__}"
            )

        missing = [p for p in self.required_params() if p not in arguments]
        if missing:
            return ToolResult.error(
                f"Tool '{self.name}' missing required argument(s): {', '.join(missing)}"
            )

        known = set(self.parameters.get("properties", {}).keys())
        clean = {k: v for k, v in arguments.items() if k in known or not known}

        try:
            return self.run(**clean)
        except ToolError as exc:
            return ToolResult.error(str(exc))
        except Exception as exc:  # noqa: BLE001 - tools must never crash the loop
            return ToolResult.error(f"Tool '{self.name}' failed: {exc}")


class ToolRegistry:
    """A collection of :class:`Tool` instances keyed by name."""

    def __init__(self, tools: Optional[Iterable[Tool]] = None) -> None:
        self._tools: Dict[str, Tool] = {}
        for tool in tools or []:
            self.register(tool)

    def register(self, tool: Tool) -> None:
        if not tool.name:
            raise ValueError("Tool must define a non-empty name")
        self._tools[tool.name] = tool

    def unregister(self, name: str) -> None:
        self._tools.pop(name, None)

    def get(self, name: str) -> Optional[Tool]:
        return self._tools.get(name)

    def names(self) -> List[str]:
        return list(self._tools.keys())

    def tools(self) -> List[Tool]:
        return list(self._tools.values())

    def openai_tools(self, names: Optional[Iterable[str]] = None) -> List[Dict[str, Any]]:
        """Return OpenAI ``tools`` schemas, optionally filtered by name."""
        if names is None:
            return [t.openai_schema() for t in self._tools.values()]
        wanted = set(names)
        return [t.openai_schema() for n, t in self._tools.items() if n in wanted]

    def execute(self, name: str, arguments: Any) -> ToolResult:
        """Execute a tool by name with dict/JSON arguments."""
        tool = self._tools.get(name)
        if tool is None:
            return ToolResult.error(f"Unknown tool: {name}")
        return tool.call(arguments)

    def execute_tool_call(self, tool_call: Dict[str, Any]) -> ToolResult:
        """Execute an OpenAI ``tool_calls[]`` entry."""
        fn = (tool_call or {}).get("function") or {}
        return self.execute(fn.get("name", ""), fn.get("arguments", "{}"))