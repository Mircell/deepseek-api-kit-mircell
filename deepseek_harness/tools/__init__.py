"""DeepSeek Harness tool framework.

This package implements the complete tool surface described for DeepSeek
Harness:

File & editing
    read, write, edit, glob, grep, list_files, read_image, present

Web
    web_search, web_fetch

Execution
    pwsh, bash

Agent orchestration
    subagent, subagent_fork, send_message, interrupt_agent, list_agents, workflow

Goals & tasks
    create_goal, get_goal, update_goal, todo_write

Jobs
    job_list, job_output, job_kill

Harness (Cordis)
    skill, cordis_inspect_list, cordis_inspect_query, plugin_manager

Interaction
    ask_user_question, exit_plan_mode

Each tool is a :class:`~deepseek_harness.tools.base.Tool` with an OpenAI JSON
schema and a ``run`` implementation. ``build_default_registry`` wires every tool
into a single :class:`~deepseek_harness.tools.base.ToolRegistry`, ready to emit
``tools`` schemas for a chat request and execute the returned ``tool_calls``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from .agent_tools import HostCallback, build_agent_tools
from .base import Tool, ToolError, ToolRegistry, ToolResult
from .file_tools import build_file_tools
from .shell_tools import build_shell_tools
from .state_tools import build_state_tools
from .web_tools import build_web_tools

__all__ = [
    "Tool",
    "ToolError",
    "ToolRegistry",
    "ToolResult",
    "HostCallback",
    "build_default_registry",
    "build_file_tools",
    "build_web_tools",
    "build_shell_tools",
    "build_state_tools",
    "build_agent_tools",
]


def build_default_registry(
    base_dir: Optional[Path] = None,
    allow_outside: bool = False,
    host: Optional[HostCallback] = None,
    search_endpoint: str = "https://duckduckgo.com/html/",
) -> ToolRegistry:
    """Assemble every available tool into one registry.

    Args:
        base_dir: Root used for file/shell path resolution (defaults to cwd).
        allow_outside: Permit file tools to touch paths outside ``base_dir``.
        host: Optional callback that executes agent/harness/interaction actions.
        search_endpoint: Search backend used by ``web_search``.

    Returns:
        A registry containing the full DeepSeek Harness tool surface.
    """
    registry = ToolRegistry()

    for tool in build_file_tools(base_dir, allow_outside):
        registry.register(tool)
    for tool in build_web_tools(search_endpoint=search_endpoint):
        registry.register(tool)
    for tool in build_shell_tools(base_dir):
        registry.register(tool)
    for tool in build_agent_tools(host):
        registry.register(tool)

    state_tools, _goal_store, _job_registry = build_state_tools()
    for tool in state_tools:
        registry.register(tool)

    return registry