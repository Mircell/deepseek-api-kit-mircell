"""Agent, harness, and interaction tools.

These mirror the DeepSeek Harness orchestration surface. They are intentionally
thin: subagent/workflow/skill/plugin operations need a host runtime to actually
execute, so each accepts an optional callback (injected at build time) and
degrades gracefully to a descriptive message when no host is attached.
"""

from __future__ import annotations

import uuid
from typing import Any, Callable, Dict, List, Optional

from .base import Tool, ToolError, ToolResult

# A host callback signature: (action: str, payload: dict) -> ToolResult | str | None
HostCallback = Callable[[str, Dict[str, Any]], Any]


def _host_result(action: str, payload: Dict[str, Any], host: Optional[HostCallback]) -> ToolResult:
    """Dispatch to the host if attached, otherwise return a stub message."""
    if host is None:
        return ToolResult.ok(
            f"[{action}] no host runtime attached; payload={payload}"
        )
    result = host(action, payload)
    if isinstance(result, ToolResult):
        return result
    return ToolResult.ok(str(result) if result is not None else f"[{action}] done")


# ---------------------------------------------------------------------------
# Subagent tools
# ---------------------------------------------------------------------------
class SubagentTool(Tool):
    name = "subagent"
    description = "Delegate a self-contained task to a subagent working in its own context."
    parameters = {
        "type": "object",
        "properties": {
            "description": {"type": "string", "description": "Short (3-5 word) task description."},
            "prompt": {"type": "string", "description": "The complete, self-contained task."},
            "run_in_background": {"type": "boolean", "description": "Run in the background."},
        },
        "required": ["description", "prompt"],
    }

    def __init__(self, host: Optional[HostCallback] = None) -> None:
        self.host = host

    def run(self, description: str, prompt: str, run_in_background: bool = True, **_: Any) -> ToolResult:
        if not prompt:
            raise ToolError("'prompt' is required.")
        return _host_result(
            "subagent",
            {"description": description, "prompt": prompt, "run_in_background": run_in_background, "agent_id": f"agent-{uuid.uuid4().hex[:8]}"},
            self.host,
        )


class SubagentForkTool(Tool):
    name = "subagent_fork"
    description = "Delegate a task to a subagent that inherits this conversation."
    parameters = {
        "type": "object",
        "properties": {
            "description": {"type": "string"},
            "prompt": {"type": "string"},
            "run_in_background": {"type": "boolean"},
        },
        "required": ["description", "prompt"],
    }

    def __init__(self, host: Optional[HostCallback] = None) -> None:
        self.host = host

    def run(self, description: str, prompt: str, run_in_background: bool = True, **_: Any) -> ToolResult:
        if not prompt:
            raise ToolError("'prompt' is required.")
        return _host_result(
            "subagent_fork",
            {"description": description, "prompt": prompt, "run_in_background": run_in_background, "agent_id": f"agent-{uuid.uuid4().hex[:8]}"},
            self.host,
        )


class SendMessageTool(Tool):
    name = "send_message"
    description = "Send a message to a background subagent, continuing its conversation."
    parameters = {
        "type": "object",
        "properties": {
            "agent_id": {"type": "string"},
            "message": {"type": "string"},
        },
        "required": ["agent_id", "message"],
    }

    def __init__(self, host: Optional[HostCallback] = None) -> None:
        self.host = host

    def run(self, agent_id: str, message: str, **_: Any) -> ToolResult:
        if not agent_id or not message:
            raise ToolError("'agent_id' and 'message' are required.")
        return _host_result("send_message", {"agent_id": agent_id, "message": message}, self.host)


class InterruptAgentTool(Tool):
    name = "interrupt_agent"
    description = "Request cancellation of a background agent's current turn."
    parameters = {
        "type": "object",
        "properties": {"agent_id": {"type": "string"}},
        "required": ["agent_id"],
    }

    def __init__(self, host: Optional[HostCallback] = None) -> None:
        self.host = host

    def run(self, agent_id: str, **_: Any) -> ToolResult:
        if not agent_id:
            raise ToolError("'agent_id' is required.")
        return _host_result("interrupt_agent", {"agent_id": agent_id}, self.host)


class ListAgentsTool(Tool):
    name = "list_agents"
    description = "List continuable background subagents by id and label."
    parameters = {
        "type": "object",
        "properties": {"scope": {"type": "string", "enum": ["children", "descendants"]}},
        "required": [],
    }

    def __init__(self, host: Optional[HostCallback] = None) -> None:
        self.host = host

    def run(self, scope: str = "children", **_: Any) -> ToolResult:
        return _host_result("list_agents", {"scope": scope}, self.host)


class WorkflowTool(Tool):
    name = "workflow"
    description = "Run a JavaScript workflow that orchestrates subagents at scale."
    parameters = {
        "type": "object",
        "properties": {
            "script": {"type": "string", "description": "The plain-JS workflow script body."},
            "meta": {"type": "object", "description": "Workflow identity (name, description, ...)."},
            "args": {"type": "object", "description": "Optional JSON input exposed as args."},
        },
        "required": ["script", "meta"],
    }

    def __init__(self, host: Optional[HostCallback] = None) -> None:
        self.host = host

    def run(self, script: str, meta: Dict[str, Any], args: Optional[Dict[str, Any]] = None, **_: Any) -> ToolResult:
        if not script or not isinstance(meta, dict):
            raise ToolError("'script' (string) and 'meta' (object) are required.")
        return _host_result("workflow", {"script": script, "meta": meta, "args": args or {}}, self.host)


# ---------------------------------------------------------------------------
# Harness / skill / plugin tools
# ---------------------------------------------------------------------------
class SkillTool(Tool):
    name = "skill"
    description = "Load the full instructions for an available skill by name."
    parameters = {
        "type": "object",
        "properties": {"name": {"type": "string"}},
        "required": ["name"],
    }

    def __init__(self, host: Optional[HostCallback] = None) -> None:
        self.host = host

    def run(self, name: str, **_: Any) -> ToolResult:
        if not name:
            raise ToolError("'name' is required.")
        return _host_result("skill", {"name": name}, self.host)


class CordisInspectListTool(Tool):
    name = "cordis_inspect_list"
    description = "List Host and Client service providers."
    parameters = {"type": "object", "properties": {}, "required": []}

    def __init__(self, host: Optional[HostCallback] = None) -> None:
        self.host = host

    def run(self, **_: Any) -> ToolResult:
        return _host_result("cordis_inspect_list", {}, self.host)


class CordisInspectQueryTool(Tool):
    name = "cordis_inspect_query"
    description = "Read the exact API of a service/event/tool/theme/slot."
    parameters = {
        "type": "object",
        "properties": {
            "platform": {"type": "string"},
            "provider": {"type": "string"},
            "method": {"type": "string"},
            "input": {"type": "object"},
        },
        "required": ["provider", "method"],
    }

    def __init__(self, host: Optional[HostCallback] = None) -> None:
        self.host = host

    def run(self, provider: str, method: str, platform: str = "", input: Optional[Dict[str, Any]] = None, **_: Any) -> ToolResult:
        if not provider or not method:
            raise ToolError("'provider' and 'method' are required.")
        return _host_result(
            "cordis_inspect_query",
            {"platform": platform, "provider": provider, "method": method, "input": input or {}},
            self.host,
        )


class PluginManagerTool(Tool):
    name = "plugin_manager"
    description = "Install/remove/enable/disable a bundle."
    parameters = {
        "type": "object",
        "properties": {
            "action": {"type": "string", "enum": ["install", "remove", "enable", "disable", "list"]},
            "target": {"type": "string"},
            "enabled": {"type": "boolean"},
            "approvedBuilds": {"type": "array", "items": {"type": "string"}},
        },
        "required": ["action"],
    }

    def __init__(self, host: Optional[HostCallback] = None) -> None:
        self.host = host

    def run(
        self,
        action: str,
        target: str = "",
        enabled: Optional[bool] = None,
        approvedBuilds: Optional[List[str]] = None,
        **_: Any,
    ) -> ToolResult:
        if not action:
            raise ToolError("'action' is required.")
        return _host_result(
            "plugin_manager",
            {"action": action, "target": target, "enabled": enabled, "approvedBuilds": approvedBuilds or []},
            self.host,
        )


# ---------------------------------------------------------------------------
# Interaction tools
# ---------------------------------------------------------------------------
class AskUserQuestionTool(Tool):
    name = "ask_user_question"
    description = "Ask the user a concise question when you need confirmation or missing information."
    parameters = {
        "type": "object",
        "properties": {
            "questions": {
                "type": "array",
                "description": "Questions to ask, each with a stable id.",
                "items": {
                    "type": "object",
                    "properties": {
                        "id": {"type": "string"},
                        "question": {"type": "string"},
                        "header": {"type": "string"},
                        "options": {"type": "array", "items": {"type": "object"}},
                        "multi_select": {"type": "boolean"},
                    },
                    "required": ["id", "question"],
                },
            }
        },
        "required": ["questions"],
    }

    def __init__(self, host: Optional[HostCallback] = None) -> None:
        self.host = host

    def run(self, questions: List[Dict[str, Any]], **_: Any) -> ToolResult:
        if not isinstance(questions, list) or not questions:
            raise ToolError("'questions' must be a non-empty array.")
        return _host_result("ask_user_question", {"questions": questions}, self.host)


class ExitPlanModeTool(Tool):
    name = "exit_plan_mode"
    description = "Present a plan for the user's review and, on approval, leave plan mode."
    parameters = {
        "type": "object",
        "properties": {"plan": {"type": "string", "description": "The complete plan as markdown."}},
        "required": ["plan"],
    }

    def __init__(self, host: Optional[HostCallback] = None) -> None:
        self.host = host

    def run(self, plan: str, **_: Any) -> ToolResult:
        if not plan:
            raise ToolError("'plan' is required.")
        return _host_result("exit_plan_mode", {"plan": plan}, self.host)


def build_agent_tools(host: Optional[HostCallback] = None) -> List[Tool]:
    """Instantiate the agent/harness/interaction tool set."""
    return [
        SubagentTool(host),
        SubagentForkTool(host),
        SendMessageTool(host),
        InterruptAgentTool(host),
        ListAgentsTool(host),
        WorkflowTool(host),
        SkillTool(host),
        CordisInspectListTool(host),
        CordisInspectQueryTool(host),
        PluginManagerTool(host),
        AskUserQuestionTool(host),
        ExitPlanModeTool(host),
    ]