"""Goal, task-list, and background-job tools.

These tools keep lightweight in-process state so they can be used standalone:

* ``todo_write`` stores the latest structured todo list.
* ``create_goal`` / ``get_goal`` / ``update_goal`` manage one session goal.
* ``job_list`` / ``job_output`` / ``job_kill`` read a shared job registry that
  a caller (e.g. :class:`~deepseek_harness.tools.shell_tools.PwshTool`) can feed.
"""

from __future__ import annotations

import time
import uuid
from typing import Any, Dict, List, Optional

from .base import Tool, ToolError, ToolResult


# ---------------------------------------------------------------------------
# todo_write
# ---------------------------------------------------------------------------
class TodoWriteTool(Tool):
    name = "todo_write"
    description = "Record and update a structured task list for the current work."
    parameters = {
        "type": "object",
        "properties": {
            "todos": {
                "type": "array",
                "description": "The complete todo list (replaces the previous one).",
                "items": {
                    "type": "object",
                    "properties": {
                        "content": {"type": "string"},
                        "status": {"type": "string", "enum": ["pending", "in_progress", "completed"]},
                    },
                    "required": ["content", "status"],
                },
            }
        },
        "required": ["todos"],
    }

    def __init__(self) -> None:
        self.todos: List[Dict[str, Any]] = []

    def run(self, todos: List[Dict[str, Any]], **_: Any) -> ToolResult:
        if not isinstance(todos, list):
            raise ToolError("'todos' must be an array.")
        for item in todos:
            if not isinstance(item, dict) or "content" not in item or "status" not in item:
                raise ToolError("Each todo needs 'content' and 'status'.")
            if item["status"] not in ("pending", "in_progress", "completed"):
                raise ToolError(f"Invalid status: {item['status']}")
        self.todos = todos
        done = sum(1 for t in todos if t["status"] == "completed")
        return ToolResult.ok(f"Todo list updated ({done}/{len(todos)} complete).", todos=todos)


# ---------------------------------------------------------------------------
# Goal tools
# ---------------------------------------------------------------------------
class GoalStore:
    """A tiny single-session goal container shared by the goal tools."""

    def __init__(self) -> None:
        self.goal: Optional[Dict[str, Any]] = None

    def _require(self, goal_id: str) -> Dict[str, Any]:
        if not self.goal or self.goal["goal_id"] != goal_id:
            raise ToolError(f"No active goal with id {goal_id!r}.")
        return self.goal


class CreateGoalTool(Tool):
    name = "create_goal"
    description = "Create one persisted same-session completion goal."
    parameters = {
        "type": "object",
        "properties": {
            "objective": {"type": "string", "description": "The concrete completion objective."},
            "max_goal_rounds": {"type": "number", "description": "Optional continuation-round limit."},
        },
        "required": ["objective"],
    }

    def __init__(self, store: GoalStore) -> None:
        self.store = store

    def run(self, objective: str, max_goal_rounds: Optional[float] = None, **_: Any) -> ToolResult:
        if not objective:
            raise ToolError("'objective' is required.")
        self.store.goal = {
            "goal_id": f"goal-{uuid.uuid4().hex[:8]}",
            "revision": 1,
            "objective": objective,
            "phase": "active",
            "rounds_completed": 0,
            "max_goal_rounds": int(max_goal_rounds) if max_goal_rounds else None,
            "created": time.time(),
        }
        g = self.store.goal
        return ToolResult.ok(f"Goal created: {g['goal_id']}", goal=g)


class GetGoalTool(Tool):
    name = "get_goal"
    description = "Read the current same-session goal and its exact id/revision."
    parameters = {"type": "object", "properties": {}, "required": []}

    def __init__(self, store: GoalStore) -> None:
        self.store = store

    def run(self, **_: Any) -> ToolResult:
        if not self.store.goal:
            return ToolResult.ok("No active goal.", goal=None)
        return ToolResult.ok(f"Active goal: {self.store.goal['goal_id']}", goal=self.store.goal)


class UpdateGoalTool(Tool):
    name = "update_goal"
    description = "Update the current goal (edit/pause/resume/complete/blocked)."
    parameters = {
        "type": "object",
        "properties": {
            "goal_id": {"type": "string"},
            "revision": {"type": "number"},
            "action": {"type": "string", "enum": ["edit", "pause", "resume", "complete", "blocked"]},
            "objective": {"type": "string", "description": "Replacement objective (edit only)."},
            "max_goal_rounds": {"type": "number", "description": "Replacement cap (edit only)."},
            "blocked_reason": {"type": "string", "description": "Required with action=blocked."},
        },
        "required": ["goal_id", "revision", "action"],
    }

    def __init__(self, store: GoalStore) -> None:
        self.store = store

    def run(
        self,
        goal_id: str,
        revision: int,
        action: str,
        objective: Optional[str] = None,
        max_goal_rounds: Optional[float] = None,
        blocked_reason: Optional[str] = None,
        **_: Any,
    ) -> ToolResult:
        goal = self.store._require(goal_id)
        if int(revision) != goal["revision"]:
            raise ToolError(f"Revision mismatch: expected {goal['revision']}, got {revision}.")

        if action == "edit":
            if objective:
                goal["objective"] = objective
            if max_goal_rounds is not None:
                goal["max_goal_rounds"] = int(max_goal_rounds)
        elif action == "pause":
            goal["phase"] = "paused"
        elif action == "resume":
            goal["phase"] = "active"
        elif action == "complete":
            goal["phase"] = "complete"
        elif action == "blocked":
            if not blocked_reason:
                raise ToolError("'blocked_reason' is required for action=blocked.")
            goal["phase"] = "blocked"
            goal["blocked_reason"] = blocked_reason
        else:
            raise ToolError(f"Unknown action: {action}")

        goal["revision"] += 1
        return ToolResult.ok(f"Goal {goal_id} -> {goal['phase']} (rev {goal['revision']}).", goal=goal)


# ---------------------------------------------------------------------------
# Job tools
# ---------------------------------------------------------------------------
class Job:
    def __init__(self, job_id: str, kind: str = "generic", output: str = "") -> None:
        self.id = job_id
        self.kind = kind
        self.output = output
        self.status = "running"


class JobRegistry:
    """Shared registry of background jobs that tools can inspect."""

    def __init__(self) -> None:
        self.jobs: Dict[str, Job] = {}

    def start(self, kind: str = "generic", output: str = "") -> Job:
        job = Job(f"job-{uuid.uuid4().hex[:8]}", kind=kind, output=output)
        self.jobs[job.id] = job
        return job


class JobListTool(Tool):
    name = "job_list"
    description = "List background jobs with their ids, kinds, and statuses."
    parameters = {"type": "object", "properties": {}, "required": []}

    def __init__(self, registry: JobRegistry) -> None:
        self.registry = registry

    def run(self, **_: Any) -> ToolResult:
        if not self.registry.jobs:
            return ToolResult.ok("No background jobs.", jobs=[])
        lines = [f"- {j.id} [{j.kind}] {j.status}" for j in self.registry.jobs.values()]
        return ToolResult.ok("\n".join(lines), jobs=[{"id": j.id, "kind": j.kind, "status": j.status} for j in self.registry.jobs.values()])


class JobOutputTool(Tool):
    name = "job_output"
    description = "Read the output of a background job."
    parameters = {
        "type": "object",
        "properties": {
            "job_id": {"type": "string"},
            "wait": {"type": "boolean"},
            "timeout_ms": {"type": "number"},
        },
        "required": ["job_id"],
    }

    def __init__(self, registry: JobRegistry) -> None:
        self.registry = registry

    def run(self, job_id: str, wait: bool = False, timeout_ms: Optional[float] = None, **_: Any) -> ToolResult:
        job = self.registry.jobs.get(job_id)
        if job is None:
            raise ToolError(f"Unknown job: {job_id}")
        return ToolResult.ok(f"{job.output}\n[status: {job.status}]", status=job.status)


class JobKillTool(Tool):
    name = "job_kill"
    description = "Request cancellation of a running background job."
    parameters = {
        "type": "object",
        "properties": {
            "job_id": {"type": "string"},
            "reason": {"type": "string"},
        },
        "required": ["job_id"],
    }

    def __init__(self, registry: JobRegistry) -> None:
        self.registry = registry

    def run(self, job_id: str, reason: str = "", **_: Any) -> ToolResult:
        job = self.registry.jobs.get(job_id)
        if job is None:
            raise ToolError(f"Unknown job: {job_id}")
        job.status = "killed"
        return ToolResult.ok(f"Job {job_id} killed. {reason}".strip(), status=job.status)


def build_state_tools():
    """Return (tools, goal_store, job_registry) for the state tool set."""
    store = GoalStore()
    registry = JobRegistry()
    tools: List[Tool] = [
        TodoWriteTool(),
        CreateGoalTool(store),
        GetGoalTool(store),
        UpdateGoalTool(store),
        JobListTool(registry),
        JobOutputTool(registry),
        JobKillTool(registry),
    ]
    return tools, store, registry