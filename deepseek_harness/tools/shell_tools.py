"""Execution tools: pwsh and bash.

Both run a command in a subprocess and return its combined stdout/stderr.
Commands run with the working directory set to ``workdir`` (or ``base_dir``).
A ``run_in_background`` flag is accepted for API parity; when set, the tool
returns immediately rather than blocking.
"""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any, List, Optional

from .base import Tool, ToolError, ToolResult

_DEFAULT_TIMEOUT_MS = 120_000


class _ShellTool(Tool):
    """Shared behaviour for command-executing tools."""

    shell: str = ""

    def __init__(self, base_dir: Optional[Path] = None) -> None:
        self.base_dir = Path(base_dir) if base_dir else Path.cwd()

    def _workdir(self, workdir: Optional[str]) -> Path:
        if not workdir:
            return self.base_dir
        candidate = Path(workdir)
        if not candidate.is_absolute():
            candidate = self.base_dir / candidate
        if not candidate.exists():
            raise ToolError(f"Workdir does not exist: {workdir}")
        return candidate

    def _run(self, argv: List[str], workdir: Path, timeout_ms: int) -> ToolResult:
        timeout_s = max(int(timeout_ms or _DEFAULT_TIMEOUT_MS), 1) / 1000.0
        try:
            proc = subprocess.run(
                argv,
                cwd=str(workdir),
                capture_output=True,
                text=True,
                timeout=timeout_s,
            )
        except subprocess.TimeoutExpired:
            raise ToolError(f"Command timed out after {timeout_s:.0f}s.")
        except FileNotFoundError as exc:
            raise ToolError(f"Executable not found: {exc}")

        output = (proc.stdout or "") + (proc.stderr or "")
        marker = f"[exit code: {proc.returncode}]" if proc.returncode else ""
        content = (output.rstrip() + ("\n" if output.strip() and marker else "") + marker).strip()
        result = ToolResult.ok(content or "(no output)", exit_code=proc.returncode)
        if proc.returncode:
            result.is_error = True
        return result


class PwshTool(_ShellTool):
    name = "pwsh"
    description = "Execute a PowerShell command and return its stdout/stderr."
    parameters = {
        "type": "object",
        "properties": {
            "command": {"type": "string", "description": "The PowerShell command to execute."},
            "description": {"type": "string", "description": "Short description of the command."},
            "timeoutMs": {"type": "integer", "description": "Timeout in milliseconds."},
            "workdir": {"type": "string", "description": "Working directory for the command."},
            "run_in_background": {"type": "boolean", "description": "Run without blocking."},
            "sandbox_permissions": {"type": "string", "description": "Wider sandbox mode (parity)."},
            "justification": {"type": "string", "description": "Reason for escalation (parity)."},
        },
        "required": ["command"],
    }

    def run(
        self,
        command: str,
        description: str = "",
        timeoutMs: int = _DEFAULT_TIMEOUT_MS,
        workdir: Optional[str] = None,
        run_in_background: bool = False,
        **_: Any,
    ) -> ToolResult:
        if not command:
            raise ToolError("'command' is required.")
        if run_in_background:
            return ToolResult.ok(
                "Background execution is not supported by this wrapper; "
                "run the command in the foreground instead.",
                background=True,
            )
        return self._run(["pwsh", "-NoProfile", "-Command", command], self._workdir(workdir), timeoutMs)


class BashTool(_ShellTool):
    name = "bash"
    description = "Execute a shell command and return its stdout/stderr."
    parameters = {
        "type": "object",
        "properties": {
            "command": {"type": "string", "description": "The shell command to execute."},
        },
        "required": ["command"],
    }

    def run(self, command: str, **_: Any) -> ToolResult:
        if not command:
            raise ToolError("'command' is required.")
        return self._run(["bash", "-lc", command], self.base_dir, _DEFAULT_TIMEOUT_MS)


def build_shell_tools(base_dir: Optional[Path] = None) -> List[Tool]:
    """Instantiate the execution tool set."""
    return [PwshTool(base_dir), BashTool(base_dir)]