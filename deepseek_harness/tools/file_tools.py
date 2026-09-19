"""File and editing tools: read, write, edit, glob, grep, list_files, read_image.

These mirror the DeepSeek Harness file surface. They operate on paths resolved
by the filesystem backend (by default, relative to ``base_dir``). A small
sandbox guard rejects paths that escape ``base_dir`` unless ``allow_outside``
is set, matching the harness' workspace-write policy.
"""

from __future__ import annotations

import base64
import fnmatch
import mimetypes
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

from .base import Tool, ToolError, ToolResult

_IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".gif"}


class _PathMixin:
    """Resolve and guard filesystem paths against a base directory."""

    def __init__(self, base_dir: Optional[Path] = None, allow_outside: bool = False) -> None:
        self.base_dir = Path(base_dir) if base_dir else Path.cwd()
        self.allow_outside = allow_outside

    def resolve(self, raw: str, must_exist: bool = False) -> Path:
        if not raw:
            raise ToolError("A path is required.")
        candidate = Path(raw)
        if not candidate.is_absolute():
            candidate = self.base_dir / candidate
        candidate = candidate.resolve()

        if not self.allow_outside:
            try:
                candidate.relative_to(self.base_dir.resolve())
            except ValueError:
                raise ToolError(
                    f"Path '{raw}' is outside the workspace ({self.base_dir}); "
                    "sandbox policy denies access."
                )
        if must_exist and not candidate.exists():
            raise ToolError(f"Path does not exist: {raw}")
        return candidate


# ---------------------------------------------------------------------------
# read
# ---------------------------------------------------------------------------
class ReadTool(_PathMixin, Tool):
    name = "read"
    description = "Read a UTF-8 text file and return line-numbered content."
    parameters = {
        "type": "object",
        "properties": {
            "file_path": {"type": "string", "description": "Path to the file to read."},
            "offset": {"type": "integer", "description": "1-based first line to return."},
            "limit": {"type": "integer", "description": "Maximum number of lines."},
        },
        "required": ["file_path"],
    }

    def run(self, file_path: str, offset: int = 1, limit: int = 2000, **_: Any) -> ToolResult:
        path = self.resolve(file_path, must_exist=True)
        if path.is_dir():
            raise ToolError(f"'{file_path}' is a directory; use list_files.")
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            raise ToolError(f"'{file_path}' is not valid UTF-8 text.")

        lines = text.splitlines()
        start = max(int(offset or 1), 1)
        count = max(int(limit or 2000), 1)
        end = min(start - 1 + count, len(lines))
        numbered = [f"{i + 1} | {lines[i]}" for i in range(start - 1, end)]
        body = "\n".join(numbered)
        header = f"(File has {len(lines)} lines total.)\n" if end < len(lines) else ""
        return ToolResult.ok(f"{header}{body}", total_lines=len(lines))


# ---------------------------------------------------------------------------
# write
# ---------------------------------------------------------------------------
class WriteTool(_PathMixin, Tool):
    name = "write"
    description = "Create or fully replace a UTF-8 text file."
    parameters = {
        "type": "object",
        "properties": {
            "file_path": {"type": "string", "description": "Path to write."},
            "content": {"type": "string", "description": "Full UTF-8 file content."},
        },
        "required": ["file_path", "content"],
    }

    def run(self, file_path: str, content: str, **_: Any) -> ToolResult:
        path = self.resolve(file_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
        return ToolResult.ok(f"Wrote {len(content)} chars to {file_path}.", path=str(path))


# ---------------------------------------------------------------------------
# edit
# ---------------------------------------------------------------------------
class EditTool(_PathMixin, Tool):
    name = "edit"
    description = "Edit an existing UTF-8 text file by replacing literal text."
    parameters = {
        "type": "object",
        "properties": {
            "file_path": {"type": "string", "description": "Path to edit."},
            "old_string": {"type": "string", "description": "Literal text to replace."},
            "new_string": {"type": "string", "description": "Replacement text."},
            "replace_all": {"type": "boolean", "description": "Replace every match."},
        },
        "required": ["file_path", "old_string", "new_string"],
    }

    def run(
        self,
        file_path: str,
        old_string: str,
        new_string: str,
        replace_all: bool = False,
        **_: Any,
    ) -> ToolResult:
        path = self.resolve(file_path, must_exist=True)
        text = path.read_text(encoding="utf-8")

        occurrences = text.count(old_string)
        if occurrences == 0:
            raise ToolError(f"old_string not found in {file_path}.")
        if occurrences > 1 and not replace_all:
            raise ToolError(
                f"old_string appears {occurrences} times in {file_path}; "
                "provide more context or set replace_all=true."
            )

        if replace_all:
            new_text = text.replace(old_string, new_string)
            replaced = occurrences
        else:
            new_text = text.replace(old_string, new_string, 1)
            replaced = 1

        path.write_text(new_text, encoding="utf-8")
        return ToolResult.ok(f"Edited {file_path} ({replaced} replacement(s)).", replaced=replaced)


# ---------------------------------------------------------------------------
# glob
# ---------------------------------------------------------------------------
class GlobTool(_PathMixin, Tool):
    name = "glob"
    description = "Find files whose paths match a glob pattern."
    parameters = {
        "type": "object",
        "properties": {
            "pattern": {"type": "string", "description": "Glob pattern, e.g. '**/*.py'."},
            "path": {"type": "string", "description": "Directory to search in."},
        },
        "required": ["pattern"],
    }

    def run(self, pattern: str, path: str = ".", **_: Any) -> ToolResult:
        root = self.resolve(path, must_exist=True)
        if not root.is_dir():
            raise ToolError(f"'{path}' is not a directory.")

        matches: List[Path] = []
        for candidate in root.rglob("*"):
            if candidate.is_file():
                rel = candidate.relative_to(root).as_posix()
                if _match_glob(rel, candidate.name, pattern):
                    matches.append(candidate)

        matches.sort(key=lambda p: p.stat().st_mtime, reverse=True)
        listed = matches[:100]
        lines = [str(p.relative_to(root).as_posix()) for p in listed]
        note = "" if len(matches) <= 100 else f"\n(showing 100 of {len(matches)} matches)"
        return ToolResult.ok("\n".join(lines) + note, count=len(matches))


def _match_glob(rel: str, name: str, pattern: str) -> bool:
    """Match a pattern against a basename or a relative path."""
    if "/" not in pattern:
        return fnmatch.fnmatch(name, pattern)
    return fnmatch.fnmatch(rel, pattern)


# ---------------------------------------------------------------------------
# grep
# ---------------------------------------------------------------------------
class GrepTool(_PathMixin, Tool):
    name = "grep"
    description = "Search file contents with a ripgrep-style regular expression."
    parameters = {
        "type": "object",
        "properties": {
            "pattern": {"type": "string", "description": "Regular expression (ripgrep syntax)."},
            "path": {"type": "string", "description": "File or directory to search."},
            "include": {"type": "string", "description": "Glob filter, e.g. '*.py'."},
        },
        "required": ["pattern"],
    }

    def run(self, pattern: str, path: str = ".", include: str = "*", **_: Any) -> ToolResult:
        root = self.resolve(path, must_exist=True)
        try:
            regex = re.compile(pattern)
        except re.error as exc:
            raise ToolError(f"Invalid regex: {exc}")

        files: List[Path]
        if root.is_file():
            files = [root]
        else:
            files = [p for p in root.rglob("*") if p.is_file() and fnmatch.fnmatch(p.name, include or "*")]

        results: List[str] = []
        total = 0
        for file in files:
            try:
                content = file.read_text(encoding="utf-8")
            except (UnicodeDecodeError, OSError):
                continue
            rel = file.relative_to(root) if root.is_dir() else file.name
            hits = []
            for lineno, line in enumerate(content.splitlines(), start=1):
                if regex.search(line):
                    hits.append(f"{lineno}: {line}")
                    total += 1
                    if total >= 250:
                        break
            if hits:
                results.append(f"{rel}\n" + "\n".join(hits))
            if total >= 250:
                break

        note = "" if total < 250 else "\n(capped at 250 matches)"
        return ToolResult.ok("\n\n".join(results) + note, count=total)


# ---------------------------------------------------------------------------
# list_files
# ---------------------------------------------------------------------------
class ListFilesTool(_PathMixin, Tool):
    name = "list_files"
    description = "List files and directories within a directory."
    parameters = {
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "Directory to list."},
            "recursive": {"type": "boolean", "description": "List recursively."},
        },
        "required": ["path"],
    }

    def run(self, path: str, recursive: bool = False, **_: Any) -> ToolResult:
        root = self.resolve(path, must_exist=True)
        if not root.is_dir():
            raise ToolError(f"'{path}' is not a directory.")

        entries: List[str] = []
        if recursive:
            for item in sorted(root.rglob("*")):
                entries.append(item.relative_to(root).as_posix() + ("/" if item.is_dir() else ""))
        else:
            for item in sorted(root.iterdir()):
                entries.append(item.name + ("/" if item.is_dir() else ""))
        return ToolResult.ok("\n".join(entries) or "(empty)", count=len(entries))


# ---------------------------------------------------------------------------
# read_image
# ---------------------------------------------------------------------------
class ReadImageTool(_PathMixin, Tool):
    name = "read_image"
    description = "Read a PNG/JPEG/WebP/GIF image and return it as a base64 data URI."
    parameters = {
        "type": "object",
        "properties": {
            "file_path": {"type": "string", "description": "Path to the image file."},
        },
        "required": ["file_path"],
    }

    def run(self, file_path: str, **_: Any) -> ToolResult:
        path = self.resolve(file_path, must_exist=True)
        ext = path.suffix.lower()
        if ext not in _IMAGE_EXTS:
            raise ToolError(f"Unsupported image type '{ext}'. Supported: {sorted(_IMAGE_EXTS)}.")
        raw = path.read_bytes()
        mime = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
        data_uri = f"data:{mime};base64,{base64.b64encode(raw).decode('ascii')}"
        return ToolResult.ok(
            f"Loaded image {file_path} ({len(raw)} bytes, {mime}).",
            data_uri=data_uri,
            mime=mime,
            size=len(raw),
        )


# ---------------------------------------------------------------------------
# present
# ---------------------------------------------------------------------------
class PresentTool(_PathMixin, Tool):
    name = "present"
    description = "Declare existing files as final deliverables for the user."
    parameters = {
        "type": "object",
        "properties": {
            "files": {
                "type": "array",
                "description": "List of {path, description} objects.",
                "items": {
                    "type": "object",
                    "properties": {
                        "path": {"type": "string"},
                        "description": {"type": "string"},
                    },
                    "required": ["path"],
                },
            }
        },
        "required": ["files"],
    }

    def run(self, files: List[Dict[str, Any]], **_: Any) -> ToolResult:
        if not isinstance(files, list) or not files:
            raise ToolError("'files' must be a non-empty list of {path, description}.")
        lines = []
        for entry in files:
            if not isinstance(entry, dict) or "path" not in entry:
                raise ToolError("Each file entry must be an object with a 'path'.")
            path = self.resolve(entry["path"], must_exist=True)
            desc = entry.get("description", "")
            lines.append(f"- {path} ({desc})" if desc else f"- {path}")
        return ToolResult.ok("Presented deliverables:\n" + "\n".join(lines), count=len(files))


def build_file_tools(base_dir: Optional[Path] = None, allow_outside: bool = False) -> List[Tool]:
    """Instantiate the file/edit tool set bound to ``base_dir``."""
    return [
        ReadTool(base_dir, allow_outside),
        WriteTool(base_dir, allow_outside),
        EditTool(base_dir, allow_outside),
        GlobTool(base_dir, allow_outside),
        GrepTool(base_dir, allow_outside),
        ListFilesTool(base_dir, allow_outside),
        ReadImageTool(base_dir, allow_outside),
        PresentTool(base_dir, allow_outside),
    ]