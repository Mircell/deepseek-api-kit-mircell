"""Parse textual tool calls emitted by the DeepSeek web model.

DeepSeek's web chat endpoint has no native tool-calling, so the model emits
tool invocations as plain XML-ish blocks, for example::

    <write>
    <file_path>snake-game.html</file_path>
    <content><!DOCTYPE html> ... </content>
    </write>

    <web_search>
    <queries>["a", "b"]</queries>
    </web_search>

This module turns those blocks into the standard OpenAI ``tool_calls`` shape:

    {"id": "toolu_...", "type": "function",
     "function": {"name": "write", "arguments": "{\\"file_path\\": ...}"}}

The parser is *balanced* (it matches nested same-name tags) so it does not
break on HTML ``<`` / ``>`` inside a ``content`` parameter, and it maps common
alternate parameter names (``path`` -> ``file_path``, ``query`` -> ``queries``,
``link`` -> ``url``, ...) back to the canonical schema names.
"""

from __future__ import annotations

import json
import re
import uuid
from typing import Any, Dict, Iterable, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Alternate parameter names the model tends to invent, keyed by canonical name.
# ---------------------------------------------------------------------------
_PARAM_ALIASES: Dict[str, Tuple[str, ...]] = {
    "file_path": ("path", "file", "filepath", "filename", "file_name"),
    "content": ("contents", "text", "data", "body", "value"),
    "queries": ("query", "q", "search", "search_queries", "terms", "keywords"),
    "url": ("link", "href", "uri", "page", "address"),
    "command": ("cmd", "shell", "script", "run"),
    "pattern": ("glob_pattern", "file_pattern", "regex"),
    "include": ("filter", "glob"),
    "old_string": ("old", "old_text", "old_str", "find", "search_string"),
    "new_string": ("new", "new_text", "new_str", "replace", "replacement"),
    "path": ("dir", "directory", "folder"),
    "code": ("source", "program"),
    "language": ("lang",),
    "limit": ("max", "count", "max_results"),
    "offset": ("skip", "start", "start_line"),
    "recursive": ("recurse",),
}

# Fallback schema for well-known tools when the caller does not pass `tools`.
_DEFAULT_TOOL_PARAMS: Dict[str, Tuple[str, ...]] = {
    "write": ("file_path", "content"),
    "read": ("file_path", "offset", "limit"),
    "edit": ("file_path", "old_string", "new_string"),
    "web_search": ("queries",),
    "web_fetch": ("url",),
    "glob": ("pattern", "path"),
    "grep": ("pattern", "path", "include"),
    "pwsh": ("command", "description", "workdir", "timeoutMs", "run_in_background"),
    "bash": ("command",),
    "list_files": ("path", "recursive"),
}

# DSML-style tags (full-width vertical bars around "DSML").
_DSML_INVOKE = r"<｜DSML｜invoke\s+name=\"([^\"]+)\"\s*>(.*?)</｜DSML｜invoke>"
_DSML_PARAM = r"<｜DSML｜parameter\s+name=\"([^\"]+)\"\s+string=\"([^\"]+)\"\s*>(.*?)</｜DSML｜parameter>"
_DSML_TAG = r"<｜DSML｜[^>]*>"

# The literal placeholder some models copy from the prompt guide:
#   <tool_name>web_fetch</tool_name>
# followed by sibling parameter tags. The tool name appears as *text* rather
# than as the tag itself, so the normal balanced-tag pass never sees it.
_LITERAL_TOOL_NAME_RE = re.compile(
    r"<tool_name>\s*([A-Za-z0-9_.\-]+)\s*</tool_name>", re.IGNORECASE
)


# ---------------------------------------------------------------------------
# Balanced tag matching
# ---------------------------------------------------------------------------
def _find_balanced_blocks(text: str, tag_name: str) -> Iterable[Tuple[int, int, int, int]]:
    """Yield ``(open_start, body_start, body_end, close_end)`` for balanced tags.

    Nesting of the *same* tag name is tracked, so a ``content`` parameter that
    contains other markup does not confuse the outer tag match.
    """
    escaped = re.escape(tag_name)
    open_re = re.compile(rf"<{escaped}(?:\s[^<>]*)?>", re.IGNORECASE)
    close_re = re.compile(rf"</{escaped}\s*>", re.IGNORECASE)

    pos = 0
    while True:
        match = open_re.search(text, pos)
        if not match:
            return

        depth = 1
        scan = match.end()
        found_close = None
        while depth > 0:
            next_open = open_re.search(text, scan)
            next_close = close_re.search(text, scan)
            if next_close is None:
                break
            if next_open is not None and next_open.start() < next_close.start():
                depth += 1
                scan = next_open.end()
            else:
                depth -= 1
                if depth == 0:
                    found_close = next_close
                    break
                scan = next_close.end()

        if found_close is None:
            return

        yield match.start(), match.end(), found_close.start(), found_close.end()
        pos = found_close.end()


# ---------------------------------------------------------------------------
# Value / parameter extraction
# ---------------------------------------------------------------------------
def _coerce(value: str) -> Any:
    """Best-effort conversion of a raw tag body into a JSON-friendly value."""
    stripped = value.strip()
    if not stripped:
        return ""
    if stripped[0] in "[{\"" or stripped in ("true", "false", "null") or re.fullmatch(r"-?\d+(?:\.\d+)?", stripped):
        try:
            return json.loads(stripped)
        except (json.JSONDecodeError, ValueError):
            pass
    return stripped


def _extract_params(body: str, param_names: List[str]) -> Dict[str, Any]:
    """Extract the declared parameters from a tool-call body."""
    args: Dict[str, Any] = {}
    canon_set = set(param_names)

    for canon in param_names:
        candidates: List[str] = [canon]
        for alias in _PARAM_ALIASES.get(canon, ()):  # alternate names the model invents
            if alias not in canon_set:  # never steal another real parameter's name
                candidates.append(alias)

        for candidate in candidates:
            blocks = list(_find_balanced_blocks(body, candidate))
            if blocks:
                _, body_start, body_end, _ = blocks[0]
                args[canon] = _coerce(body[body_start:body_end])
                break

    return args


# ---------------------------------------------------------------------------
# Tool schema handling
# ---------------------------------------------------------------------------
def _tool_definition(tool: Any) -> Dict[str, Any]:
    if isinstance(tool, dict):
        inner = tool.get("function")
        if isinstance(inner, dict):
            return inner
        return tool
    return {}


def build_tool_params(tools: Optional[List[Any]]) -> Dict[str, List[str]]:
    """Map each tool name to its declared parameter names.

    Falls back to :data:`_DEFAULT_TOOL_PARAMS` when no schema is supplied or
    when a tool is missing from the supplied schema.
    """
    params: Dict[str, List[str]] = {}

    for tool in tools or []:
        fn = _tool_definition(tool)
        name = fn.get("name")
        if not name:
            continue
        schema = fn.get("parameters") or {}
        properties = schema.get("properties") or {}
        if properties:
            params[name] = list(properties.keys())
        else:
            params.setdefault(name, [])

    for name, defaults in _DEFAULT_TOOL_PARAMS.items():
        params.setdefault(name, list(defaults))

    return params


def _make_call(name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "id": f"toolu_{uuid.uuid4().hex[:8]}",
        "type": "function",
        "function": {
            "name": name,
            "arguments": json.dumps(arguments, ensure_ascii=False),
        },
    }


def _parse_literal_tool_name_calls(
    text: str, tool_params: Dict[str, List[str]]
) -> List[Tuple[int, Dict[str, Any]]]:
    """Recover the malformed ``<tool_name>NAME</tool_name>`` call form.

    Some models copy the literal ``tool_name`` placeholder from the prompt
    guide and emit the tool's real name as *text* inside that tag, followed by
    the parameter tags and a stray closing ``</tool_name>``. This parses that
    shape so the call is not silently dropped by the balanced-tag pass.
    """
    results: List[Tuple[int, Dict[str, Any]]] = []
    known = set(tool_params)

    for match in _LITERAL_TOOL_NAME_RE.finditer(text):
        candidate = match.group(1).strip()
        if candidate not in known:
            continue

        # The sibling parameter region runs from after the name's closing tag
        # to the next opening header or the stray closing tag, whichever is
        # first; otherwise it runs to the end of the text.
        body_start = match.end()
        boundary_positions = [
            pos
            for pos in (
                text.find("<tool_name", body_start),
                text.find("</tool_name>", body_start),
            )
            if pos != -1
        ]
        body_end = min(boundary_positions) if boundary_positions else len(text)

        body = text[body_start:body_end]
        arguments = _extract_params(body, tool_params[candidate])
        if arguments:
            results.append((match.start(), _make_call(candidate, arguments)))

    return results


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def parse_tool_calls_from_text(text: str, tools: Optional[List[Any]] = None) -> List[Dict[str, Any]]:
    """Detect tool calls (DSML or XML) and return OpenAI-style ``tool_calls``."""
    if not text:
        return []

    # 1. DSML-style invocations take precedence.
    dsml_calls: List[Dict[str, Any]] = []
    for invoke in re.finditer(_DSML_INVOKE, text, re.DOTALL):
        tool_name = invoke.group(1)
        invoke_body = invoke.group(2)

        arguments: Dict[str, Any] = {}
        for param in re.finditer(_DSML_PARAM, invoke_body, re.DOTALL):
            arguments[param.group(1)] = _coerce(param.group(3))

        if not arguments:
            try:
                arguments = json.loads(invoke_body.strip())
            except (json.JSONDecodeError, ValueError):
                arguments = {"content": invoke_body.strip()}

        dsml_calls.append(_make_call(tool_name, arguments))

    if dsml_calls:
        return dsml_calls

    # 2. Plain XML-style tool blocks.
    tool_params = build_tool_params(tools)
    found: List[Tuple[int, Dict[str, Any]]] = []

    for tool_name, param_names in tool_params.items():
        for open_start, body_start, body_end, _ in _find_balanced_blocks(text, tool_name):
            body = text[body_start:body_end]
            arguments = _extract_params(body, param_names)
            if not arguments:
                continue  # an empty/irrelevant tag, not a real call
            found.append((open_start, _make_call(tool_name, arguments)))

    # 3. Lenient fallback for the malformed literal form the model sometimes
    #    emits (see _parse_literal_tool_name_calls), so those calls are not lost.
    found.extend(_parse_literal_tool_name_calls(text, tool_params))

    found.sort(key=lambda item: item[0])
    return [call for _, call in found]


def _literal_tool_name_spans(
    text: str, tool_params: Dict[str, List[str]]
) -> List[Tuple[int, int]]:
    """Return ``(start, end)`` spans of malformed ``<tool_name>NAME</tool_name>``
    blocks (including their sibling parameter tags and the stray closing tag),
    so :func:`remove_tool_tags` can strip the same shapes the parser recovers.
    """
    spans: List[Tuple[int, int]] = []
    known = set(tool_params)

    for match in _LITERAL_TOOL_NAME_RE.finditer(text):
        if match.group(1).strip() not in known:
            continue

        body_start = match.end()
        boundary_positions = [
            pos
            for pos in (
                text.find("<tool_name", body_start),
                text.find("</tool_name>", body_start),
            )
            if pos != -1
        ]
        body_end = min(boundary_positions) if boundary_positions else len(text)

        end = body_end
        if text.startswith("</tool_name>", body_end):
            end = body_end + len("</tool_name>")
        spans.append((match.start(), end))

    return spans


def remove_tool_tags(text: str, tools: Optional[List[Any]] = None) -> str:
    """Remove every tool-call block (XML and DSML) from ``text``."""
    if not text:
        return ""

    spans: List[Tuple[int, int]] = []

    tool_params = build_tool_params(tools)
    for tool_name in tool_params:
        for open_start, _, _, close_end in _find_balanced_blocks(text, tool_name):
            spans.append((open_start, close_end))

    # Malformed literal "<tool_name>NAME</tool_name>" blocks (parsed by the
    # fallback) must also be stripped from the visible message.
    spans.extend(_literal_tool_name_spans(text, tool_params))

    for invoke in re.finditer(_DSML_INVOKE, text, re.DOTALL):
        spans.append((invoke.start(), invoke.end()))

    spans.sort()
    pieces: List[str] = []
    last_end = 0
    for start, end in spans:
        if start < last_end:
            last_end = max(last_end, end)
            continue
        pieces.append(text[last_end:start])
        last_end = end
    pieces.append(text[last_end:])

    cleaned = "".join(pieces)
    cleaned = re.sub(_DSML_TAG, "", cleaned)
    return "\n".join(line.rstrip() for line in cleaned.splitlines() if line.strip())