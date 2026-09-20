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

The model frequently ignores that guide and falls back to its own native
"DSML" dialect, which carries the same information inside full-width-bar
tags::

    <|DSML|invoke name="web_fetch">
    <|DSML|parameter name="url" string="true">https://example.com</|DSML|parameter>
    </|DSML|invoke>

Real emissions of that dialect are sloppy: a space after the bar
(``<|DSML| invoke``), an optional/missing ``string`` attribute, a
``<|DSML|calls>`` wrapper, a stray closing tag, and — for large values like a
file's ``content`` — a **missing or embedded parameter closing tag**. To
survive that, DSML parameters are not matched with balanced tags; each
parameter's value simply runs from its own opening tag to the *next*
parameter's opening tag (or to the end of the invoke), so a missing closing
tag never drops the call.

Both dialects become the standard OpenAI ``tool_calls`` shape:

    {"id": "toolu_...", "type": "function",
     "function": {"name": "write", "arguments": "{\\"file_path\\": ...}"}}

The parser is *balanced* for the plain-XML dialect (it matches nested
same-name tags) so it does not break on HTML ``<`` / ``>`` inside a
``content`` parameter, and it maps common alternate parameter names
(``path`` -> ``file_path``, ``query`` -> ``queries``, ``link`` -> ``url``,
...) back to the canonical schema names.
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

# Reverse lookup: an invented name -> the canonical name it stands for.
# ``setdefault`` keeps the first owner of a name, so ``path`` stays an alias of
# ``file_path`` even though ``path`` is itself canonical for ``glob``/``grep``.
# A declared parameter always wins over this table (see :func:`_canonical_param`).
_ALIAS_TO_CANON: Dict[str, str] = {}
for _canon, _aliases in _PARAM_ALIASES.items():
    for _alias in _aliases:
        _ALIAS_TO_CANON.setdefault(_alias.lower(), _canon)

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

# The full-width vertical bar the model's native dialect is built from.
_BAR = "\uff5c"
_DSML = f"{_BAR}DSML{_BAR}"

# Any native-dialect tag (opening, closing, or the ``calls`` wrapper).
_DSML_TAG_RE = re.compile(rf"</?{_DSML}\s*[^>]*>", re.IGNORECASE)

# Individual native-dialect tags. The opening tags capture their attribute
# string (everything up to the first ``>``); the ``name`` attribute is then
# read out with :data:`_NAME_ATTR_RE`.
_DSML_INVOKE_OPEN_RE = re.compile(rf"<{_DSML}\s*invoke\b([^>]*)>", re.IGNORECASE)
_DSML_INVOKE_CLOSE_RE = re.compile(rf"</{_DSML}\s*invoke\s*>", re.IGNORECASE)
_DSML_PARAM_OPEN_RE = re.compile(rf"<{_DSML}\s*parameter\b([^>]*)>", re.IGNORECASE)
# A parameter closing tag at the very end of a value (with optional trailing
# whitespace) is the model's own delimiter and must be stripped from the value.
_DSML_PARAM_TRAILING_CLOSE_RE = re.compile(
    rf"</{_DSML}\s*parameter\s*>\s*$", re.IGNORECASE
)

# Attributes on a native-dialect tag, e.g. name="url" string="true".
_NAME_ATTR_RE = re.compile(r"""\bname\s*=\s*(?:"([^"]*)"|'([^']*)')""", re.IGNORECASE)

# The literal placeholder some models copy from the prompt guide:
#   <tool_name>web_fetch</tool_name>
# followed by sibling parameter tags. The tool name appears as *text* rather
# than as the tag itself, so the normal balanced-tag pass never sees it.
_LITERAL_TOOL_NAME_RE = re.compile(
    r"<tool_name>\s*([A-Za-z0-9_.\-]+)\s*</tool_name>", re.IGNORECASE
)


# ---------------------------------------------------------------------------
# Balanced tag matching (plain-XML dialect)
# ---------------------------------------------------------------------------
def _find_balanced_blocks(text: str, tag_name: str) -> Iterable[Tuple[int, int, int, int]]:
    """Yield ``(open_start, body_start, body_end, close_end)`` for XML tags.

    Nesting of the *same* tag name is tracked, so a ``content`` parameter that
    contains other markup does not confuse the outer tag match. A region whose
    closing tag is missing is skipped entirely.
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
# Native DSML dialect (lenient, delimiter-based)
# ---------------------------------------------------------------------------
def _attr_name(attributes: str) -> Optional[str]:
    """Read the ``name`` attribute out of a native-dialect tag's attributes."""
    match = _NAME_ATTR_RE.search(attributes)
    if not match:
        return None
    value = match.group(1) if match.group(1) is not None else match.group(2)
    value = (value or "").strip()
    return value or None


def _dsml_invoke_regions(text: str) -> Iterable[Tuple[int, int]]:
    """Yield ``(start, end)`` spans of native-dialect invocations.

    The end is the matching ``</|DSML|invoke>`` when present; otherwise the
    span runs to the next invocation or to the end of the text, so a missing
    closing tag never hides a call.
    """
    pos = 0
    while True:
        open_match = _DSML_INVOKE_OPEN_RE.search(text, pos)
        if not open_match:
            return

        next_open = _DSML_INVOKE_OPEN_RE.search(text, open_match.end())
        close_match = _DSML_INVOKE_CLOSE_RE.search(text, open_match.end())

        if close_match is not None and (
            next_open is None or close_match.start() < next_open.start()
        ):
            end = close_match.end()
        else:
            end = next_open.start() if next_open is not None else len(text)

        yield open_match.start(), end
        pos = end if end > open_match.end() else open_match.end()


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
    """Extract the declared parameters from a plain-XML tool-call body."""
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


def _extract_dsml_params(body: str) -> Dict[str, Any]:
    """Extract native-dialect ``parameter`` values from an invocation body.

    Each value runs from its own opening tag to the next parameter's opening
    tag (or the end of the body), and any trailing ``</|DSML|parameter>`` is
    stripped. This does not require the model to emit a closing tag for every
    parameter, which is exactly what breaks on long ``content`` values.
    """
    raw: Dict[str, Any] = {}
    opens = list(_DSML_PARAM_OPEN_RE.finditer(body))

    for index, open_match in enumerate(opens):
        param_name = _attr_name(open_match.group(1) or "")
        if not param_name:
            continue
        value_start = open_match.end()
        value_end = opens[index + 1].start() if index + 1 < len(opens) else len(body)
        value = body[value_start:value_end]
        value = _DSML_PARAM_TRAILING_CLOSE_RE.sub("", value)
        raw[param_name] = _coerce(value)

    return raw


def _canonical_param(name: str, param_names: List[str]) -> Optional[str]:
    """Map an emitted parameter name onto a declared one, or ``None`` if unknown.

    A declared name always wins, which is what lets the same word mean
    ``file_path`` for ``write`` and ``path`` for ``glob``.
    """
    if not param_names:
        return name
    declared = {param.lower(): param for param in param_names}
    lowered = name.lower()
    if lowered in declared:
        return declared[lowered]
    canon = _ALIAS_TO_CANON.get(lowered)
    if canon is not None and canon.lower() in declared:
        return declared[canon.lower()]
    return None


def _canonicalize(
    raw: Dict[str, Any], param_names: List[str], strict: bool
) -> Dict[str, Any]:
    """Rewrite/keep emitted parameter names against the declared schema.

    ``strict`` is set when the request actually declared a schema for this
    tool; undeclared names are then dropped rather than forwarded to the
    harness, which would reject the call with ``INVALID_ARGS``.
    """
    if not strict or not param_names:
        return dict(raw)
    result: Dict[str, Any] = {}
    for key, value in raw.items():
        canon = _canonical_param(key, param_names)
        if canon is not None:
            result[canon] = value
    return result


def _maybe_json_object(value: str) -> Optional[Dict[str, Any]]:
    """Parse ``value`` as a JSON object, or return ``None``."""
    if not value.startswith("{"):
        return None
    try:
        parsed = json.loads(value)
    except (json.JSONDecodeError, ValueError):
        return None
    return parsed if isinstance(parsed, dict) else None


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


def _declared_tool_names(tools: Optional[List[Any]]) -> set:
    """Names the request itself declared (ignoring the built-in fallbacks)."""
    names = set()
    for tool in tools or []:
        fn = _tool_definition(tool)
        name = fn.get("name")
        if name:
            names.add(name)
    return names


def _make_call(name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "id": f"toolu_{uuid.uuid4().hex[:8]}",
        "type": "function",
        "function": {
            "name": name,
            "arguments": json.dumps(arguments, ensure_ascii=False),
        },
    }


# ---------------------------------------------------------------------------
# Native DSML dialect
# ---------------------------------------------------------------------------
def _parse_dsml_calls(
    text: str,
    tool_params: Dict[str, List[str]],
    declared_names: set,
) -> List[Tuple[int, int, Dict[str, Any]]]:
    """Parse the model's native ``<|DSML|invoke>`` dialect into tool calls.

    Returns ``(start, end, call)`` triples so the caller can merge these with
    the XML dialect results in source order.
    """
    results: List[Tuple[int, int, Dict[str, Any]]] = []
    # A schema was declared: only honour tools it actually offers, so a
    # hallucinated name never reaches the harness.
    strict_schema = bool(declared_names)

    for start, end in _dsml_invoke_regions(text):
        invoke = text[start:end]
        open_match = _DSML_INVOKE_OPEN_RE.match(invoke)
        if open_match is None:
            continue
        tool_name = _attr_name(open_match.group(1) or "")
        if not tool_name:
            continue
        if strict_schema and tool_name not in declared_names:
            continue

        body = invoke[open_match.end():]
        # Drop the invocation's own closing tag from the body, if present.
        body = _DSML_INVOKE_CLOSE_RE.sub("", body, count=1)

        raw = _extract_dsml_params(body)
        param_names = tool_params.get(tool_name, [])
        if raw:
            arguments = _canonicalize(raw, param_names, strict_schema)
        else:
            parsed = _maybe_json_object(body.strip())
            if parsed is not None:
                arguments = _canonicalize(parsed, param_names, strict_schema)
            else:
                arguments = {}

        # A tool that declares parameters but received none is a malformed
        # emission, not a call — forwarding it only produces INVALID_ARGS.
        if not arguments and param_names:
            continue

        results.append((start, end, _make_call(tool_name, arguments)))

    return results


# ---------------------------------------------------------------------------
# Literal ``<tool_name>NAME</tool_name>`` recovery
# ---------------------------------------------------------------------------
def _literal_tool_name_region(text: str, match: "re.Match[str]") -> Tuple[int, int]:
    """Span of the sibling parameter region following a literal ``tool_name`` tag."""
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
    return body_start, body_end


def _parse_literal_tool_name_calls(
    text: str, tool_params: Dict[str, List[str]]
) -> List[Tuple[int, int, Dict[str, Any]]]:
    """Recover the malformed ``<tool_name>NAME</tool_name>`` call form.

    Some models copy the literal ``tool_name`` placeholder from the prompt
    guide and emit the tool's real name as *text* inside that tag, followed by
    the parameter tags and a stray closing ``</tool_name>``. This parses that
    shape so the call is not silently dropped by the balanced-tag pass.
    """
    results: List[Tuple[int, int, Dict[str, Any]]] = []
    known = set(tool_params)

    for match in _LITERAL_TOOL_NAME_RE.finditer(text):
        candidate = match.group(1).strip()
        if candidate not in known:
            continue

        body_start, body_end = _literal_tool_name_region(text, match)
        body = text[body_start:body_end]
        param_names = tool_params[candidate]
        arguments = _extract_params(body, param_names)
        if arguments or not param_names:
            results.append((match.start(), body_end, _make_call(candidate, arguments)))

    return results


def _literal_tool_name_spans(
    text: str, tool_params: Dict[str, List[str]]
) -> List[Tuple[int, int]]:
    """Return spans of malformed literal-form blocks so they can be stripped."""
    spans: List[Tuple[int, int]] = []
    known = set(tool_params)

    for match in _LITERAL_TOOL_NAME_RE.finditer(text):
        if match.group(1).strip() not in known:
            continue

        _, body_end = _literal_tool_name_region(text, match)
        end = body_end
        if text.startswith("</tool_name>", body_end):
            end = body_end + len("</tool_name>")
        spans.append((match.start(), end))

    return spans


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def parse_tool_calls_from_text(text: str, tools: Optional[List[Any]] = None) -> List[Dict[str, Any]]:
    """Detect tool calls (native DSML or plain XML) and return OpenAI ``tool_calls``.

    Both dialects are collected in one pass and returned in source order, so a
    reply that mixes them does not lose the plain-XML calls.
    """
    if not text:
        return []

    tool_params = build_tool_params(tools)
    declared_names = _declared_tool_names(tools)
    found: List[Tuple[int, int, Dict[str, Any]]] = []

    # 1. The model's native DSML dialect.
    found.extend(_parse_dsml_calls(text, tool_params, declared_names))

    # 2. Plain XML-style tool blocks.
    for tool_name, param_names in tool_params.items():
        for open_start, body_start, body_end, close_end in _find_balanced_blocks(text, tool_name):
            body = text[body_start:body_end]
            arguments = _extract_params(body, param_names)
            if not arguments and param_names:
                continue  # an empty/irrelevant tag, not a real call
            found.append((open_start, close_end, _make_call(tool_name, arguments)))

    # 3. Lenient fallback for the malformed literal form the model sometimes
    #    emits (see _parse_literal_tool_name_calls), so those calls are not lost.
    found.extend(_parse_literal_tool_name_calls(text, tool_params))

    found.sort(key=lambda item: (item[0], item[1]))

    calls: List[Dict[str, Any]] = []
    last_end = -1
    for start, end, call in found:
        if start < last_end:
            continue  # overlaps a call already accepted
        calls.append(call)
        last_end = max(end, start + 1)
    return calls


def remove_tool_tags(text: str, tools: Optional[List[Any]] = None) -> str:
    """Remove every tool-call block (plain XML and native DSML) from ``text``."""
    if not text:
        return ""

    spans: List[Tuple[int, int]] = []

    tool_params = build_tool_params(tools)
    for tool_name in tool_params:
        for open_start, _, _, close_end in _find_balanced_blocks(text, tool_name):
            spans.append((open_start, close_end))

    # Native-dialect invocations (lenient: a missing close runs to the next
    # invocation or to the end of the text).
    spans.extend(_dsml_invoke_regions(text))

    # Malformed literal "<tool_name>NAME</tool_name>" blocks (parsed by the
    # fallback) must also be stripped from the visible message.
    spans.extend(_literal_tool_name_spans(text, tool_params))

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
    # Drop any leftover native-dialect scaffolding (the ``calls`` wrapper, a
    # stray closing tag) that was not part of a recognised invocation.
    cleaned = _DSML_TAG_RE.sub("", cleaned)
    return "\n".join(line.rstrip() for line in cleaned.splitlines() if line.strip())