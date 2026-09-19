"""OpenAI-compatible provider backed by the DeepSeek web API.

Built on top of ``fastapi-openai-compat``: this module exposes ``list_models``
and ``run_completion`` callables that the router factory wires into
``/v1/chat/completions`` and ``/v1/models``.

The heavy lifting (request envelope validation, SSE serialization, streaming,
reasoning content and tool call deltas) is handled by the library. Here we only
translate between OpenAI-style messages and the DeepSeek web API, map the
requested model name to the DeepSeek thinking/search switches, and recover from
stale sessions by resetting and retrying once.
"""

from __future__ import annotations

import time
from collections.abc import Generator
from pathlib import Path
from typing import Any, Optional

from fastapi_openai_compat import ChatCompletion, Choice, Message

from common.api import DeepSeekAPI
from common.config import DEEPSEEK_API_KEY

from .dsml_parser import build_tool_params, parse_tool_calls_from_text, remove_tool_tags
from .session_store import SessionStore

# Model name -> (thinking_enabled, search_enabled)
MODELS: dict[str, tuple[bool, bool]] = {
    "thinking_not_search": (True, False),
    "thinking_search": (True, True),
    "not_thinking_not_search": (False, False),
    "not_thinking_search": (False, True),
}

DEFAULT_MODEL = "thinking_not_search"

# A stale/expired session or an empty provider response means we should drop
# the session and try once more with a brand new one.
_SESSION_ERROR_HINTS = (
    "session",
    "not found",
    "invalid",
    "expired",
    "does not exist",
    "empty response",
    "unparsable",
    "provider returned",
)

_MAX_ATTEMPTS = 2

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_session_store = SessionStore(
    data_file=_PROJECT_ROOT / ".session_data.json",
    legacy_file=_PROJECT_ROOT / ".session_id",
)
api = DeepSeekAPI(DEEPSEEK_API_KEY)


class EmptyResponseError(Exception):
    """The provider streamed no content and no response id."""


def _is_session_error(exc: Exception) -> bool:
    message = str(exc).lower()
    return isinstance(exc, EmptyResponseError) or any(
        hint in message for hint in _SESSION_ERROR_HINTS
    )


def _extract_content(content: Any) -> str:
    """Flatten OpenAI message content (str or list of parts) into text."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for part in content:
            if isinstance(part, dict):
                parts.append(part.get("text") or "")
            else:
                parts.append(getattr(part, "text", "") or "")
        return "\n".join(p for p in parts if p)
    return str(content) if content is not None else ""


def _tools_instruction(tools: Optional[list[dict]]) -> str:
    """Build a format guide telling the model how to emit tool calls.

    DeepSeek's web chat has no native tool-calling, so we describe the exact
    XML block the model must produce for each available tool, including the
    canonical parameter names the harness validates against.

    The worked example uses a REAL tool name and REAL parameter tags from the
    current request. It deliberately avoids literal ``tool_name`` /
    ``param_name`` placeholders: some models copy those verbatim and emit e.g.
    ``<tool_name>web_fetch</tool_name>`` instead of ``<web_fetch>``, which the
    parser cannot match.
    """
    params_by_tool = build_tool_params(tools)
    if not params_by_tool:
        return ""

    # Choose a representative tool that actually has parameters, so the model
    # sees concrete parameter tags in the example.
    example_name: Optional[str] = None
    for name, params in params_by_tool.items():
        if params:
            example_name = name
            break
    if example_name is None:
        example_name = next(iter(params_by_tool))

    example_lines = [f"<{example_name}>"]
    for param in params_by_tool[example_name]:
        example_lines.append(f"<{param}>value</{param}>")
    example_lines.append(f"</{example_name}>")
    example_block = "\n".join(example_lines)

    lines = [
        "To call a tool, emit ONLY one XML block, using the tool's REAL name as the tag:",
        "",
        example_block,
        "",
        f'- The tag is the tool name (for example <{example_name}>), never the literal word "tool_name".',
        "- One tag per parameter, named exactly as listed below.",
        "- Emit one tool call per block; no prose and no markdown fences around the block.",
        "- Do NOT wrap values in JSON unless the value itself is a list/object.",
        "",
        "Available tools and their exact parameter names:",
    ]
    for name, params in params_by_tool.items():
        rendered = ", ".join(f"<{p}>...</{p}>" for p in params) if params else "(no parameters)"
        lines.append(f"- {name}: {rendered}")

    return "\n".join(lines)


def _messages_to_prompt(messages: list[dict], tools: Optional[list[dict]] = None) -> str:
    """Render OpenAI-style messages into a single DeepSeek prompt string."""
    parts = []
    instruction = _tools_instruction(tools)
    if instruction:
        parts.append(f"[TOOLS]\n{instruction}")
    for msg in messages:
        role = msg.get("role", "user")
        content = _extract_content(msg.get("content"))
        parts.append(f"[{role.upper()}]\n{content}")
    return "\n\n".join(parts)


def _resolve_switches(model: str) -> tuple[bool, bool]:
    return MODELS.get(model, MODELS[DEFAULT_MODEL])


def _chunk(
    model: str,
    chat_id: str,
    delta: Message,
    finish_reason: Optional[str] = None,
) -> ChatCompletion:
    return ChatCompletion(
        id=f"chatcmpl-{chat_id}",
        object="chat.completion.chunk",
        created=int(time.time()),
        model=model,
        choices=[Choice(index=0, delta=delta, finish_reason=finish_reason)],
    )


def _session() -> SessionStore:
    """Lazily load the persisted session on first use."""
    if not _session_store.sessions:
        _session_store.load()
    return _session_store


def _consume(prompt: str, thinking: bool, search: bool, chat_id: str, parent_message_id: Optional[str]):
    return api.chat_completion(
        chat_id,
        prompt,
        parent_message_id=parent_message_id,
        thinking_enabled=thinking,
        search_enabled=search,
    )


def list_models() -> list[str]:
    """Return the available model identifiers."""
    return list(MODELS.keys())


def _collect(prompt: str, thinking: bool, search: bool, chat_id: str, parent_message_id: Optional[str]):
    """Consume a full (non-streamed) completion.

    Returns ``(full_text, full_thinking, response_message_id)`` and raises
    :class:`EmptyResponseError` when the provider produced nothing at all.
    """
    full_text = ""
    full_thinking = ""
    response_id = None

    for piece in _consume(prompt, thinking, search, chat_id, parent_message_id):
        kind = piece.get("type")
        if kind == "content":
            full_text += piece.get("delta", "")
        elif kind == "thinking":
            full_thinking += piece.get("delta", "")
        elif kind == "finished":
            response_id = piece.get("response_message_id")
            break

    if response_id is None and not full_text and not full_thinking:
        raise EmptyResponseError("Empty response from DeepSeek API")

    return full_text, full_thinking, response_id


def _tool_call_chunks(model: str, chat_id: str, tool_calls: list[dict]) -> list[ChatCompletion]:
    """Build OpenAI-conformant streaming tool-call deltas.

    Each tool call is announced with a stable ``index`` plus id/name, then its
    JSON arguments are appended in a second delta with the SAME ``index``. The
    ``index`` is required by the OpenAI streaming contract; without it a
    client's tool-call accumulator produces ``undefined`` entries and crashes.
    """
    chunks: list[ChatCompletion] = []

    announce = []
    for i, call in enumerate(tool_calls):
        fn = call.get("function", {})
        announce.append(
            {
                "index": i,
                "id": call.get("id", f"call_{i}"),
                "type": "function",
                "function": {"name": fn.get("name", ""), "arguments": ""},
            }
        )
    if announce:
        chunks.append(_chunk(model, chat_id, Message(role="assistant", tool_calls=announce)))

    for i, call in enumerate(tool_calls):
        fn = call.get("function", {})
        arg_delta = [{"index": i, "function": {"arguments": fn.get("arguments", "")}}]
        chunks.append(_chunk(model, chat_id, Message(role="assistant", tool_calls=arg_delta)))

    return chunks


class _ToolTagSuppressor:
    """Keep textual tool-call XML out of streamed assistant content.

    DeepSeek emits tool calls as XML text, and the proxy also converts them into
    native OpenAI ``tool_calls``. Emitting the raw XML in ``content`` makes the
    client see the same call twice (once as text, once as a tool call), which
    strict OpenAI clients reject. This holds back a short tail so a tag split
    across deltas is never emitted, and stops emitting once a known tool opening
    tag (or a DSML marker) appears.
    """

    def __init__(self, tool_names: list[str], holdback: int = 64) -> None:
        self.tags = tuple(f"<{name}" for name in tool_names) + ("<\uff5cDSML\uff5c",)
        self.holdback = holdback
        self.buf = ""
        self.suppressed = False

    def _first_tag(self) -> int:
        best = -1
        for tag in self.tags:
            idx = self.buf.find(tag)
            if idx != -1 and (best == -1 or idx < best):
                best = idx
        return best

    def feed(self, delta: str) -> str:
        if self.suppressed:
            return ""
        self.buf += delta
        idx = self._first_tag()
        if idx != -1:
            visible = self.buf[:idx]
            self.buf = ""
            self.suppressed = True
            return visible
        if len(self.buf) > self.holdback:
            visible = self.buf[: -self.holdback]
            self.buf = self.buf[-self.holdback:]
            return visible
        return ""

    def flush(self) -> str:
        if self.suppressed:
            return ""
        visible = self.buf
        self.buf = ""
        return visible


def _stream(
    model: str,
    thinking: bool,
    search: bool,
    prompt: str,
    body: dict,
) -> Generator[ChatCompletion, None, None]:
    store = _session()
    tools = body.get("tools")
    # Tool names come from the request schema; without a schema the opening tags
    # are unknown, so XML suppression is skipped and content passes through.
    tool_names = list(build_tool_params(tools).keys()) if tools else []

    full_text = ""
    last_response_message_id = None
    has_content = False
    chat_id = ""

    for attempt in range(_MAX_ATTEMPTS):
        chat_id = store.resolve(api, body.get("session_id"))
        parent_message_id = store.parent_message_id(chat_id)

        full_text = ""
        last_response_message_id = None
        has_content = False
        suppressor = _ToolTagSuppressor(tool_names) if tool_names else None

        try:
            for piece in _consume(prompt, thinking, search, chat_id, parent_message_id):
                kind = piece.get("type")
                if kind == "thinking":
                    delta = piece.get("delta", "")
                    if delta:
                        has_content = True
                        yield _chunk(model, chat_id, Message(role="assistant", reasoning_content=delta))
                elif kind == "content":
                    delta = piece.get("delta", "")
                    if delta:
                        has_content = True
                        full_text += delta
                        if suppressor is None:
                            yield _chunk(model, chat_id, Message(role="assistant", content=delta))
                        else:
                            visible = suppressor.feed(delta)
                            if visible:
                                yield _chunk(model, chat_id, Message(role="assistant", content=visible))
                elif kind == "finished":
                    last_response_message_id = piece.get("response_message_id")
                    break
        except Exception as exc:  # noqa: BLE001 - classify provider failures
            # Only safe to retry when nothing has been emitted to the client yet.
            if has_content or attempt + 1 >= _MAX_ATTEMPTS or not _is_session_error(exc):
                raise
            store.reset(api)
            continue

        if suppressor is not None:
            tail = suppressor.flush()
            if tail:
                yield _chunk(model, chat_id, Message(role="assistant", content=tail))

        if has_content or last_response_message_id is not None:
            break

        # Nothing at all came back: drop the session and retry once.
        store.reset(api)

    if not has_content and last_response_message_id is None:
        raise EmptyResponseError("Empty response from DeepSeek API")

    if last_response_message_id:
        store.update(chat_id, last_response_message_id)

    # Tool calls detected in the aggregated text become indexed stream deltas.
    tool_calls = parse_tool_calls_from_text(full_text, tools)
    if tool_calls:
        for chunk in _tool_call_chunks(model, chat_id, tool_calls):
            yield chunk
        yield _chunk(model, chat_id, Message(role="assistant"), finish_reason="tool_calls")
    else:
        yield _chunk(model, chat_id, Message(role="assistant"), finish_reason="stop")


def run_completion(model: str, messages: list[dict], body: dict) -> Any:
    """Run a chat completion for the given model and conversation.

    Returns a generator of ``ChatCompletion`` chunks when ``stream`` is set,
    otherwise a single ``ChatCompletion`` object. Stale sessions are reset and
    the request retried once.
    """
    thinking, search = _resolve_switches(model)
    prompt = _messages_to_prompt(messages, body.get("tools"))

    if body.get("stream"):
        return _stream(model, thinking, search, prompt, body)

    store = _session()
    tools = body.get("tools")

    full_text = ""
    full_thinking = ""
    last_response_message_id = None
    chat_id = ""

    for attempt in range(_MAX_ATTEMPTS):
        chat_id = store.resolve(api, body.get("session_id"))
        parent_message_id = store.parent_message_id(chat_id)
        try:
            full_text, full_thinking, last_response_message_id = _collect(
                prompt, thinking, search, chat_id, parent_message_id
            )
        except Exception as exc:  # noqa: BLE001 - classify provider failures
            if attempt + 1 < _MAX_ATTEMPTS and _is_session_error(exc):
                store.reset(api)
                continue
            raise
        break

    if last_response_message_id:
        store.update(chat_id, last_response_message_id)

    tool_calls = parse_tool_calls_from_text(full_text, tools)
    clean_text = remove_tool_tags(full_text, tools) if tool_calls else full_text

    message = Message(
        role="assistant",
        content=clean_text or None,
        reasoning_content=full_thinking or None,
    )
    if tool_calls:
        message.tool_calls = tool_calls

    return ChatCompletion(
        id=f"chatcmpl-{chat_id}",
        object="chat.completion",
        created=int(time.time()),
        model=model,
        choices=[
            Choice(
                index=0,
                message=message,
                finish_reason="tool_calls" if tool_calls else "stop",
            )
        ],
        usage={"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
        session_id=chat_id,
        response_message_id=last_response_message_id,
    )