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
    """
    params_by_tool = build_tool_params(tools)
    if not params_by_tool:
        return ""

    lines = [
        "You can call tools. To call a tool, emit ONLY an XML block like:",
        "<tool_name>",
        "<param_name>value</param_name>",
        "...",
        "</tool_name>",
        "",
        "Do NOT wrap values in JSON unless the parameter itself is a list/object.",
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


def _stream(
    model: str,
    thinking: bool,
    search: bool,
    prompt: str,
    body: dict,
) -> Generator[ChatCompletion, None, None]:
    store = _session()
    tools = body.get("tools")

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
                        yield _chunk(model, chat_id, Message(role="assistant", content=delta))
                elif kind == "finished":
                    last_response_message_id = piece.get("response_message_id")
                    break
        except Exception as exc:  # noqa: BLE001 - classify provider failures
            # Only safe to retry when nothing has been emitted to the client yet.
            if has_content or attempt + 1 >= _MAX_ATTEMPTS or not _is_session_error(exc):
                raise
            store.reset(api)
            continue

        if has_content or last_response_message_id is not None:
            break

        # Nothing at all came back: drop the session and retry once.
        store.reset(api)

    if not has_content and last_response_message_id is None:
        raise EmptyResponseError("Empty response from DeepSeek API")

    if last_response_message_id:
        store.update(chat_id, last_response_message_id)

    # Tool calls detected in the aggregated text become a dedicated delta.
    tool_calls = parse_tool_calls_from_text(full_text, tools)
    if tool_calls:
        yield _chunk(
            model,
            chat_id,
            Message(role="assistant", tool_calls=tool_calls),
            finish_reason="tool_calls",
        )
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