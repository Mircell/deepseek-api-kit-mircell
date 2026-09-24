from fastapi import Request as FastAPIRequest
from fastapi.middleware.cors import CORSMiddleware
from fastapi_offline import FastAPIOffline
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field, model_validator
from typing import List, Optional, Union, Dict, Any
import time, json, os, uuid
from datetime import datetime
from pathlib import Path
from common.api import DeepSeekAPI
from common.config import DEEPSEEK_API_KEY

app = FastAPIOffline()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

api = DeepSeekAPI(DEEPSEEK_API_KEY)

sessions: Dict[str, dict] = {}
SESSION_FILE = Path(__file__).parent.parent / ".session_id"
SESSION_DATA_FILE = Path(__file__).parent.parent / ".session_data.json"

DEBUG = os.environ.get("DEBUG_PROXY", "0") == "1"


def _dbg(*args, **kwargs):
    if DEBUG:
        print("[PROXY-DEBUG]", *args, **kwargs)


def load_session_from_file():
    if SESSION_DATA_FILE.exists():
        try:
            with open(SESSION_DATA_FILE, 'r') as f:
                data = json.load(f)
                session_id = data.get("session_id")
                last_message_id = data.get("last_message_id")
                if session_id:
                    sessions[session_id] = {
                        "created": time.time(),
                        "last_message_id": last_message_id
                    }
                    with open(SESSION_FILE, 'w') as sf:
                        sf.write(session_id)
                    return session_id
        except (json.JSONDecodeError, KeyError):
            pass
    return None


def save_session_to_file(session_id, last_message_id=None):
    data = {"session_id": session_id, "last_message_id": last_message_id}
    with open(SESSION_DATA_FILE, 'w') as f:
        json.dump(data, f)
    with open(SESSION_FILE, 'w') as sf:
        sf.write(session_id)


def reset_session():
    global sessions
    sessions.clear()
    new_session_id = api.create_chat_session()
    sessions[new_session_id] = {"created": time.time(), "last_message_id": None}
    save_session_to_file(new_session_id, None)
    print(f"🔄 Session reset: {new_session_id}")
    return new_session_id


def is_session_error(exception: Exception) -> bool:
    error_msg = str(exception).lower()
    keywords = [
        "session", "not found", "invalid", "expired", "does not exist",
        "invalid api response", "empty response", "unparsable response",
        "provider returned", "provider-side", "empty", "unparsable", "Invalid API Response"
    ]
    return any(kw in error_msg for kw in keywords)


loaded_session_id = load_session_from_file()
if loaded_session_id:
    print(f"✅ Loaded session from file: {loaded_session_id}")
else:
    print("ℹ️  No existing session found. A new session will be created on first request.")


AVAILABLE_MODELS = [
    {"id": "thinking_not_search", "object": "model", "created": 1677610602, "owned_by": "you"},
    {"id": "thinking_search", "object": "model", "created": 1677610602, "owned_by": "you"},
    {"id": "not_thinking_not_search", "object": "model", "created": 1677610602, "owned_by": "you"},
    {"id": "not_thinking_search", "object": "model", "created": 1677610602, "owned_by": "you"},
]


# ---------- Models ----------
class ContentPart(BaseModel):
    type: str = "text"
    text: Optional[str] = ""


class Message(BaseModel):
    role: str = "user"
    content: Union[str, List[ContentPart]] = ""
    reasoning_content: Optional[str] = None

    @model_validator(mode="before")
    @classmethod
    def fill_defaults(cls, values):
        if values is None:
            return {"role": "user", "content": ""}
        if isinstance(values, dict):
            values.setdefault("role", "user")
            values.setdefault("content", "")
            values.setdefault("reasoning_content", "")
            return values
        return values


class ChatRequest(BaseModel):
    model_config = {"extra": "ignore"}
    messages: List[Message]
    model: str = "thinking_not_search"
    stream: Optional[bool] = False
    stream_options: Optional[Dict[str, Any]] = None
    temperature: Optional[float] = None
    max_tokens: Optional[int] = None
    session_id: Optional[str] = None


def extract_content(content: Union[str, List[ContentPart]]) -> str:
    if isinstance(content, str):
        return content
    return "\n".join(part.text or "" for part in content if part.type == "text")


def messages_to_api_format(messages: List[Message]) -> str:
    parts = []
    for msg in messages:
        content = extract_content(msg.content)
        parts.append(f"[{msg.role.upper()}]\n{content}")
    return "\n\n".join(parts)


@app.middleware("http")
async def log_time(request: FastAPIRequest, call_next):
    start = datetime.now()
    print(f"[{start.strftime('%H:%M:%S.%f')[:-3]}] --> {request.method} {request.url.path}")
    response = await call_next(request)
    end = datetime.now()
    print(f"[{end.strftime('%H:%M:%S.%f')[:-3]}] <-- {response.status_code} (took {(end-start).total_seconds():.2f}s)")
    return response


# ---------- Endpoints ----------
@app.post("/v1/chat/completions")
async def chat_completions(request: ChatRequest):
    chat_id = None
    max_retries = 1
    retry_count = 0

    while retry_count <= max_retries:
        try:
            if request.session_id and request.session_id in sessions:
                chat_id = request.session_id
            else:
                if sessions:
                    chat_id = next(iter(sessions.keys()))
                else:
                    chat_id = api.create_chat_session()
                    sessions[chat_id] = {"created": time.time(), "last_message_id": None}
                    save_session_to_file(chat_id, None)

            parent_message_id = sessions.get(chat_id, {}).get("last_message_id")
            prompt = messages_to_api_format(request.messages)

            if request.model == "not_thinking_not_search":
                thinking, search = False, False
            elif request.model == "thinking_not_search":
                thinking, search = True, False
            elif request.model == "thinking_search":
                thinking, search = True, True
            elif request.model == "not_thinking_search":
                thinking, search = False, True
            else:
                thinking, search = True, False

            _dbg(f"chat_id={chat_id} parent={parent_message_id} thinking={thinking} search={search}")

            # ---------------- Streaming ----------------
            if request.stream:
                def generate():
                    last_response_message_id = None
                    has_content = False
                    final_content_fallback = ""
                    final_thinking_fallback = ""

                    try:
                        for chunk in api.chat_completion(
                            chat_id, prompt,
                            parent_message_id=parent_message_id,
                            thinking_enabled=thinking,
                            search_enabled=search
                        ):
                            chunk_type = chunk.get("type")
                            _dbg("chunk type:", chunk_type)

                            if chunk_type == 'thinking':
                                delta_text = chunk.get("delta", "") or ""
                                if delta_text:
                                    has_content = True
                                    rc = {
                                        "id": f"chatcmpl-{chat_id}",
                                        "object": "chat.completion.chunk",
                                        "created": int(time.time()),
                                        "model": request.model,
                                        "choices": [{
                                            "index": 0,
                                            "delta": {"reasoning_content": delta_text},
                                            "finish_reason": None
                                        }]
                                    }
                                    yield f"data: {json.dumps(rc)}\n\n"

                            elif chunk_type == 'content':
                                delta_text = chunk.get("delta", "") or ""
                                if delta_text:
                                    has_content = True
                                    rc = {
                                        "id": f"chatcmpl-{chat_id}",
                                        "object": "chat.completion.chunk",
                                        "created": int(time.time()),
                                        "model": request.model,
                                        "choices": [{
                                            "index": 0,
                                            "delta": {"content": delta_text},
                                            "finish_reason": None
                                        }]
                                    }
                                    yield f"data: {json.dumps(rc)}\n\n"

                            elif chunk_type == 'finished':
                                last_response_message_id = chunk.get("response_message_id")
                                final_content_fallback = chunk.get("content", "") or ""
                                final_thinking_fallback = chunk.get("thinking_content", "") or ""

                                if not has_content:
                                    if final_thinking_fallback:
                                        rc = {
                                            "id": f"chatcmpl-{chat_id}",
                                            "object": "chat.completion.chunk",
                                            "created": int(time.time()),
                                            "model": request.model,
                                            "choices": [{
                                                "index": 0,
                                                "delta": {"reasoning_content": final_thinking_fallback},
                                                "finish_reason": None
                                            }]
                                        }
                                        yield f"data: {json.dumps(rc)}\n\n"
                                    if final_content_fallback:
                                        rc = {
                                            "id": f"chatcmpl-{chat_id}",
                                            "object": "chat.completion.chunk",
                                            "created": int(time.time()),
                                            "model": request.model,
                                            "choices": [{
                                                "index": 0,
                                                "delta": {"content": final_content_fallback},
                                                "finish_reason": None
                                            }]
                                        }
                                        yield f"data: {json.dumps(rc)}\n\n"
                                break

                        if not has_content and not final_content_fallback and not final_thinking_fallback:
                            raise Exception("Empty response from DeepSeek API")

                        if last_response_message_id:
                            sessions[chat_id]["last_message_id"] = last_response_message_id
                            save_session_to_file(chat_id, last_response_message_id)

                        final_chunk = {
                            "id": f"chatcmpl-{chat_id}",
                            "object": "chat.completion.chunk",
                            "created": int(time.time()),
                            "model": request.model,
                            "choices": [{
                                "index": 0,
                                "delta": {},
                                "finish_reason": "stop"
                            }],
                            "session_id": chat_id,
                            "response_message_id": last_response_message_id
                        }
                        yield f"data: {json.dumps(final_chunk)}\n\n"
                        yield "data: [DONE]\n\n"

                    except Exception as e:
                        if is_session_error(e):
                            reset_session()
                            error_chunk = {
                                "error": {
                                    "message": "Session expired. Please retry with new session.",
                                    "type": "session_error",
                                    "session_reset": True
                                }
                            }
                            yield f"data: {json.dumps(error_chunk)}\n\n"
                            yield "data: [DONE]\n\n"
                        else:
                            raise e

                return StreamingResponse(generate(), media_type="text/event-stream")

            # ---------------- Non-streaming ----------------
            full_text = ""
            full_thinking = ""
            last_response_message_id = None
            final_content_fallback = ""
            final_thinking_fallback = ""

            for chunk in api.chat_completion(
                chat_id, prompt,
                parent_message_id=parent_message_id,
                thinking_enabled=thinking,
                search_enabled=search
            ):
                chunk_type = chunk.get("type")
                _dbg("chunk type:", chunk_type)

                if chunk_type == 'content':
                    full_text += chunk.get("delta", "") or ""
                elif chunk_type == 'thinking':
                    full_thinking += chunk.get("delta", "") or ""
                elif chunk_type == 'finished':
                    last_response_message_id = chunk.get("response_message_id")
                    final_content_fallback = chunk.get("content", "") or ""
                    final_thinking_fallback = chunk.get("thinking_content", "") or ""
                    break

            if not full_text and final_content_fallback:
                full_text = final_content_fallback
            if not full_thinking and final_thinking_fallback:
                full_thinking = final_thinking_fallback

            if last_response_message_id is None and not full_text and not full_thinking:
                raise Exception("Empty response from DeepSeek API")

            if last_response_message_id:
                sessions[chat_id]["last_message_id"] = last_response_message_id
                save_session_to_file(chat_id, last_response_message_id)

            return {
                "id": f"chatcmpl-{chat_id}",
                "object": "chat.completion",
                "created": int(time.time()),
                "model": request.model,
                "choices": [{
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": full_text,
                        "reasoning_content": full_thinking if full_thinking else None
                    },
                    "finish_reason": "stop"
                }],
                "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
                "session_id": chat_id,
                "response_message_id": last_response_message_id
            }

        except Exception as e:
            if is_session_error(e) and retry_count < max_retries:
                retry_count += 1
                print(f"⚠️ Session error detected. Retrying... (attempt {retry_count})")
                reset_session()
                continue
            else:
                raise e


@app.get("/")
async def root():
    return {"status": "ok", "message": "DeepSeek API Proxy is running"}


@app.get("/health")
async def health():
    return {"status": "healthy"}


@app.get("/v1/models")
async def list_models():
    return {"object": "list", "data": AVAILABLE_MODELS}