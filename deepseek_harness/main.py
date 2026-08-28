from fastapi import Request as FastAPIRequest
from fastapi.middleware.cors import CORSMiddleware
from fastapi_offline import FastAPIOffline
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field, model_validator
from typing import List, Optional, Union, Dict, Any
import re
import time, json, os, uuid
from datetime import datetime
from pathlib import Path
from common.api import DeepSeekAPI
from common.config import DEEPSEEK_API_KEY

app = FastAPIOffline()

# Add CORS middleware
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

def load_session_from_file():
    """بارگذاری session از فایل JSON در هنگام استارت سرور"""
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
                    # همچنین فایل .session_id را هم برای سازگاری با ابزار send_with_session.py به‌روز کن
                    with open(SESSION_FILE, 'w') as sf:
                        sf.write(session_id)
                    return session_id
        except (json.JSONDecodeError, KeyError):
            pass
    return None

def save_session_to_file(session_id, last_message_id=None):
    """ذخیره session_id و last_message_id در فایل JSON"""
    data = {
        "session_id": session_id,
        "last_message_id": last_message_id
    }
    with open(SESSION_DATA_FILE, 'w') as f:
        json.dump(data, f)
    # همچنین فایل .session_id را هم برای سازگاری به‌روز کن
    with open(SESSION_FILE, 'w') as sf:
        sf.write(session_id)

def reset_session():
    """ایجاد session جدید و بازنشانی فایل"""
    global sessions
    # حذف session قبلی
    sessions.clear()
    # ایجاد session جدید
    new_session_id = api.create_chat_session()
    sessions[new_session_id] = {"created": time.time(), "last_message_id": None}
    save_session_to_file(new_session_id, None)
    print(f"🔄 Session reset: {new_session_id}")
    return new_session_id

def is_session_error(exception: Exception) -> bool:
    """
    تشخیص اینکه آیا exception مربوط به invalid session یا خطای provider است که نیاز به reset session دارد.
    در صورت نیاز می‌توانید این تابع را بر اساس نوع خطای خاص API سفارشی کنید.
    """
    error_msg = str(exception).lower()
    # کلمات کلیدی مرتبط با خطای session یا خطای provider که نیاز به reset دارد
    keywords = [
        "session", "not found", "invalid", "expired", "does not exist",
        "invalid api response", "empty response", "unparsable response",
        "provider returned", "provider-side", "empty", "unparsable"
    ]
    return any(keyword in error_msg for keyword in keywords)
# بارگذاری session از فایل در هنگام استارت سرور
loaded_session_id = load_session_from_file()
if loaded_session_id:
    print(f"✅ Loaded session from file: {loaded_session_id}")
else:
    print("ℹ️  No existing session found. A new session will be created on first request.")

AVAILABLE_MODELS = [{"id": "thinking_not_search", "object": "model","created": 1677610602, "owned_by": "you"},
                    {"id": "thinking_search", "object": "model","created": 1677610602, "owned_by": "you"},
                    {"id": "not_thinking_not_search", "object": "model","created": 1677610602, "owned_by": "you"},
                    {"id": "not_thinking_search", "object": "model","created": 1677610602, "owned_by": "you"}]

# ---------- Models ----------
class ContentPart(BaseModel):
    model_config = {"extra": "allow"}

    type: str = "text"
    text: Optional[str] = ""
    content: Optional[Union[str, List[Dict[str, Any]]]] = None
    tool_use_id: Optional[str] = None
    name: Optional[str] = None
    input: Optional[Dict[str, Any]] = None

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
    tools: Optional[List[Dict[str, Any]]] = None
    tool_choice: Optional[Union[str, Dict[str, Any]]] = None

class MessagesRequest(BaseModel):
    model_config = {"extra": "ignore"}

    model: str = "thinking_not_search"
    messages: List[Message]
    system: Optional[Union[str, List[ContentPart]]] = None
    tools: Optional[List[Dict[str, Any]]] = None
    tool_choice: Optional[Union[str, Dict[str, Any]]] = None
    max_tokens: int = 16000
    stream: bool = False
    temperature: Optional[float] = None
    session_id: Optional[str] = None
    
# ---------- Helper ----------
def extract_content(content: Union[str, List[ContentPart]]) -> str:
    if isinstance(content, str):
        return content
    parts = []
    for part in content:
        if part.type == "text":
            parts.append(part.text or "")
        elif part.type == "tool_result":
            result = part.content if part.content is not None else part.text or ""
            parts.append(f"[TOOL_RESULT {part.tool_use_id}]\n{result}")
        elif part.type == "tool_use":
            parts.append(
                f"[TOOL_CALL {part.name}]\n{json.dumps(part.input or {}, ensure_ascii=False)}"
            )
    return "\n".join(parts)

def messages_to_api_format(
    messages: List[Message],
    include_history: bool = True,
    tools: Optional[List[Dict[str, Any]]] = None,
    tool_choice: Optional[Union[str, Dict[str, Any]]] = None,
) -> str:
    """تبدیل پیام‌های OpenAI به prompt مورد انتظار DeepSeek."""
    if not include_history:
        messages = [message for message in reversed(messages) if message.role == "user"][:1]
        messages.reverse()

    parts = []
    for msg in messages:
        content = extract_content(msg.content)
        parts.append(f"[{msg.role.upper()}]\n{content}")

    if tools:
        tool_lines = [
            "You can call tools when needed. Reply with exactly one JSON object inside this tag:",
            "<tool_call>{\"name\":\"tool_name\",\"input\":{}}</tool_call>",
            "Available tools:",
        ]
        for tool in tools:
            tool_lines.append(json.dumps(tool, ensure_ascii=False))
        if tool_choice:
            tool_lines.append(f"Requested tool choice: {json.dumps(tool_choice, ensure_ascii=False)}")
        parts.insert(0, "\n".join(tool_lines))
    return "\n\n".join(parts)

def messages_api_to_chat_request(request: MessagesRequest) -> ChatRequest:
    messages = list(request.messages)
    if request.system:
        messages.insert(0, Message(role="system", content=request.system))

    return ChatRequest(
        model=request.model,
        messages=messages,
        stream=request.stream,
        temperature=request.temperature,
        max_tokens=request.max_tokens,
        session_id=request.session_id,
        tools=request.tools,
        tool_choice=request.tool_choice,
    )

def parse_tool_call(
    text: str,
    tools: Optional[List[Dict[str, Any]]] = None,
) -> Optional[Dict[str, Any]]:
    marker_patterns = (
        r"<tool_call>\s*",
        r"\[TOOL_CALL\s+([A-Za-z_][\w.-]*)\]\s*",
        r"```(?:json)?\s*",
    )
    marker = None
    marker_tool_name = None
    for pattern in marker_patterns:
        match = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
        if match and (marker is None or match.start() < marker.start()):
            marker = match
            marker_tool_name = match.group(1) if match.lastindex else None

    payload = text[marker.end():] if marker else text.lstrip()
    object_start = payload.find("{")
    if object_start < 0:
        return None
    try:
        call, end = json.JSONDecoder().raw_decode(payload[object_start:])
    except json.JSONDecodeError:
        return None
    if marker and marker.group(0).lstrip().lower().startswith("<tool_call"):
        if not re.match(r"\s*</tool_call>", payload[object_start + end:], re.DOTALL):
            return None
    if not isinstance(call, dict):
        return None
    if marker_tool_name and "name" not in call:
        call = {"name": marker_tool_name, "input": call}
    if "function" in call and isinstance(call["function"], dict):
        function = call["function"]
        call = {"name": function.get("name"), "input": function.get("arguments", function.get("input", {}))}
        if isinstance(call["input"], str):
            try:
                call["input"] = json.loads(call["input"])
            except json.JSONDecodeError:
                return None
    if not isinstance(call.get("name"), str) or not isinstance(call.get("input", {}), dict):
        return None
    if tools and call["name"] not in {
        tool.get("name") or tool.get("function", {}).get("name")
        for tool in tools
    }:
        return None
    return {
        "id": f"toolu_{uuid.uuid4().hex}",
        "name": call["name"],
        "input": call.get("input", {}),
    }

def chat_response_to_messages_response(response: dict) -> dict:
    choice = response.get("choices", [{}])[0]
    message = choice.get("message", {})
    content = message.get("content") or ""
    tool_calls = message.get("tool_calls") or []
    blocks = [{"type": "text", "text": content}] if content else []
    for tool_call in tool_calls:
        function = tool_call.get("function", {})
        try:
            tool_input = json.loads(function.get("arguments", "{}"))
        except json.JSONDecodeError:
            tool_input = {}
        blocks.append({
            "type": "tool_use",
            "id": tool_call.get("id", f"toolu_{uuid.uuid4().hex}"),
            "name": function.get("name", ""),
            "input": tool_input,
        })
    return {
        "id": response.get("id", f"msg_{uuid.uuid4().hex}"),
        "type": "message",
        "role": "assistant",
        "model": response.get("model"),
        "content": blocks,
        "stop_reason": "tool_use" if tool_calls else "end_turn",
        "stop_sequence": None,
        "usage": {
            "input_tokens": response.get("usage", {}).get("prompt_tokens", 0),
            "output_tokens": response.get("usage", {}).get("completion_tokens", 0),
        },
        "session_id": response.get("session_id"),
        "response_message_id": response.get("response_message_id"),
    }
# ---------- Middleware برای لاگ ----------
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
    # Manage chat session
    chat_id = None
    max_retries = 1  # حداکثر یک بار تلاش مجدد
    retry_count = 0
    
    while retry_count <= max_retries:
        try:
            # 1. اگر کلاینت session_id ارسال کرده باشد، اولویت با آن است
            if request.session_id and request.session_id in sessions:
                chat_id = request.session_id
            else:
                # 2. اگر session_id در حافظه وجود دارد (بارگذاری شده از فایل)، از آن استفاده کن
                if sessions:
                    # از اولین session موجود استفاده کن (معمولاً فقط یکی است)
                    chat_id = next(iter(sessions.keys()))
                else:
                    # 3. ایجاد session جدید
                    chat_id = api.create_chat_session()
                    sessions[chat_id] = {"created": time.time(), "last_message_id": None}
                    # ذخیره در فایل
                    save_session_to_file(chat_id, None)
            
            parent_message_id = sessions.get(chat_id, {}).get("last_message_id")
            
            prompt = messages_to_api_format(
                request.messages,
                include_history=parent_message_id is None,
                tools=request.tools,
                tool_choice=request.tool_choice,
            )

            if request.model=="not_thinking_not_search":
                    thinking=False
                    search=False
            elif request.model=="thinking_not_search":
                    thinking=True
                    search=False 
            elif request.model=="thinking_search":
                    thinking=True
                    search=True
            elif request.model=="not_thinking_search":
                    thinking=False
                    search=True
            else:
                thinking, search = True, False  # default fallback
                
            if request.stream:
                def generate():
                    last_response_message_id = None
                    has_content = False  # برای تشخیص پاسخ خالی
                    try:
                        # ارسال به API با کل history
                        for chunk in api.chat_completion(
                            chat_id, 
                            prompt,  # کل messages به صورت prompt
                            parent_message_id=parent_message_id,
                            thinking_enabled=thinking,
                            search_enabled=search
                        ):
                            chunk_type = chunk.get("type")
                            
                            if chunk_type == 'thinking':
                                # ارسال thinking به عنوان reasoning_content (طبق استاندارد OpenAI)
                                delta = {"reasoning_content": chunk.get("delta", "")}
                                has_content = True
                            elif chunk_type == 'content':
                                delta = {"content": chunk.get("delta", "")}
                                has_content = True
                            elif chunk_type == 'finished':
                                last_response_message_id = chunk.get("response_message_id")
                                break  # خروج از حلقه برای ارسال final chunk
                            else:
                                continue  # نوع ناشناخته، نادیده بگیر
                            
                            response_chunk = {
                                "id": f"chatcmpl-{chat_id}",
                                "object": "chat.completion.chunk",
                                "created": int(time.time()),
                                "model": request.model,
                                "choices": [
                                    {
                                        "index": 0,
                                        "delta": delta,
                                        "finish_reason": None
                                    }
                                ]
                            }
                            yield f"data: {json.dumps(response_chunk)}\n\n"

                        # اگر هیچ محتوایی دریافت نشد و session نیز به‌روز نشد، خطا پرتاب کن
                        if not has_content and last_response_message_id is None:
                            raise Exception("Empty response from DeepSeek API")

                        # Update session with last message id
                        if last_response_message_id:
                            sessions[chat_id]["last_message_id"] = last_response_message_id
                            # ذخیره در فایل
                            save_session_to_file(chat_id, last_response_message_id)
                        
                        final_chunk = {
                            "id": f"chatcmpl-{chat_id}",
                            "object": "chat.completion.chunk",
                            "created": int(time.time()),
                            "model": request.model,
                            "choices": [
                                {
                                    "index": 0,
                                    "delta": {},
                                    "finish_reason": "stop"
                                }
                            ],
                            "session_id": chat_id,
                            "response_message_id": last_response_message_id
                        }
                        yield f"data: {json.dumps(final_chunk)}\n\n"
                        yield "data: [DONE]\n\n"
                    except Exception as e:
                        # اگر خطای session رخ داد، session را بازنشانی کن
                        if is_session_error(e):
                            reset_session()
                            # ارسال خطا به کلاینت
                            error_chunk = {
                                "error": {
                                    "message": f"Session expired. Please retry with new session.",
                                    "type": "session_error",
                                    "session_reset": True
                                }
                            }
                            yield f"data: {json.dumps(error_chunk)}\n\n"
                            yield "data: [DONE]\n\n"
                        else:
                            # خطای دیگر را propagate کن
                            raise e

                return StreamingResponse(generate(), media_type="text/event-stream")

            # حالت غیر-استریم
            full_text = ""
            full_thinking = ""
            last_response_message_id = None
            
            for chunk in api.chat_completion(
                chat_id, 
                prompt,  # کل messages
                parent_message_id=parent_message_id,
                thinking_enabled=thinking,
                search_enabled=search
            ):
                chunk_type = chunk.get("type")
                    
                if chunk_type == 'content':
                    full_text += chunk.get("delta", "")
                elif chunk_type == 'thinking':
                    full_thinking += chunk.get("delta", "")
                elif chunk_type == 'finished':
                    last_response_message_id = chunk.get("response_message_id")
                    break

            # اگر پاسخ خالی بود، خطا پرتاب کن تا retry فعال شود
            if last_response_message_id is None and not full_text and not full_thinking:
                raise Exception("Empty response from DeepSeek API")

            # Update session with last message id
            if last_response_message_id:
                sessions[chat_id]["last_message_id"] = last_response_message_id
                # ذخیره در فایل
                save_session_to_file(chat_id, last_response_message_id)

            tool_call = parse_tool_call(full_text, request.tools) if request.tools else None
            response_message = {
                "role": "assistant",
                "content": None if tool_call else full_text,
                "reasoning_content": full_thinking if full_thinking else None,
            }
            response_choice = {
                "index": 0,
                "message": response_message,
                "finish_reason": "tool_calls" if tool_call else "stop",
            }
            if tool_call:
                response_message["tool_calls"] = [{
                    "id": tool_call["id"],
                    "type": "function",
                    "function": {
                        "name": tool_call["name"],
                        "arguments": json.dumps(tool_call["input"], ensure_ascii=False),
                    },
                }]

            return {
                "id": f"chatcmpl-{chat_id}",
                "object": "chat.completion",
                "created": int(time.time()),
                "model": request.model,
                "choices": [response_choice],
                "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
                "session_id": chat_id,
                "response_message_id": last_response_message_id
            }
            
        except Exception as e:
            # اگر خطای session رخ داد و هنوز تلاش مجدد باقی مانده است
            if is_session_error(e) and retry_count < max_retries:
                retry_count += 1
                print(f"⚠️ Session error detected. Retrying with new session... (attempt {retry_count})")
                reset_session()
                continue
            else:
                # اگر خطا از نوع session نبود یا تلاش مجدد تمام شد، خطا را propagate کن
                raise e

@app.post("/v1/messages")
async def messages(request: MessagesRequest):
    chat_request = messages_api_to_chat_request(request)

    # Buffer the backend response so both text and tool_use use one reliable SSE path.
    response = await chat_completions(chat_request.model_copy(update={"stream": False}))
    messages_response = chat_response_to_messages_response(response)

    if not request.stream:
        return messages_response

    async def generate_messages_events():

        def event(event_type: str, data: dict) -> str:
            return f"event: {event_type}\ndata: {json.dumps(data)}\n\n"

        message = {
            key: messages_response[key]
            for key in ("id", "type", "role", "model")
        }
        message.update({
            "content": [],
            "stop_reason": None,
            "stop_sequence": None,
            "usage": {"input_tokens": 0, "output_tokens": 0},
        })
        yield event("message_start", {
            "type": "message_start",
            "message": message,
        })
        for index, block in enumerate(messages_response["content"]):
            block_type = block.get("type")
            if block_type == "text":
                stream_block = {"type": "text", "text": ""}
            elif block_type == "tool_use":
                stream_block = {"type": "tool_use", "id": block["id"], "name": block["name"], "input": {}}
            else:
                continue
            yield event("content_block_start", {"type": "content_block_start", "index": index, "content_block": stream_block})
            if block_type == "text" and block.get("text"):
                yield event("content_block_delta", {"type": "content_block_delta", "index": index, "delta": {"type": "text_delta", "text": block["text"]}})
            elif block_type == "tool_use":
                yield event("content_block_delta", {"type": "content_block_delta", "index": index, "delta": {"type": "input_json_delta", "partial_json": json.dumps(block.get("input", {}), ensure_ascii=False)}})
            yield event("content_block_stop", {
                "type": "content_block_stop",
                "index": index,
            })
        yield event("message_delta", {"type": "message_delta", "delta": {"stop_reason": messages_response["stop_reason"], "stop_sequence": None}, "usage": messages_response["usage"]})
        yield event("message_stop", {"type": "message_stop"})

    return StreamingResponse(
        generate_messages_events(),
        media_type="text/event-stream",
        headers={"cache-control": "no-cache", "connection": "keep-alive"},
    )

@app.get("/")
async def root():
    return {"status": "ok", "message": "DeepSeek API Proxy is running"}

@app.get("/health")
async def health():
    return {"status": "healthy"}

@app.get("/v1/models")
async def list_models():
    return {"object": "list", "data": AVAILABLE_MODELS}
