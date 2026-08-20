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
    
# ---------- Helper ----------
def extract_content(content: Union[str, List[ContentPart]]) -> str:
    if isinstance(content, str):
        return content
    return "\n".join(part.text or "" for part in content if part.type == "text")

def messages_to_api_format(messages: List[Message]) -> str:
    """تبدیل messages به فرمت API (اگر API از آرایه messages پشتیبانی کنه)"""
    parts = []
    for msg in messages:
        content = extract_content(msg.content)
        parts.append(f"[{msg.role.upper()}]\n{content}")
    return "\n\n".join(parts)
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
            
            prompt = messages_to_api_format(request.messages)

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

            return {
                "id": f"chatcmpl-{chat_id}",
                "object": "chat.completion",
                "created": int(time.time()),
                "model": request.model,
                "choices": [
                    {
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": full_text,
                            "reasoning_content": full_thinking if full_thinking else None
                        },

                        "finish_reason": "stop"
                    }
                ],
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
@app.get("/")
async def root():
    return {"status": "ok", "message": "DeepSeek API Proxy is running"}

@app.get("/health")
async def health():
    return {"status": "healthy"}

@app.get("/v1/models")
async def list_models():
    return {"object": "list", "data": AVAILABLE_MODELS}
