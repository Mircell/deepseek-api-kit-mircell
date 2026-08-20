import sys
import io
import requests
import json
import time

# Fix encoding for Windows console
if sys.stdout.encoding != 'utf-8':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

BASE_URL = "http://127.0.0.1:8000"

def send_message(session_id, message, is_first=False):
    payload = {
        "model": "thinking_not_search",
        "messages": [{"role": "user", "content": message}],
        "stream": False
    }
    if not is_first:
        payload["session_id"] = session_id
    
    resp = requests.post(f"{BASE_URL}/v1/chat/completions", json=payload)
    data = resp.json()
    content = data.get("choices", [{}])[0].get("message", {}).get("content", "")
    session_id = data.get("session_id")
    response_message_id = data.get("response_message_id")
    return session_id, content, response_message_id

print("=" * 60)
print("شروع مکالمه با ۳ دور پیام")
print("=" * 60)

# Round 1
print("\n[Round 1] User: سلام، اسم من احمد است")
session_id, content, msg_id = send_message(None, "سلام، اسم من احمد است", is_first=True)
print(f"Assistant: {content[:100]}...")
print(f"Session ID: {session_id}\n")

time.sleep(1)

# Round 2
print("[Round 2] User: شغل من برنامه‌نویسی است")
session_id, content, msg_id = send_message(session_id, "شغل من برنامه‌نویسی است")
print(f"Assistant: {content[:100]}...")
print(f"Session ID: {session_id}\n")

time.sleep(1)

# Round 3
print("[Round 3] User: اسم و شغل من را به خاطر داری؟")
session_id, content, msg_id = send_message(session_id, "اسم و شغل من را به خاطر داری؟")
print(f"Assistant: {content[:200]}...")
print(f"Session ID: {session_id}\n")

# Check if both name and job are remembered
if "احمد" in content and "برنامه" in content:
    print("✅ Session works perfectly: assistant remembered both name and job.")
else:
    print("⚠️ Session may not be fully working.")
    print("   Content:", content)