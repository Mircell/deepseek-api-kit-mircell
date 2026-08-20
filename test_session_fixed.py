import sys
import io
import requests
import json

# Fix encoding for Windows console
if sys.stdout.encoding != 'utf-8':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

BASE_URL = "http://127.0.0.1:8000"

# First request - get session_id
payload = {
    "model": "thinking_not_search",
    "messages": [{"role": "user", "content": "سلام، اسم من علی است"}],
    "stream": False
}
print("Sending first request...")
resp = requests.post(f"{BASE_URL}/v1/chat/completions", json=payload)
print("First response status:", resp.status_code)
data = resp.json()
print("First response:", json.dumps(data, indent=2, ensure_ascii=False))
session_id = data.get("session_id")
print(f"Session ID: {session_id}")

if not session_id:
    print("No session_id returned")
    sys.exit(1)

# Second request - use same session_id
payload2 = {
    "model": "thinking_not_search",
    "messages": [{"role": "user", "content": "اسم من را به خاطر داری؟"}],
    "stream": False,
    "session_id": session_id
}
print("\nSending second request with session_id:", session_id)
resp2 = requests.post(f"{BASE_URL}/v1/chat/completions", json=payload2)
print("Second response status:", resp2.status_code)
data2 = resp2.json()
print("Second response:", json.dumps(data2, indent=2, ensure_ascii=False))

# Check if the second response contains the remembered name
content = data2.get("choices", [{}])[0].get("message", {}).get("content", "")
print("\nAssistant response (second):", content)
if "علی" in content:
    print("✅ Session works: assistant remembered the name.")
else:
    print("⚠️ Session may not be working properly.")