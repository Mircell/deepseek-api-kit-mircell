import requests
import json

BASE_URL = "http://127.0.0.1:8000"

# First request - get session_id
payload = {
    "model": "thinking_not_search",
    "messages": [{"role": "user", "content": "سلام، اسم من علی است"}],
    "stream": False
}
resp = requests.post(f"{BASE_URL}/v1/chat/completions", json=payload)
print("First response status:", resp.status_code)
data = resp.json()
print("First response:", json.dumps(data, indent=2, ensure_ascii=False))
session_id = data.get("session_id")
print(f"Session ID: {session_id}")

if not session_id:
    print("No session_id returned")
    exit()

# Second request - use same session_id
payload2 = {
    "model": "thinking_not_search",
    "messages": [{"role": "user", "content": "اسم من را به خاطر داری؟"}],
    "stream": False,
    "session_id": session_id
}
resp2 = requests.post(f"{BASE_URL}/v1/chat/completions", json=payload2)
print("Second response status:", resp2.status_code)
data2 = resp2.json()
print("Second response:", json.dumps(data2, indent=2, ensure_ascii=False))