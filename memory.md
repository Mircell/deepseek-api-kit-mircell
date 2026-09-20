# DeepSeek API Kit - Comprehensive Documentation

## Overview

**DeepSeek API Kit** is a collection of lightweight, OpenAI-compatible proxy servers that provide seamless access to DeepSeek's language models through a familiar REST API interface. It features automatic session management, persistent chat history, intelligent error recovery, and (for the harness backend) textual tool-calling with an OpenAI-conformant streaming contract.

The project is designed to be easily deployable and usable, whether you're integrating it into existing applications or using it as a standalone chat service.

The kit ships with **two independent OpenAI-compatible servers**:

- **`openai_proxy/`** – the original standalone proxy server.
- **`deepseek_harness/`** – a refactored provider built on top of the [`fastapi-openai-compat`](https://github.com/deepset-ai/fastapi-openai-compat) router library, purpose-built for the DeepSeek Harness (`dsh`).

Both back onto the same `common/api.py` DeepSeek web-chat client.

---

## Features

- **OpenAI-Compatible API** – Drop-in replacement for OpenAI's `/v1/chat/completions` endpoint.
- **Persistent Chat Sessions** – Maintain conversation history across requests with automatic session persistence.
- **Auto-Reset Mechanism** – Automatically recovers from session expiry or invalid responses by resetting the session and retrying once.
- **Streaming & Non-Streaming Modes** – Full support for both real-time streaming and standard JSON responses.
- **Textual Tool Calling** – Parses tool invocations emitted as XML/DSML text into standard OpenAI `tool_calls`, with a streamed delta contract (stable `index`) that clients can accumulate correctly.
- **Model Selection** – Choose from four model variants:
  - `thinking_not_search` – reasoning enabled, no internet search.
  - `thinking_search` – reasoning enabled with internet search.
  - `not_thinking_not_search` – no reasoning, no search.
  - `not_thinking_search` – no reasoning, with search.
- **Utility Scripts** – Provided `send_with_session.py` for easy testing and interaction with the proxy.
- **Easy Setup** – Minimal configuration via environment variables.

---

## Requirements

- Python 3.8+
- pip (Python package manager)
- (Optional) Virtual environment (recommended)

Key dependencies (see `requirements.txt`): `fastapi`, `fastapi-offline`, `fastapi-openai-compat`, `pydantic`, `requests`, `curl-cffi`, `uvicorn`.

---

## Installation

1. **Clone the repository** (or download the source):
   ```bash
   git clone https://github.com/Mircell/deepseek-api-kit-mircell.git
   cd deepseek-api-kit
   ```

2. **Create and activate a virtual environment** (optional but recommended):
   ```bash
   python -m venv .venv
   source .venv/bin/activate   # On Windows: .venv\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up environment variables**:
   - Copy `.env.example` to `.env`:
     ```bash
     cp .env.example .env
     ```
   - Edit `.env` and add your DeepSeek API key:
     ```
     DEEPSEEK_API_KEY=your_api_key_here
     ```

---

## Configuration

| Environment Variable | Description |
|----------------------|-------------|
| `DEEPSEEK_API_KEY`   | Your DeepSeek API key (required). |
| `SESSION_FILE`       | (Optional) Path to the session ID file (default: `.session_id`). |
| `SESSION_DATA_FILE`  | (Optional) Path to the session data JSON file (default: `.session_data.json`). |

All configuration can be set in the `.env` file or directly in the environment.

---

## Project Structure

```
deepseek-api-kit/
├── deepseek_harness/            # OpenAI-compatible provider built on fastapi-openai-compat
│   ├── __init__.py
│   ├── main.py                  # FastAPI app; builds the router via create_chat_completion_router
│   ├── provider.py              # list_models + run_completion on top of common.api.DeepSeekAPI
│   ├── session_store.py         # Session persistence (.session_data.json / .session_id)
│   ├── dsml_parser.py           # Parse XML/DSML tool calls into OpenAI tool_calls
│   └── adapter.py               # Backward-compatibility shim (DeepSeekAdapter)
├── openai_proxy/
│   ├── __init__.py
│   └── main.py                  # Original standalone FastAPI proxy server
├── common/
│   ├── __init__.py
│   ├── api.py                   # DeepSeek API client (SSE web-chat)
│   ├── bypass.py                # Cloudflare bypass utilities
│   ├── CloudflareBypasser.py    # Cloudflare challenge solver
│   ├── config.py                # Configuration loader
│   ├── cookies.json             # Cookie storage
│   ├── pow.py                   # Proof-of-work helpers
│   ├── run_and_get_cookies.py   # Cookie acquisition script
│   ├── server.py                # Server helpers
│   └── wasm/                    # WebAssembly modules
├── deepseek_chat/
│   ├── __init__.py
│   ├── main.py                  # DeepSeek chat interface
│   ├── panel.html               # Web chat panel (optional)
│   └── session_store.py         # Session storage utilities
├── vscode_chat/
│   ├── __init__.py
│   └── main.py                  # VS Code chat proxy
├── .env.example                 # Example environment file
├── .session_data.json           # Persistent session data (auto-generated)
├── .session_id                  # Session ID file (auto-generated)
├── deepseek_harness.bat         # Windows batch: run deepseek_harness on port 8002
├── deepseek-api.bat             # Windows batch: run the proxy server
├── vscode-chat.bat              # Windows batch: run the VS Code chat proxy
├── example.py                   # Example usage script
├── requirements.txt             # Python dependencies
├── send_with_session.py         # Utility to send requests with session persistence
├── test_*.py                    # Test scripts
└── memory.md                    # This documentation file
```

---

## The `deepseek_harness` Provider

`deepseek_harness` is a thin translation layer on top of
`fastapi-openai-compat`. The library generates the entire HTTP surface
(`/v1/chat/completions`, `/chat/completions`, `/v1/models`, `/models`,
request validation, SSE serialization, reasoning content, and tool-call
deltas); the harness package only wires DeepSeek in.

### `main.py`
Builds the app with `FastAPIOffline` and mounts a single router:

```python
router = create_chat_completion_router(
    list_models=list_models,
    run_completion=run_completion,
    owned_by="deepseek",
)
app.include_router(router)
```

It also keeps `/` and `/health` for liveness checks.

### `provider.py`
Exposes the two callables the router factory expects:

- `list_models()` → returns the four model identifiers.
- `run_completion(model, messages, body)` → returns either a generator of
  `ChatCompletion` chunks (when `body["stream"]` is set) or a single
  `ChatCompletion` object.

It maps the requested model name to DeepSeek's thinking/search switches,
renders OpenAI-style messages into a single DeepSeek prompt string, forwards
the tool schema into the prompt (`_tools_instruction`), and normalizes the
streamed response into OpenAI deltas.

- `thinking_not_search` → `(True, False)`
- `thinking_search` → `(True, True)`
- `not_thinking_not_search` → `(False, False)`
- `not_thinking_search` → `(False, True)`

### `session_store.py`
A small `SessionStore` class that mirrors the legacy single-session behavior:
it resolves or creates a DeepSeek chat session, tracks the last
`response_message_id`, and persists both to `.session_data.json` (plus the
legacy `.session_id` file used by `send_with_session.py`).

### `adapter.py`
A backward-compatibility shim so existing imports such as
`from deepseek_harness.adapter import DeepSeekAdapter` keep resolving.

---

## Tool Calling (Textual Parsing)

DeepSeek's **web chat has no native tool-calling**, so the model emits tool
invocations as XML-ish text. The harness converts those blocks into standard
OpenAI `tool_calls`.

### Input format the model produces

```xml
<write>
<path>snake-game.html</path>
<content><!DOCTYPE html> ... </content>
</write>

<web_search>
<queries>["a", "b"]</queries>
</web_search>
```

### `dsml_parser.py`
The parser understands **two dialects** and merges their calls in source
order. A reply that mixes them no longer loses the plain-XML calls.

- **Plain XML dialect** (what the prompt guide teaches):
  `<web_fetch><url>…</url></web_fetch>`.
- **Native DSML dialect** (what the model falls back to): the full-width-bar
  `<|DSML|invoke name="web_fetch">` / `<|DSML|parameter name="url">` form.
  The model emits this sloppily — a space after the bar (`<|DSML| invoke`),
  an optional/missing `string` attribute, a `<|DSML|calls>` wrapper, a stray
  extra closing tag, and for large values (a file's `content`) a **missing or
  embedded parameter closing tag**. DSML parameters are therefore **not**
  matched with balanced tags: each value simply runs from its own opening tag
  to the next parameter's opening tag (or the end of the invoke), and any
  trailing `</|DSML|parameter>` is stripped. A missing closing tag can no
  longer drop the whole call, which is what previously broke `write` while
  `web_fetch` worked.
- **Balanced tag matching** (`_find_balanced_blocks`) is used for the plain
  XML dialect only. It tracks nesting of the same tag name, so HTML `<` / `>`
  inside a `<content>` parameter does not break parsing.
- **Declared-parameter extraction** reads only the parameters the request's
  `tools` schema declares, using the tool names as the block tags.
- **Value coercion** turns `["a","b"]` into a JSON list, numbers into numbers,
  and everything else into strings.
- **Alias mapping** rewrites the names the model tends to invent back to the
  canonical schema names (`path` → `file_path`, `query`/`search` → `queries`,
  `link` → `url`, `cmd` → `command`, …). This applies to **both** dialects: a
  DSML `<|DSML|parameter name="link">` becomes `url` for `web_fetch`, so the
  harness does not reject it with `INVALID_ARGS`.
- **Schema enforcement** (`_canonicalize`, strict when the request declared a
  schema): tool names not in the schema are dropped, and undeclared parameter
  names are dropped, so a hallucinated tool/parameter never reaches the
  harness. When no schema is supplied the call is forwarded as-is.
- **Malformed emission rejection**: a native invocation that declares
  parameters but supplies none is not turned into a bogus
  `{"content": "…"}` call.
- **Malformed literal-form recovery** (`_parse_literal_tool_name_calls`): some
  models copy the literal `tool_name` placeholder from the prompt guide and
  emit e.g. `<tool_name>web_fetch</tool_name>` (with the real name as *text*)
  followed by sibling parameter tags and a stray `</tool_name>`. The parser
  recovers that shape into a normal call, and `remove_tool_tags` strips it from
  the visible message too.
- Public helpers:
  - `build_tool_params(tools)` → `{tool_name: [param, …]}` (with a fallback
    table for well-known tools).
  - `parse_tool_calls_from_text(text, tools)` → list of OpenAI `tool_calls`.
  - `remove_tool_tags(text, tools)` → the same text with the tool blocks
    stripped (used to clean the visible assistant message).

### Prompt-guide rules
`provider._tools_instruction` renders a worked example using a **real** tool
name and its **real** parameter tags from the current request, and states
explicitly that the tag is the tool's name — never the literal word
`"tool_name"`. This prevents the model from emitting the malformed
`<tool_name>NAME</tool_name>` form in the first place; the parser fallback
above recovers it if it still happens.

The guide also **forbids the native DSML dialect outright** ("Use this plain
XML form ONLY. Do NOT emit any invoke/parameter/markup dialect, and never wrap
the call in a `calls` element."). Without that explicit prohibition the model
regularly regresses to its native prior whenever it ignores the guide; naming
and banning it is what keeps the common path on plain XML. The DSML path in
the parser remains as a recovery net for the times the model ignores the ban
anyway.

### `provider.py` — streamed tool-call contract
For streaming, tool calls are emitted as **OpenAI-conformant deltas** via
`_tool_call_chunks`:

1. A delta that **announces** every tool call with a stable `index` plus
   `id` and `function.name` (empty `arguments`).
2. A delta **per tool call** that appends its JSON `function.arguments`,
   carrying the **same `index`**.
3. A final delta with `finish_reason="tool_calls"`.

The stable `index` is required by the OpenAI streaming contract: a client's
tool-call accumulator keys partial deltas by `index`, and without it the
accumulator produces `undefined` entries. The proxy always emits a stable
`index`, so tool-call accumulation on the client works correctly.

In non-streaming mode the parsed `tool_calls` are attached directly to the
`ChatCompletion` message with `finish_reason="tool_calls"`.

---

## Session Management

The proxy automatically handles chat sessions to maintain conversation history:

- **Session Creation** – When no `session_id` is provided, the server creates
  a new session via the DeepSeek API and stores the ID in memory and on disk
  (`.session_data.json` and `.session_id`).
- **Persistence** – Sessions are saved to files, so they survive server
  restarts.
- **History** – Each session tracks the last `response_message_id`, enabling
  the server to carry the conversation forward with each new request.
- **Session ID Usage** – Clients can provide a `session_id` in requests to
  resume a specific conversation.

---

## Auto-Reset Mechanism (Error Recovery)

The provider includes a robust auto-reset feature to handle common failure
scenarios automatically. It is implemented in `deepseek_harness/provider.py`:

1. **Session Expiry / Invalid Session** – If the DeepSeek API returns an error
   related to an invalid or expired session, the provider detects this via
   keyword matching.
2. **Empty or Unparsable Responses** – An empty response with no
   `response_message_id` is raised as `EmptyResponseError` and treated as a
   session error.
3. **Detection Keywords** – `_is_session_error()` scans error messages for
   keywords like `"session"`, `"not found"`, `"invalid"`, `"expired"`,
   `"empty response"`, `"unparsable"`, `"provider returned"`, and always
   matches `EmptyResponseError`.
4. **Retry Logic** – `_MAX_ATTEMPTS = 2`. On detecting such an error, the
   provider:
   - Clears the in-memory session.
   - Creates a new session via `api.create_chat_session()`.
   - Updates `.session_data.json` and `.session_id` with the new session.
   - For **non-streaming** requests, it resets and retries the request once.
   - For **streaming** requests, it only retries when nothing has been emitted
     to the client yet (to avoid duplicated output).

This keeps the proxy functional without manual intervention (e.g., deleting
files and restarting).

---

## API Usage

### Base URLs

- `deepseek_harness`: `http://localhost:8002` (see `deepseek_harness.bat`)
- `openai_proxy`: `http://localhost:8000` (default)

### Endpoints

Both servers expose the same OpenAI-compatible surface. `deepseek_harness`
additionally exposes `/chat/completions` and `/models` aliases via
`fastapi-openai-compat`.

#### `POST /v1/chat/completions`
This endpoint mirrors the OpenAI API specification with additional session
management features.

**Request Body (JSON):**

| Field | Type | Description |
|-------|------|-------------|
| `messages` | `List[Message]` | Array of message objects with `role` and `content`. |
| `model` | `string` | One of the four model variants. Default: `thinking_not_search`. |
| `stream` | `boolean` | Enable streaming responses. Default: `false`. |
| `session_id` | `string` (optional) | Provide a specific session ID to continue a conversation. |
| `tools` | `List[object]` (optional) | OpenAI-style tool definitions. Their schema is injected into the prompt and used to parse textual tool calls. |
| `tool_choice` | `string`/`object` (optional) | Accepted for compatibility. |
| `temperature` | `float` (optional) | Sampling temperature (not fully implemented). |
| `max_tokens` | `integer` (optional) | Max tokens (not fully implemented). |

**Message Object:**

```json
{
  "role": "user",          // "user", "assistant", or "system"
  "content": "Hello, world!"
}
```

**Example Request (Non-Streaming):**

```bash
curl -X POST http://localhost:8002/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [{"role": "user", "content": "What is the capital of France?"}],
    "model": "thinking_not_search",
    "stream": false
  }'
```

**Example Response (Non-Streaming):**

```json
{
  "id": "chatcmpl-<session_id>",
  "object": "chat.completion",
  "created": 1234567890,
  "model": "thinking_not_search",
  "choices": [
    {
      "index": 0,
      "message": {
        "role": "assistant",
        "content": "The capital of France is Paris.",
        "reasoning_content": "(optional thinking process)"
      },
      "finish_reason": "stop"
    }
  ],
  "usage": {
    "prompt_tokens": 0,
    "completion_tokens": 0,
    "total_tokens": 0
  },
  "session_id": "abc-123",
  "response_message_id": 42
}
```

**Example Request (Streaming with tools):**

```bash
curl -X POST http://localhost:8002/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [{"role": "user", "content": "Search YouTube for AI agent tutorials."}],
    "model": "thinking_search",
    "stream": true,
    "tools": [
      {"type": "function", "function": {"name": "web_search",
        "parameters": {"type": "object", "properties": {"queries": {"type": "array"}}}}}
    ]
  }'
```

Streaming responses are Server-Sent Events (SSE) with `data:` prefixes,
compatible with OpenAI's streaming format. When the model emits a textual
tool call, the stream contains `tool_calls` deltas carrying a stable `index`,
followed by a `finish_reason="tool_calls"` chunk.

---

#### `GET /v1/models`
Returns the list of available models.

**Example Response:**

```json
{
  "object": "list",
  "data": [
    {"id": "thinking_not_search", "object": "model", "created": 1677610602, "owned_by": "deepseek"},
    {"id": "thinking_search", "object": "model", "created": 1677610602, "owned_by": "deepseek"},
    {"id": "not_thinking_not_search", "object": "model", "created": 1677610602, "owned_by": "deepseek"},
    {"id": "not_thinking_search", "object": "model", "created": 1677610602, "owned_by": "deepseek"}
  ]
}
```

---

## Utility Scripts

### `send_with_session.py`
This script provides a command-line interface for sending requests while
automatically managing session persistence. It reads the current session ID
from `.session_id` and reuses it across requests.

**Usage:**
```bash
python send_with_session.py "Your message here"
```

### `.bat` launchers (Windows)
- `deepseek_harness.bat` – starts the harness provider on port `8002`:
  `uvicorn deepseek_harness.main:app --host 127.0.0.1 --port 8002`
- `deepseek-api.bat` – starts the standalone proxy.
- `vscode-chat.bat` – starts the VS Code chat proxy.

---

## Testing

Several test scripts are included to verify functionality:

- `test_dsml_parser.py` – Verifies the nested tool-tag parser and schema
  handling across `write`, `web_search`, and `web_fetch`.
- `test_session_fixed.py` – Session handling with error recovery.
- `test_conversation.py` – Multi-turn conversation test.
- `example.py` – Demonstrates basic usage of the proxy.

You can run these individually to ensure the system works as expected.

---

## Troubleshooting

### Common Issues

| Issue | Solution |
|-------|----------|
| **Server won't start** | Check that `DEEPSEEK_API_KEY` is set in `.env` and that all dependencies are installed (`fastapi-openai-compat` included). |
| **`Cannot read properties of undefined (reading 'prepare')` (from the DeepSeek Harness client)** | This error is **not** produced by the proxy. It happens inside the DSH host when a tool call is dispatched: `packages/core/agent-loop/src/tool-calls.ts` reads `ctx.tools[TOOL_RUNTIME_SCHEDULER].prepare(...)`, where `TOOL_RUNTIME_SCHEDULER` is a `unique symbol` defined in `@deepseek-ai/dsh-tools`. The symbol resolves to `undefined` only when **two copies of `@deepseek-ai/dsh-tools`** are loaded (one that built `ToolRuntime`, another imported by `dsh-agent-loop`) — a dual-package hazard. **A controlled session proves the split**: a plain-text reply succeeds (`turn/end` with `reason: "completed"`), while every reply that emits a tool call fails at `turn/end` with this error, even though DSH already recorded the complete `tool/call` (`id`, `name`, `arguments`) — i.e. streaming, auth and textual parsing all work; only tool *dispatch* breaks. The proxy already emits a correct, complete tool call; the crash occurs before the tool body runs (the tool shows `0 ms` / `interrupted`). **Fix belongs on the DSH side:** dedupe `@deepseek-ai/dsh-tools` (e.g. `pnpm why @deepseek-ai/dsh-tools` then `pnpm dedupe`, or remove the nested `node_modules` copy under `packages/core/agent-loop/node_modules/@deepseek-ai/dsh-tools`), then reinstall/rebuild and restart DSH. Run `python diagnose_dsh_tools_dupes.py --root <dsh-checkout>` from this repo to confirm whether more than one physical copy exists. |
| **`INVALID_ARGS` / missing property from the harness** | The textual tool call was not parsed into the declared parameters. Verify `dsml_parser.parse_tool_calls_from_text` receives the request `tools` and that parameter aliases resolve to canonical names. |
| **"Session not found" errors** | The auto-reset mechanism should handle this automatically. If not, delete `.session_data.json` and `.session_id` and restart the server. |
| **Empty responses from DeepSeek** | The provider resets the session and retries once. If it persists, check your internet connection and DeepSeek availability. |
| **Cloudflare challenges** | The `common/bypass.py` module attempts to solve Cloudflare challenges. If issues persist, obtain cookies manually and place them in `common/cookies.json`. |

### Logging

The server logs all requests and responses with timestamps. Check the console
output for debugging information.

---

## Notes & Limitations

- DeepSeek's web chat endpoint has **no native tool calling**; tool support is
  implemented by parsing XML/DSML text from the model, which is inherently
  best-effort. The combination of a robust balanced parser plus exact schema
  injection maximizes reliability.
- Native tool calling (no text parsing) would require the **official DeepSeek
  API** (`https://api.deepseek.com`), which needs a separate API key and
  billing. The current `provider.py` structure is ready for such a backend to
  be added alongside the web-chat backend.
- **DSH integration note:** If a tool call fails inside DeepSeek Harness with
  `Cannot read properties of undefined (reading 'prepare')`, this is a DSH-side
  module-identity problem (a duplicated `@deepseek-ai/dsh-tools`, i.e. two
  loaded copies so the `unique symbol` `TOOL_RUNTIME_SCHEDULER` differs between
  the module that built `ToolRuntime` and the one read by `dsh-agent-loop`),
  not a proxy bug. The proxy's job ends at emitting a valid OpenAI `tool_calls`
  payload, which the DSH session log confirms it does. A controlled session
  makes the split unambiguous: pure text replies complete normally, and only
  replies that emit a tool call crash — so the transport, session and textual
  tool-call parsing are all healthy, and only DSH's tool-dispatch scheduler is
  unreachable. The helper `diagnose_dsh_tools_dupes.py` (in this repository)
  scans a DSH checkout for duplicate `@deepseek-ai/dsh-tools` installs to
  confirm the hazard before applying the dedupe/rebuild fix.

---

## License

This project is open-source and available under the [MIT License](LICENSE).

---

## Contributing

Contributions are welcome! Please submit issues and pull requests on the
[GitHub repository](https://github.com/Mircell/deepseek-api-kit-mircell).

---

## Acknowledgments

- DeepSeek for providing the language model API.
- [`fastapi-openai-compat`](https://github.com/deepset-ai/fastapi-openai-compat) for the OpenAI-compatible router factory.
- FastAPI for the web framework.
- All contributors and users of this project.

---

*Last Updated: September 2026*