"""Ad-hoc verification for the rewritten dsml_parser (nested tool-call tags)."""

from __future__ import annotations

import json

from deepseek_harness.dsml_parser import (
    build_tool_params,
    parse_tool_calls_from_text,
    remove_tool_tags,
)

# 1. write with a nested <path> + HTML <content> (the exact failing case).
write_text = (
    "Sure, I'll create the game.\n\n"
    "<write>\n"
    "<path>snake-game.html</path>\n"
    "<content><!DOCTYPE html>\n"
    "<html><body><h1>hi</h1></body></html></content>\n"
    "</write>"
)

# 2. web_search with a JSON array of queries.
search_text = (
    "Let me search.\n\n"
    "<web_search>\n"
    '<queries>["Millennium Prize Problems", "Clay Institute"]</queries>\n'
    "</web_search>"
)

# 3. web_fetch with a url.
fetch_text = "Fetching.\n\n<web_fetch>\n<url>https://www.claymath.org/million-problems/</url>\n</web_fetch>"

# 4. The canonical name already (file_path instead of path) must still work.
canonical_text = (
    "<write>\n<file_path>a.txt</file_path>\n<content>hello</content>\n</write>"
)

# 5. The malformed literal "<tool_name>NAME</tool_name>" form some models copy
#    from the placeholder in the prompt guide, followed by a stray closing tag.
malformed_text = (
    "I'll fetch the paper.\n\n"
    "<tool_name>web_fetch</tool_name>\n"
    "<url>https://arxiv.org/abs/2607.02770</url>\n"
    "</tool_name>"
)

# Tool schema as the DSH harness sends it (parameters.properties keys).
TOOLS = [
    {"type": "function", "function": {"name": "write", "parameters": {"properties": {"file_path": {}, "content": {}}}}},
    {"type": "function", "function": {"name": "web_search", "parameters": {"properties": {"queries": {}}}}},
    {"type": "function", "function": {"name": "web_fetch", "parameters": {"properties": {"url": {}}}}},
]


def _show(label: str, text: str) -> None:
    calls = parse_tool_calls_from_text(text, TOOLS)
    print(f"--- {label} ---")
    print(json.dumps(calls, ensure_ascii=False, indent=2))
    print("clean:", repr(remove_tool_tags(text, TOOLS)[:80]))
    print()


if __name__ == "__main__":
    print("tool params:", build_tool_params(TOOLS))
    print()
    _show("write (path alias + nested html)", write_text)
    _show("web_search (JSON array)", search_text)
    _show("web_fetch (url)", fetch_text)
    _show("write (canonical file_path)", canonical_text)
    _show("web_fetch (malformed literal tool_name)", malformed_text)
