"""Verification for the rewritten dsml_parser (both tool-call dialects).

Run directly: ``python test_dsml_parser.py``. Exits non-zero on any failed
assertion so it works as a regression check, not just a printer.
"""

from __future__ import annotations

import json
import sys
from typing import Any, Dict, List

from deepseek_harness.dsml_parser import (
    build_tool_params,
    parse_tool_calls_from_text,
    remove_tool_tags,
)

BAR = "\uff5c"
D = f"{BAR}DSML{BAR}"

# 1. write with a nested <path> + HTML <content> (the original failing case).
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

# 6. The model's native dialect EXACTLY as it leaked into a real session:
#    a `calls` wrapper, a space after the bar, and a stray closing invoke tag.
#    This is the shape the old regexes could not match at all.
native_exact_text = (
    "نتایج را در فایل جمع می‌کنم.\n\n"
    f"<{D} calls>\n"
    f'<{D} invoke name="web_fetch">\n'
    f'<{D} parameter name="url" string="true">https://typesafe.ai/blog</{D} parameter>\n'
    f"</{D} invoke>\n"
    f"</{D} calls>"
)

# 7. Native dialect, but the model invented the alias `link` instead of `url`
#    and dropped the `string` attribute.
native_alias_text = (
    f'<{D} invoke name="web_fetch">'
    f'<{D} parameter name="link">https://docs.typesafe.ai/models</{D} parameter>'
    f"</{D} invoke>"
)

# 8. Native dialect for `write` carrying a file path alias and HTML content.
native_write_text = (
    f'<{D} invoke name="write">'
    f'<{D} parameter name="path" string="true">snake-game.html</{D} parameter>'
    f'<{D} parameter name="content" string="true"><!DOCTYPE html><html></html></{D} parameter>'
    f"</{D} invoke>"
)

# 9. A reply that mixes both dialects must not lose the plain-XML call.
mixed_text = (
    "Fetching then searching.\n\n"
    "<web_fetch>\n<url>https://example.com/a</url>\n</web_fetch>\n\n"
    f'<{D} invoke name="web_search">'
    f'<{D} parameter name="queries" string="true">["one", "two"]</{D} parameter>'
    f"</{D} invoke>"
)

# 10. A hallucinated tool name in the native dialect is dropped when the
#     request declared a real schema.
hallucinated_text = (
    f'<{D} invoke name="definitely_not_a_tool">'
    f'<{D} parameter name="whatever" string="true">x</{D} parameter>'
    f"</{D} invoke>"
)

# 11. The SAME hallucinated name is honoured when no schema was declared, since
#     the proxy cannot know it is invalid.
hallucinated_no_schema_text = (
    f'<{D} invoke name="mystery_tool">'
    f'<{D} parameter name="alpha" string="true">x</{D} parameter>'
    f"</{D} invoke>"
)

# 12. A native invocation that declares parameters but supplies none is
#     malformed, not a call, and must not become a bogus `{"content": ...}`.
native_empty_params_text = (
    f'<{D} invoke name="web_fetch">garbage with no parameter tags</{D} invoke>'
)

# 13. Native `write` with a MULTI-LINE markdown body and the parameter's
#     closing tag MISSING — the exact shape that broke DSH. The value must run
#     to the next parameter/end of invoke, not require a closing tag.
native_write_missing_close_text = (
    f'<{D} calls>\n'
    f'<{D} invoke name="write">\n'
    f'<{D} parameter name="file_path" string="true">jev-llmmodel.md</{D} parameter>\n'
    f'<{D} parameter name="content" string="true"># Jev — مدل «System One» شرکت TypeSafe AI\n\n'
    f'## خلاصه\n\nاین یک متن طولانی است.\n'
    f'</{D} invoke>\n'
    f'</{D} calls>'
)

# 14. Native `write` where the content itself embeds text that LOOKS like a
#     closing parameter tag; delimiter-based extraction must keep it intact.
native_write_embedded_text = (
    f'<{D} invoke name="write">'
    f'<{D} parameter name="path" string="true">a.md</{D} parameter>'
    f'<{D} parameter name="content" string="true">line1\n</{D} parameter> not the end\nline3</{D} parameter>'
    f"</{D} invoke>"
)

# 15. Native `write` with content placed BEFORE file_path (parameter order the
#     model sometimes flips) must still resolve both.
native_write_reordered_text = (
    f'<{D} invoke name="write">'
    f'<{D} parameter name="content" string="true">hello world</{D} parameter>'
    f'<{D} parameter name="file_path" string="true">b.txt</{D} parameter>'
    f"</{D} invoke>"
)

# Tool schema as the DSH harness sends it (parameters.properties keys).
TOOLS: List[Dict[str, Any]] = [
    {"type": "function", "function": {"name": "write", "parameters": {"properties": {"file_path": {}, "content": {}}}}},
    {"type": "function", "function": {"name": "web_search", "parameters": {"properties": {"queries": {}}}}},
    {"type": "function", "function": {"name": "web_fetch", "parameters": {"properties": {"url": {}}}}},
]

_failures: List[str] = []


def _args(call: Dict[str, Any]) -> Dict[str, Any]:
    return json.loads(call["function"]["arguments"])


def _names(calls: List[Dict[str, Any]]) -> List[str]:
    return [call["function"]["name"] for call in calls]


def _check(label: str, condition: bool, detail: str = "") -> None:
    status = "ok  " if condition else "FAIL"
    print(f"  [{status}] {label}{'' if condition else f' -> {detail}'}")
    if not condition:
        _failures.append(f"{label}: {detail}")


def _show(label: str, text: str) -> List[Dict[str, Any]]:
    calls = parse_tool_calls_from_text(text, TOOLS)
    print(f"--- {label} ---")
    print(json.dumps(calls, ensure_ascii=False, indent=2))
    print("  clean:", repr(remove_tool_tags(text, TOOLS)[:70]))
    return calls


def main() -> int:
    print("tool params:", build_tool_params(TOOLS))
    print()

    calls = _show("write (path alias + nested html)", write_text)
    _check("write parsed", _names(calls) == ["write"], str(_names(calls)))
    _check("path -> file_path", _args(calls[0]).get("file_path") == "snake-game.html", str(_args(calls[0])))

    calls = _show("web_search (JSON array)", search_text)
    _check("queries is a list", _args(calls[0]).get("queries") == ["Millennium Prize Problems", "Clay Institute"], str(_args(calls[0])))

    calls = _show("web_fetch (url)", fetch_text)
    _check("url captured", _args(calls[0]).get("url") == "https://www.claymath.org/million-problems/", str(_args(calls[0])))

    calls = _show("write (canonical file_path)", canonical_text)
    _check("canonical name kept", _args(calls[0]).get("file_path") == "a.txt", str(_args(calls[0])))

    calls = _show("web_fetch (malformed literal tool_name)", malformed_text)
    _check("literal form recovered", _names(calls) == ["web_fetch"], str(_names(calls)))
    _check("literal url captured", _args(calls[0]).get("url") == "https://arxiv.org/abs/2607.02770", str(_args(calls[0])))

    # --- native dialect: the exact shape that broke the old regexes ---
    calls = _show("native exact (space after bar + calls wrapper)", native_exact_text)
    _check("native call recovered", _names(calls) == ["web_fetch"], str(_names(calls)))
    _check("native url captured", _args(calls[0]).get("url") == "https://typesafe.ai/blog", str(_args(calls[0])))
    cleaned = remove_tool_tags(native_exact_text, TOOLS)
    _check("wrapper stripped from visible text", "DSML" not in cleaned and "calls" not in cleaned, repr(cleaned))
    _check("prose kept in visible text", "نتایج را در فایل جمع" in cleaned, repr(cleaned))

    calls = _show("native alias (link -> url, no string attr)", native_alias_text)
    _check("alias canonicalised to url", _args(calls[0]).get("url") == "https://docs.typesafe.ai/models", str(_args(calls[0])))

    calls = _show("native write (path + html content)", native_write_text)
    _check("native write parsed", _names(calls) == ["write"], str(_names(calls)))
    _check("native path -> file_path", _args(calls[0]).get("file_path") == "snake-game.html", str(_args(calls[0])))
    _check("native html content intact", _args(calls[0]).get("content") == "<!DOCTYPE html><html></html>", str(_args(calls[0])))

    calls = _show("mixed (plain XML + native DSML)", mixed_text)
    _check("both dialects returned", sorted(_names(calls)) == ["web_fetch", "web_search"], str(_names(calls)))

    calls = _show("native hallucinated tool (schema declared)", hallucinated_text)
    _check("hallucinated tool dropped", calls == [], str(calls))

    # No schema at all: the proxy cannot know the name is invalid, so it is
    # forwarded as-is (parsed with the built-in fallback table, not `TOOLS`).
    print("--- native unknown tool (no schema) ---")
    calls = parse_tool_calls_from_text(hallucinated_no_schema_text)
    print(json.dumps(calls, ensure_ascii=False, indent=2))
    _check("unknown tool kept without schema", _names(calls) == ["mystery_tool"], str(_names(calls)))
    _check(
        "unknown tool params kept without schema",
        _args(calls[0]).get("alpha") == "x",
        str(_args(calls[0])),
    )

    calls = _show("native with params declared but none supplied", native_empty_params_text)
    _check("malformed emission rejected", calls == [], str(calls))

    # --- write + long/missing-close content: the reported DSH failure ---
    calls = _show("native write (long markdown, MISSING close tag)", native_write_missing_close_text)
    _check("write parsed despite missing close", _names(calls) == ["write"], str(_names(calls)))
    _check(
        "missing-close file_path captured",
        _args(calls[0]).get("file_path") == "jev-llmmodel.md",
        str(_args(calls[0])),
    )
    _check(
        "missing-close content captured intact",
        "# Jev — مدل «System One» شرکت TypeSafe AI" in _args(calls[0]).get("content", "")
        and _args(calls[0]).get("content", "").rstrip().endswith("این یک متن طولانی است."),
        str(_args(calls[0])),
    )

    calls = _show("native write (content embeds a close-tag-like line)", native_write_embedded_text)
    _check("embedded-close write parsed", _names(calls) == ["write"], str(_names(calls)))
    _check(
        "embedded-close content kept",
        "not the end" in _args(calls[0]).get("content", "")
        and "line3" in _args(calls[0]).get("content", ""),
        str(_args(calls[0])),
    )
    _check("embedded-close path captured", _args(calls[0]).get("file_path") == "a.md", str(_args(calls[0])))

    calls = _show("native write (content before file_path)", native_write_reordered_text)
    _check("reordered write parsed", _names(calls) == ["write"], str(_names(calls)))
    _check("reordered path captured", _args(calls[0]).get("file_path") == "b.txt", str(_args(calls[0])))
    _check("reordered content captured", _args(calls[0]).get("content") == "hello world", str(_args(calls[0])))

    print()
    if _failures:
        print(f"{len(_failures)} check(s) FAILED:")
        for failure in _failures:
            print(f"  - {failure}")
        return 1
    print("all checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())