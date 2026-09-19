"""Web tools: web_search and web_fetch.

These perform real HTTP requests. ``web_search`` queries a configurable search
endpoint (default: DuckDuckGo HTML) and returns a summary plus source URLs;
``web_fetch`` retrieves a URL and returns its decoded text. Both are
best-effort and treat fetched content as untrusted data.
"""

from __future__ import annotations

import html
import re
from typing import Any, List
from urllib.parse import quote_plus, unquote, urlparse

from .base import Tool, ToolError, ToolResult

_DEFAULT_UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/120.0 Safari/537.36"
)

_TAG_RE = re.compile(r"<[^>]+>")
_WS_RE = re.compile(r"[ \t\r\f\v]+")
_BLANK_RE = re.compile(r"\n{3,}")


def _strip_html(markup: str) -> str:
    """Convert HTML to readable text (no external deps)."""
    text = re.sub(r"(?is)<(script|style)[^>]*>.*?</\1>", " ", markup)
    text = re.sub(r"(?i)<br\s*/?>", "\n", text)
    text = re.sub(r"(?i)</(p|div|li|h[1-6]|tr)>", "\n", text)
    text = _TAG_RE.sub("", text)
    text = html.unescape(text)
    text = _WS_RE.sub(" ", text)
    text = _BLANK_RE.sub("\n\n", text)
    return text.strip()


def _http_get(url: str, timeout: float = 20.0, user_agent: str = _DEFAULT_UA) -> str:
    import requests  # imported lazily so the module loads without the dep

    resp = requests.get(
        url,
        timeout=timeout,
        headers={"User-Agent": user_agent, "Accept-Language": "en-US,en;q=0.9"},
    )
    resp.raise_for_status()
    return resp.text


class WebSearchTool(Tool):
    name = "web_search"
    description = "Search the web for current information and return a summary plus source URLs."
    parameters = {
        "type": "object",
        "properties": {
            "queries": {
                "type": "array",
                "description": "1-4 search queries.",
                "items": {"type": "string"},
            }
        },
        "required": ["queries"],
    }

    def __init__(self, search_endpoint: str = "https://duckduckgo.com/html/", user_agent: str = _DEFAULT_UA) -> None:
        self.search_endpoint = search_endpoint
        self.user_agent = user_agent

    def run(self, queries: Any, **_: Any) -> ToolResult:
        if isinstance(queries, str):
            queries = [queries]
        if not isinstance(queries, list) or not queries:
            raise ToolError("'queries' must be a non-empty array of strings (1-4).")
        queries = [str(q) for q in queries][:4]

        sources: List[str] = []
        snippets: List[str] = []
        for query in queries:
            url = f"{self.search_endpoint}?q={quote_plus(query)}"
            try:
                page = _http_get(url, user_agent=self.user_agent)
            except Exception as exc:  # noqa: BLE001 - network is best-effort
                snippets.append(f"[{query}] search failed: {exc}")
                continue

            for href, title in _extract_ddg_results(page):
                if href not in sources:
                    sources.append(href)
                    snippets.append(f"- {title} ({href})")

        if not snippets:
            return ToolResult.ok(
                "No results. (The search endpoint may block automated requests; "
                "configure a different 'search_endpoint'.)",
                queries=queries,
            )

        answer = f"Search results for: {', '.join(queries)}\n\n" + "\n".join(snippets)
        return ToolResult.ok(answer, queries=queries, sources=sources)


def _extract_ddg_results(page: str) -> List[tuple]:
    """Extract ``(url, title)`` pairs from a DuckDuckGo HTML results page."""
    results: List[tuple] = []
    pattern = r'<a[^>]+class="[^"]*result__a[^"]*"[^>]+href="([^"]+)"[^>]*>(.*?)</a>'
    for match in re.finditer(pattern, page, re.DOTALL | re.IGNORECASE):
        href = html.unescape(match.group(1))
        title = _strip_html(match.group(2))
        if "uddg=" in href:
            for part in urlparse(href).query.split("&"):
                if part.startswith("uddg="):
                    href = unquote(part[5:])
                    break
        if href.startswith("http") and title:
            results.append((href, title))
    return results


class WebFetchTool(Tool):
    name = "web_fetch"
    description = "Fetch a specific HTTP(S) URL and return its decoded text content."
    parameters = {
        "type": "object",
        "properties": {
            "url": {"type": "string", "description": "The HTTP(S) URL to fetch."},
        },
        "required": ["url"],
    }

    def __init__(self, user_agent: str = _DEFAULT_UA, timeout: float = 20.0) -> None:
        self.user_agent = user_agent
        self.timeout = timeout

    def run(self, url: str, **_: Any) -> ToolResult:
        if not url or not re.match(r"^https?://", url):
            raise ToolError("A valid http(s) URL is required.")
        try:
            page = _http_get(url, timeout=self.timeout, user_agent=self.user_agent)
        except Exception as exc:  # noqa: BLE001 - network is best-effort
            raise ToolError(f"Failed to fetch {url}: {exc}")

        text = _strip_html(page) if "<" in page[:2000] else page
        return ToolResult.ok(text, url=url)


def build_web_tools(search_endpoint: str = "https://duckduckgo.com/html/") -> List[Tool]:
    """Instantiate the web tool set."""
    return [WebSearchTool(search_endpoint=search_endpoint), WebFetchTool()]