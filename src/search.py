#  Copyright (c) 2026 Schnitzer Christoph. All rights reserved.
#
#  Permission is hereby granted, free of charge, to any person obtaining a copy
#  of this software and associated documentation files (the "Software"), to deal
#  in the Software without restriction, including without limitation the rights
#  to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
#  copies of the Software, and to permit persons to whom the Software is
#  furnished to do so, subject to the following conditions:
#
#  The above copyright notice and this permission notice shall be included in all
#  copies or substantial portions of the Software.
#
#  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
#  IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
#  FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
#  AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
#  LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
#  OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
#  SOFTWARE.

import os
import re

from langchain_community.tools import DuckDuckGoSearchResults
from langchain_community.utilities import DuckDuckGoSearchAPIWrapper
from langchain_core.tools import Tool

from src.config import load_env

# Domains with unwanted language results (Change if you want to allow any of these)
_BLOCKED_TLD_PATTERN = re.compile(
    r"https?://[^/]*\.("
    r"ru|es|fr|pt|br|it|pl|nl|tr|kr|cn|tw|ar|th|vn|id|ro|cz|hu|se|no|dk|fi"
    r")(?:/|$)",
    re.IGNORECASE,
)

_ALLOWED_LANG_TLDS = {".jp", ".de", ".com", ".org", ".net", ".edu", ".gov", ".co.uk", ".io"}

load_env()

def _env_int(name: str, default: int) -> int:
    value = os.getenv(name, str(default)).strip()
    try:
        return int(value)
    except ValueError:
        return default

def _filter_results(results: list[dict]) -> list[dict]:
    filtered = []
    for r in results:
        link = r.get("link", "")
        if _BLOCKED_TLD_PATTERN.search(link):
            continue
        filtered.append(r)
    return filtered if filtered else results

def _truncate(text: str, limit: int) -> str:
    text = re.sub(r"\s+", " ", text or "").strip()
    if len(text) <= limit:
        return text
    return f"{text[: limit - 3].rstrip()}..."

def _format_results(results: list[dict], max_returned_results: int, max_snippet_chars: int) -> str:
    rows: list[str] = []
    for idx, item in enumerate(results[:max_returned_results], start=1):
        title = _truncate(item.get("title", "Untitled"), 120)
        link = item.get("link", "")
        snippet = _truncate(item.get("snippet", ""), max_snippet_chars)
        rows.append(f"{idx}. {title}\nURL: {link}\nSnippet: {snippet}")
    if not rows:
        return "No useful search results found."
    return "\n\n".join(rows)

def get_search_tool():
    load_env()
    max_returned_results = _env_int("SEARCH_MAX_RETURNED_RESULTS", 3)
    max_snippet_chars = _env_int("SEARCH_MAX_SNIPPET_CHARS", 220)
    max_results = _env_int("SEARCH_MAX_RESULTS", 5)

    wrapper = DuckDuckGoSearchAPIWrapper(
        max_results=max_results,
        backend="auto",
    )
    ddg_search = DuckDuckGoSearchResults(
        api_wrapper=wrapper,
        num_results=max_results,
        output_format="list",
    )

    def _safe_search(query: str) -> str:
        try:
            raw = ddg_search.run(query)
            if isinstance(raw, list):
                return _format_results(_filter_results(raw), max_returned_results, max_snippet_chars)
            return raw
        except Exception as e:
            return f"Search failed: {e}. Please try rephrasing the query or answer from your own knowledge."

    return Tool(
        name="internet_search",
        description=(
            "Search the internet for song lyrics, slang definitions, "
            "cultural references, grammar explanations, and current language usage. "
            "Use this for: 歌詞検索 (lyrics search), スラング (slang), "
            "cultural context, or any real-time information.\n"
            "IMPORTANT: The student only understands English, Japanese, and German. "
            "Always write search queries in English or Japanese. "
            "Do NOT search in or cite sources in Spanish, Russian, French, Chinese, "
            "Korean, or any other language. Ignore non-EN/JA/DE results."
        ),
        func=_safe_search,
    )