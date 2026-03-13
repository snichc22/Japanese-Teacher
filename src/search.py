import re

from langchain_community.tools import DuckDuckGoSearchResults
from langchain_community.utilities import DuckDuckGoSearchAPIWrapper
from langchain_core.tools import Tool

# Domains with unwanted language results (Change if you want to allow any of these)
_BLOCKED_TLD_PATTERN = re.compile(
    r"https?://[^/]*\.("
    r"ru|es|fr|pt|br|it|pl|nl|tr|kr|cn|tw|ar|th|vn|id|ro|cz|hu|se|no|dk|fi"
    r")(?:/|$)",
    re.IGNORECASE,
)

_ALLOWED_LANG_TLDS = {".jp", ".de", ".com", ".org", ".net", ".edu", ".gov", ".co.uk", ".io"}


def _filter_results(results: list[dict]) -> list[dict]:
    filtered = []
    for r in results:
        link = r.get("link", "")
        if _BLOCKED_TLD_PATTERN.search(link):
            continue
        filtered.append(r)
    return filtered if filtered else results


def get_search_tool():
    wrapper = DuckDuckGoSearchAPIWrapper(
        max_results=8,
        backend="auto",
    )
    ddg_search = DuckDuckGoSearchResults(
        api_wrapper=wrapper,
        num_results=8,
        output_format="list",
    )

    def _safe_search(query: str) -> str:
        try:
            raw = ddg_search.run(query)
            if isinstance(raw, list):
                return str(_filter_results(raw)[:5])
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