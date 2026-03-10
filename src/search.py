from langchain_community.tools import DuckDuckGoSearchResults
from langchain_community.utilities import DuckDuckGoSearchAPIWrapper
from langchain_core.tools import Tool


def get_search_tool():
    wrapper = DuckDuckGoSearchAPIWrapper(
        max_results=5,
        backend="auto",
    )
    ddg_search = DuckDuckGoSearchResults(
        api_wrapper=wrapper,
        num_results=5,
        output_format="list"
    )

    def _safe_search(query: str) -> str:
        try:
            return ddg_search.run(query)
        except Exception as e:
            return f"Search failed: {e}. Please try rephrasing the query or answer from your own knowledge."

    return Tool(
        name="internet_search",
        description=(
            "Search the internet for song lyrics, slang definitions, "
            "cultural references, grammar explanations, and current language usage. "
            "Use this for: 歌詞検索 (lyrics search), スラング (slang), "
            "cultural context, or any real-time information."
        ),
        func=_safe_search,
    )