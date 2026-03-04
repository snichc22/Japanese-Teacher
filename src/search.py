from langchain_community.tools import DuckDuckGoSearchResults
from langchain_core.tools import Tool


def get_search_tool():
    ddg_search = DuckDuckGoSearchResults(
        num_results=5,
        output_format="list"
    )

    return Tool(
        name="internet_search",
        description=(
            "Search the internet for song lyrics, slang definitions, "
            "cultural references, grammar explanations, and current language usage. "
            "Use this for: 歌詞検索 (lyrics search), スラング (slang), "
            "cultural context, or any real-time information."
        ),
        func=ddg_search.run
    )