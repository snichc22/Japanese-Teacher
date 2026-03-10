from langchain_classic.agents import create_tool_calling_agent, AgentExecutor
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.tools import Tool

from src.llm import get_llm
from src.prompts import SYSTEM_PROMPT
from src.rag import get_retriever
from src.search import get_search_tool


def _get_rag_tool():
    try:
        retriever = get_retriever()

        def query_documents(query: str) -> str:
            docs = retriever.invoke(query)
            if not docs:
                return "No relevant documents found."
            return "\n\n---\n\n".join(
                f"Source: {d.metadata.get('source', 'unknown')}\n{d.page_content}"
                for d in docs
            )

        return Tool(
            name="document_search",
            description=(
                "Search through the student's uploaded learning materials, "
                "textbooks, notes, and reference documents. "
                "学習資料やノートの中から情報を検索します。"
            ),
            func=query_documents
        )
    except Exception:
        return None

def create_agent(reasoning: bool | None = False):
    llm = get_llm(reasoning=reasoning)
    tools = [get_search_tool()]
    rag_tool = _get_rag_tool()

    if rag_tool:
        tools.append(rag_tool)

    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        MessagesPlaceholder(variable_name="chat_history"),
        MessagesPlaceholder(variable_name="input"),
        MessagesPlaceholder(variable_name="agent_scratchpad")
    ])

    agent = create_tool_calling_agent(llm, tools, prompt)

    return AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=True,
        max_iterations=5,
        handle_parsing_errors=True
    )