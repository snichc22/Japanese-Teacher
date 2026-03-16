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

from langchain_classic.agents import create_tool_calling_agent, AgentExecutor
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.tools import Tool

from src.config import load_env
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
    load_env()
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
        verbose=os.getenv("AGENT_VERBOSE", "true").lower() == "true",
        max_iterations=int(os.getenv("AGENT_MAX_ITERATIONS", "2")),
        early_stopping_method="generate",
        handle_parsing_errors=True,
    )