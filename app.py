import os
import shutil
from pathlib import Path

import gradio
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, AIMessage

from src.agent import create_agent
from src.rag import DOCUMENT_DIR, ingest_documents

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent

load_dotenv(PROJECT_ROOT / ".env.teacher")

agent = create_agent()

def respond(message: str, chat_history: list):
    lc_history = []
    for msg in chat_history:
        if msg["role"] == "user":
            lc_history.append(HumanMessage(content=msg["content"]))
        elif msg["role"] == "assistant":
            lc_history.append(AIMessage(content=msg["content"]))

    result = agent.invoke({
        "input": message,
        "chat_history": lc_history
    })

    return result["output"]

def ingest_files(files):
    os.makedirs(DOCUMENT_DIR, exist_ok=True)
    for file in files:
        shutil.copy(file.name, f"{DOCUMENT_DIR}/{os.path.basename(file.name)}")

    ingest_documents()
    return f"Ingested {len(files)} files."

with gradio.Blocks(
    title="Japanese Teacher",
) as demo:
    gradio.Markdown("# Japanese Teacher")
    gradio.Markdown(
        "Ask me anything about Japanese or English! "
        "日本語でも英語でも質問してください！"
    )

    chatbot = gradio.ChatInterface(
        fn=respond,
        examples=[
            "Teach me common Japanese slang",
            "「木漏れ日」とはどういう意味ですか？英語で説明してください",
            "What are the lyrics to Yoasobi's 'Idol' and what do they mean?",
            "Explain the difference between は and が",
            "How do I say 'it can't be helped' naturally in Japanese?",
        ]
    )

    with gradio.Accordion("Upload Learning Materials (教材をアップロード)", open=False):
        file_upload = gradio.File(
            label="Upload PDFs, text files, or markdown notes",
            file_count="multiple"
        )
        ingest_button = gradio.Button("Ingest Files (ファイルを取り込む)")
        ingest_output = gradio.Textbox(label="Status")
        ingest_button.click(ingest_files, inputs=file_upload, outputs=ingest_output)

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, theme=gradio.themes.Soft())
