import base64
import os
import shutil
from pathlib import Path

import gradio as gr
from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage

from src.agent import create_agent
from src.rag import DOCUMENT_DIR, ingest_documents

load_dotenv(".env.teacher")

_agent_cache: dict = {"reasoning": False, "agent": None}

def _get_agent(reasoning: bool):
    if _agent_cache["agent"] is None or _agent_cache["reasoning"] != reasoning:
        _agent_cache["reasoning"] = reasoning
        _agent_cache["agent"] = create_agent(reasoning=reasoning)
    return _agent_cache["agent"]

_get_agent(False)

_IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp"}


def _encode_image_to_data_url(path: str) -> str:
    ext = Path(path).suffix.lower()
    mime = {
        ".png": "image/png",
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".gif": "image/gif",
        ".webp": "image/webp",
        ".bmp": "image/bmp",
    }.get(ext, "image/png")
    with open(path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode()
    return f"data:{mime};base64,{b64}"

def _is_image(path: str) -> bool:
    return Path(path).suffix.lower() in _IMAGE_EXTENSIONS

def respond(message: dict, chat_history: list, thinking_enabled: bool):
    text: str = message.get("text", "").strip()
    files: list[str] = message.get("files", [])

    images = [f for f in files if _is_image(f)]

    content_blocks: list[dict] = []

    print(images)

    for img_path in images:
        content_blocks.append(
            {
                "type": "image_url",
                "image_url": {"url": _encode_image_to_data_url(img_path)},
            }
        )

    if text:
        content_blocks.append({"type": "text", "text": text})

    if not content_blocks:
        yield "Please send a message or an image! メッセージか画像を送ってください！"
        return

    lc_history: list = []
    for msg in chat_history:
        if msg["role"] == "user":
            lc_history.append(HumanMessage(content=msg["content"]))
        elif msg["role"] == "assistant":
            lc_history.append(AIMessage(content=msg["content"]))

    lc_input = [HumanMessage(content=content_blocks if images else text)]

    agent = _get_agent(thinking_enabled)
    result = agent.invoke({"input": lc_input, "chat_history": lc_history})

    output: str = result["output"]

    if thinking_enabled:
        reasoning_text = ""
        for step in reversed(result.get("intermediate_steps", [])):
            pass
        if hasattr(result, "get"):
            raw_messages = result.get("messages", [])
            for m in reversed(raw_messages):
                if hasattr(m, "additional_kwargs"):
                    reasoning_text = m.additional_kwargs.get(
                        "reasoning_content", ""
                    )
                    if reasoning_text:
                        break

        if reasoning_text:
            output = (
                f"<details><summary>Thinking Process (思考過程)</summary>"
                f"\n\n{reasoning_text}\n\n</details>\n\n{output}"
            )

    yield output


def ingest_files(files):
    if not files:
        return "No files selected."
    os.makedirs(DOCUMENT_DIR, exist_ok=True)
    for file in files:
        shutil.copy(file.name, f"{DOCUMENT_DIR}/{os.path.basename(file.name)}")
    ingest_documents()
    return f"Ingested {len(files)} file(s). ファイルを{len(files)}件取り込みました。"

with gr.Blocks(title="Japanese Teacher") as demo:
    gr.Markdown("# 🇯🇵 Japanese Teacher（日本語の先生）")
    gr.Markdown(
        "Ask me anything about Japanese or English! "
        "Send images of Japanese text for translation.\n\n"
        "日本語でも英語でも質問してください！日本語のテキスト画像も送れます。"
    )

    thinking_toggle = gr.Checkbox(
        label="Enable Thinking / 思考モードを有効にする",
        value=False,
        info="When enabled, the model reasons step-by-step (slower). "
        "Disable for faster, direct answers.",
    )

    chatbot = gr.Chatbot(
        label="Chat",
        height=520,
    )

    msg = gr.MultimodalTextbox(
        placeholder="Type a message or upload an image of Japanese text…",
        file_types=["image"],
        show_label=False,
        scale=8,
    )

    gr.Examples(
        examples=[
            [{"text": "Teach me common Japanese slang"}],
            [{"text": "「木漏れ日」とはどういう意味ですか？英語で説明してください"}],
            [{"text": "What are the lyrics to Yoasobi's 'Idol' and what do they mean?"}],
            [{"text": "Explain the difference between は and が"}],
            [{"text": "How do I say 'it can't be helped' naturally in Japanese?"}],
        ],
        inputs=[msg],
    )

    with gr.Accordion(
        "Upload Learning Materials（教材をアップロード）", open=False
    ):
        file_upload = gr.File(
            label="Upload PDFs, text files, or markdown notes",
            file_count="multiple",
        )
        ingest_button = gr.Button("Ingest Files（ファイルを取り込む）")
        ingest_output = gr.Textbox(label="Status")
        ingest_button.click(ingest_files, inputs=file_upload, outputs=ingest_output)

    saved_msg = gr.State({})

    def user_message(message, history):
        text = message.get("text", "")
        files = message.get("files", [])

        parts: list = []
        for f in files:
            if _is_image(f):
                parts.append(gr.FileData(path=f))
        if text:
            parts.append(text)

        user_content = parts[0] if len(parts) == 1 else (parts or text)

        history = history + [{"role": "user", "content": user_content}]
        return gr.MultimodalTextbox(value=None), history, message

    def bot_response(original_message, history, thinking):
        history = history + [{"role": "assistant", "content": ""}]
        for chunk in respond(original_message, history[:-1], thinking):
            history[-1]["content"] = chunk
            yield history

    msg.submit(
        user_message, [msg, chatbot], [msg, chatbot, saved_msg]
    ).then(
        bot_response, [saved_msg, chatbot, thinking_toggle], chatbot
    )


if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
