import json
import subprocess
import shutil
import sys
from pathlib import Path

import torch
from dotenv import load_dotenv
from peft import PeftModel
from transformers import AutoConfig, Qwen3_5ForCausalLM, AutoTokenizer

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent

load_dotenv(PROJECT_ROOT / ".env.teacher")

BASE_MODEL = "Qwen/Qwen3.5-9B-Base"
# The instruct variant has the full chat template (tool-calling, thinking, vision).
# Qwen3.5 naming: "Qwen3.5-9B" = instruct, "Qwen3.5-9B-Base" = base (no -Instruct suffix).
# Only used to copy the chat template into the merged model — no weights are taken from it.
INSTRUCT_MODEL = "Qwen/Qwen3.5-9B"
LORA_ADAPTER = "lora_adapter"
MERGED_OUTPUT = "merged_model"
GGUF_OUTPUT = "merged_model.gguf"
GGUF_OUTTYPE = "f16"

def find_llama_cpp_converter():
    converter = shutil.which("convert_hf_to_gguf")
    if converter:
        return [converter]

    candidates = [
        PROJECT_ROOT / "llama.cpp" / "convert_hf_to_gguf.py",
        Path.home() / "llama.cpp" / "convert_hf_to_gguf.py",
    ]
    for p in candidates:
        if p.exists():
            return [sys.executable, str(p.resolve())]

    return None


def inject_chat_template(merged_output_dir: Path):
    tokenizer_config_path = merged_output_dir / "tokenizer_config.json"
    if not tokenizer_config_path.exists():
        print("WARNING: tokenizer_config.json not found in merged model directory.")
        return

    local_jinja = SCRIPT_DIR / LORA_ADAPTER / "chat_template.jinja"
    chat_template: str | None = None

    if local_jinja.exists():
        chat_template = local_jinja.read_text(encoding="utf-8")
        print(f"Using chat template from {local_jinja}")
    else:
        print(f"Local chat_template.jinja not found; downloading from {INSTRUCT_MODEL}...")
        try:
            instruct_tok = AutoTokenizer.from_pretrained(
                INSTRUCT_MODEL, trust_remote_code=True
            )
            chat_template = instruct_tok.chat_template

            local_jinja.parent.mkdir(parents=True, exist_ok=True)
            local_jinja.write_text(chat_template or "", encoding="utf-8")
            print(f"Saved chat template to {local_jinja}")
        except Exception as e:
            print(f"WARNING: Could not load instruct tokenizer: {e}")
            print("Tool-calling will NOT be available in the fine-tuned model.")
            return

    if not chat_template:
        print("WARNING: chat_template is empty — skipping injection.")
        return

    with open(tokenizer_config_path, "r", encoding="utf-8") as f:
        config = json.load(f)

    config["chat_template"] = chat_template

    with open(tokenizer_config_path, "w", encoding="utf-8") as f:
        json.dump(config, f, ensure_ascii=False, indent=2)

    print("✅ Chat template injected into tokenizer_config.json — tool-calling enabled.")


def merge_and_export():
    config = AutoConfig.from_pretrained(BASE_MODEL, trust_remote_code=True)
    text_config = config.text_config

    model = Qwen3_5ForCausalLM.from_pretrained(
        BASE_MODEL,
        config=text_config,
        torch_dtype=torch.bfloat16,
        device_map="cpu",
        trust_remote_code=True
    )
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)

    # Workaround: transformers 5.x stores _no_split_modules as a set, but
    # accelerate 1.13 chokes on it in get_balanced_memory(). Flatten to list.
    if hasattr(model, "_no_split_modules") and isinstance(model._no_split_modules, set):
        model._no_split_modules = list(model._no_split_modules)

    print("Loading LoRA adapter and merging with base model...")
    model = PeftModel.from_pretrained(model, LORA_ADAPTER, device_map="cpu")
    model = model.merge_and_unload()

    print(f"Saving merged model to {MERGED_OUTPUT}...")
    model.save_pretrained(MERGED_OUTPUT)
    tokenizer.save_pretrained(MERGED_OUTPUT)

    # --- Inject the instruct chat template so the GGUF supports tool-calling ---
    inject_chat_template(SCRIPT_DIR / MERGED_OUTPUT)

    # --- Convert to GGUF format ---
    # Ollama cannot directly import Qwen3.5 safetensors (unsupported architecture).
    print(f"Converting merged model to GGUF format (outtype={GGUF_OUTTYPE})...")
    gguf_path = Path(GGUF_OUTPUT)

    convert_cmd = find_llama_cpp_converter()
    if convert_cmd is None:
        print("ERROR: Could not find convert_hf_to_gguf.py!")
        print("Expected it at: " + str(PROJECT_ROOT / "llama.cpp" / "convert_hf_to_gguf.py"))
        print("\nTo fix this:")
        print(f"  cd \"{PROJECT_ROOT}\"")
        print("  git clone https://github.com/ggml-org/llama.cpp")
        print("  pip install gguf")
        sys.exit(1)

    convert_args = convert_cmd + [
        MERGED_OUTPUT,
        "--outfile", str(gguf_path),
        "--outtype", GGUF_OUTTYPE,
    ]

    print(f"Running: {' '.join(convert_args)}")
    result = subprocess.run(convert_args)

    if result.returncode != 0:
        print("GGUF conversion failed!")
        print(f"\nYou can try manually converting:")
        print(f"  cd \"{SCRIPT_DIR}\"")
        print(f"  python \"{PROJECT_ROOT / 'llama.cpp' / 'convert_hf_to_gguf.py'}\" {MERGED_OUTPUT} --outfile {GGUF_OUTPUT} --outtype {GGUF_OUTTYPE}")
        sys.exit(1)

    print(f"GGUF conversion complete: {gguf_path}")

    modelfile_content = f"""FROM finetune/{gguf_path}

PARAMETER temperature 0.7
PARAMETER num_ctx 32768
PARAMETER num_gpu 99
PARAMETER stop "<|im_end|>"
PARAMETER stop "<|endoftext|>"

SYSTEM \"\"\"
You are a billingual language teacher fluent in both Japanese (日本語) and English.

Your responsibilities:
- Teach Japanese to English speakers
- Explain grammar, vocabulary, kanji, slang, and cultural context
- When the user speaks Japanese, respond primarily in Japanese with English explanations
- When the user speaks English, respond in Japanese and English with explanations
- Use furigana notation like 漢字(かんじ) for difficult kanji
- You can look up song lyrics, modern slang, and cultural references
- You can analyze images of Japanese text, kanji, signs, menus, etc.
- Be encouraging, patient, and adapt to the student's level

あなたは日英バイリンガルのAI語学教師です。生徒のレベルに合わせて教えてください。
画像の中の日本語テキストも分析できます。
\"\"\"
"""

    with open("../Modelfile.finetuned", "w", encoding="utf-8") as f:
        f.write(modelfile_content)

    print("Creating Ollama model...")
    subprocess.run(
        ["ollama", "create", "japanese-teacher-ft", "-f", "../Modelfile.finetuned"],
        check=True
    )
    print("Model merged and exported to Ollama as 'japanese-teacher-ft'.")
    print("Run with: ollama run japanese-teacher-ft")

if __name__ == "__main__":
    merge_and_export()