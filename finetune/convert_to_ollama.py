import subprocess

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

BASE_MODEL = "Qwen/Qwen3.5-9B-Base"
LORA_ADAPTER = "finetune/lora_adapter"
MERGED_OUTPUT = "finetune/merged_model"

def merge_and_export():
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True
    )
    tokenizer = AutoTokenizer.from_pretrained(model, LORA_ADAPTER)

    model = PeftModel.from_pretrained(model, LORA_ADAPTER)
    model = model.merge_and_unload()
    model.save_pretrained(MERGED_OUTPUT)

    modelfile_content = f"""FROM {MERGED_OUTPUT}
    
PARAMETER temperature 0.7
PARAMETER num_ctx 8192
PARAMETER num_gpu 99

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

    with open("Modelfile.finetuned", "w") as f:
        f.write(modelfile_content)

    subprocess.run(
        ["ollama", "create", "japanese-teacher-ft", "-f", "Modelfile.finetuned"],
        check=True
    )
    print("Model merged and exported to Ollama as 'japanese-teacher-ft'.")
    print("Run with: ollama run japanese-teacher-ft")

if __name__ == "__main__":
    merge_and_export()