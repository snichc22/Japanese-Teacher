import os
from langchain_ollama import ChatOllama

# Ollama model names:
# japanese-teacher
# japanese-teacher-ft
_DEFAULT_MODEL = os.getenv("OLLAMA_MODEL", "japanese-teacher-ft")

print("Model in use:", _DEFAULT_MODEL)

def get_llm(
    model: str = _DEFAULT_MODEL,
    temperature: float = 0.7,
    reasoning: bool | None = False,
):
    return ChatOllama(
        model=model,
        temperature=temperature,
        num_ctx=16384,
        reasoning=reasoning,
    )