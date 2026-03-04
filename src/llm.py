from langchain_ollama import ChatOllama

def get_llm(model: str = "japanese-teacher", temperature: float = 0.7):
    return ChatOllama(
        model=model,
        temperature=temperature,
        num_ctx=16384
    )