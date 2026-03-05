# Japanese Teacher AI

A bilingual (Japanese/English) AI language teacher powered by a locally-hosted LLM via [Ollama](https://ollama.com/), with RAG document search and internet lookup capabilities. Built with LangChain, Gradio, and ChromaDB.

**Base Model:** `qwen3.5:9B`

---

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Quick Start](#quick-start)
3. [Step-by-Step Setup](#step-by-step-setup)
4. [Running the App](#running-the-app)
5. [Fine-Tuning (Optional)](#fine-tuning-optional)
6. [Project Structure](#project-structure)
7. [My Specs](#my-specs)

---

## Prerequisites

- **Python 3.10+**
- **NVIDIA GPU** with CUDA support (recommended; CPU works but is slower)

---

## Quick Start

```powershell
# 1. Clone the repo and cd into it
cd "F:\Personal Projects\Japanese Teacher"

# 2. Set up the Python virtual environment & install dependencies
.\scripts\setup_env.ps1

# 3. Install Ollama, pull the base model, and create the japanese-teacher model
.\scripts\setup.ps1

# 4. Run the app (starts Ollama if needed, then launches Gradio)
.\scripts\run.ps1
```

Then open **http://localhost:7860** in your browser.

---

## Step-by-Step Setup

### 1. Create a Python Virtual Environment

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

> **Note:** If you get an execution policy error, run this first:
> ```powershell
> Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
> ```

This installs all Python dependencies including LangChain, Gradio, ChromaDB, PyTorch (with CUDA 12.6 support), sentence-transformers, and more.

### 2. Install Ollama & Pull the Base Model

```powershell
# Install Ollama (Windows)
irm https://ollama.com/install.ps1 | iex

# Pull the base Qwen 3.5 9B model
ollama pull qwen3.5:9b
```

### 3. Create the Custom Ollama Model

From the project root directory:

```powershell
ollama create japanese-teacher -f Modelfile
```

This creates a custom Ollama model named `japanese-teacher` using the system prompt defined in the [`Modelfile`](Modelfile).

---

## Running the App

### Option A: Use the run script (recommended)

```powershell
.\scripts\run.ps1
```

This script will:
1. Check if Ollama is running and start it if it's not
2. Activate the virtual environment
3. Launch the Gradio app on `http://localhost:7860`

### Option B: Run manually

```powershell
# Terminal 1 — Start Ollama (if not already running)
ollama serve

# Terminal 2 — Activate venv and start the app
.\.venv\Scripts\Activate.ps1
python app.py
```

---

## Fine-Tuning (Optional)

If you want to fine-tune the base model on custom training data, follow these steps **in order**. This requires a CUDA-capable GPU with sufficient VRAM.

### 1. Prepare Training Data

```powershell
cd finetune
python prepare_data.py
```

This generates `train_data.jsonl` with example teacher–student conversations. Edit [`finetune/prepare_data.py`](finetune/prepare_data.py) to add more training examples (200–500+ recommended).

### 2. Train the LoRA Adapter

```powershell
python train.py
```

This will:
- Download the `Qwen/Qwen3.5-9B-Base` model from HuggingFace
- Fine-tune it using QLoRA (4-bit quantization + LoRA)
- Save the LoRA adapter to `finetune/lora_adapter/`
- Save checkpoints to `finetune/output/`

**Training config:** batch size 4, gradient accumulation 8, 3 epochs, cosine LR schedule, bfloat16.

### 3. Merge & Convert to Ollama

Before running this step, ensure you have `llama.cpp` cloned in the project root:

```powershell
cd ..
git clone https://github.com/ggml-org/llama.cpp
pip install gguf
cd finetune
```

Then run the conversion:

```powershell
python convert_to_ollama.py
```

This will:
1. Merge the LoRA adapter with the base model → `finetune/merged_model/`
2. Convert the merged model to GGUF format → `finetune/merged_model.gguf`
3. Generate `Modelfile.finetuned`
4. Create an Ollama model named `japanese-teacher-ft`

### 4. Use the Fine-Tuned Model

To use the fine-tuned model instead of the base model, update the model name in [`src/llm.py`](src/llm.py):

```python
def get_llm(model: str = "japanese-teacher-ft", temperature: float = 0.7):
```

Or run it directly with Ollama:

```powershell
ollama run japanese-teacher-ft
```

---

## My Specs

- **GPU**: NVIDIA RTX 4080 Laptop
- **RAM**: 32GB