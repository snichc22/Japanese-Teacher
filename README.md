# Japanese Teacher AI

A bilingual (Japanese/English) AI language teacher powered by a locally-hosted LLM via [Ollama](https://ollama.com/), with RAG document search and internet lookup capabilities. Built with LangChain, Gradio, and ChromaDB.

**Base Model:** `qwen3.5:9B`

---

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Quick Start](#quick-start)
3. [Environment Variables](#environment-variables)
4. [Step-by-Step Setup](#step-by-step-setup)
5. [Running the App](#running-the-app)
6. [Launch from Windows Search](#launch-from-windows-search)
7. [Fine-Tuning (Optional)](#fine-tuning-optional)
8. [Project Structure](#project-structure)
9. [My Specs](#my-specs)

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

> **Note on the URL:** The app binds to `0.0.0.0` (all interfaces) but you must open **`http://localhost:7860`** in your browser. 
> `http://0.0.0.0:7860` is not a valid browser URL on Windows.

---

## Environment Variables

Configuration is loaded from `.env.teacher` in the project root. Copy `.envTemplate` to `.env.teacher` and fill in your values.

| Variable | Required | Default | Description |
|---|---|---|---|
| `HF_TOKEN` | No | *(empty)* | HuggingFace API token for downloading models during fine-tuning. Get one at [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens). |
| `OLLAMA_MODEL` | No | `japanese-teacher` | Ollama model name used by the agent. Must support tool-calling. Both `japanese-teacher` and `japanese-teacher-ft` are valid after running the full pipeline. |
| `DOCUMENT_DIR` | No | `documents` | Folder where uploaded PDFs / `.txt` / `.md` learning materials are stored for RAG. |
| `VECTORSTORE_DIR` | No | `vectorstore` | Folder where the ChromaDB vector-store index is persisted. |
| `EMBEDDING_DEVICE` | No | *(auto)* | Device for HuggingFace embeddings. `cuda` forces GPU, `cpu` forces CPU. Leave blank to auto-detect. |

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

## Launch from Windows Search

Create a Start Menu shortcut once, then launch the app by typing its name in Windows Search.

```powershell
.\scripts\create_windows_search_shortcut.ps1
```

Optional: customize the search name shown in Windows Search.

```powershell
.\scripts\create_windows_search_shortcut.ps1 -ShortcutName "Japanese Teacher AI"
```

By default, the script uses `icon.png` from the project root and converts it to `icon.ico` for the Windows shortcut.

```powershell
.\scripts\create_windows_search_shortcut.ps1 -IconPath ".\icon.png"
```

Attribution: `image: Flaticon.com`. This icon uses is from Flaticon.com.

After creating the shortcut:
1. Press `Win`
2. Type `Japanese Teacher` (or your custom name)
3. Launch the result

This creates a `.lnk` file in:

`%APPDATA%\Microsoft\Windows\Start Menu\Programs`

---

## Fine-Tuning (Optional)

If you want to fine-tune the base model on custom training data, follow these steps **in order**. This requires a CUDA-capable GPU with sufficient VRAM.

### 1. Prepare Training Data

```powershell
cd finetune
python prepare_data.py
```

This generates `train_data.jsonl` with example teacher–student conversations. 

Edit [`finetune/prepare_data.py`](finetune/prepare_data.py) to add more training examples.

### 2. Train the LoRA Adapter

```powershell
python train.py
```

This will:
- Download `Qwen/Qwen3.5-9B-Instruct` tokenizer (for its chat template) and `Qwen/Qwen3.5-9B-Base` weights
- Fine-tune using QLoRA (4-bit quantization + LoRA)
- Save the LoRA adapter to `finetune/lora_adapter/`
- Save the full instruct chat template to `finetune/lora_adapter/chat_template.jinja`
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
2. **Inject the Qwen3.5-Instruct chat template** into the merged model's tokenizer config — this is what enables tool-calling, thinking mode, and vision support in the fine-tuned model
3. Convert the merged model to GGUF format → `finetune/merged_model.gguf`
4. Generate `Modelfile.finetuned`
5. Create an Ollama model named `japanese-teacher-ft`

### 4. Use the Fine-Tuned Model

The fine-tuned model is selected via the `OLLAMA_MODEL` variable in `.env.teacher` (defaults to `japanese-teacher-ft`). No code changes needed.

Run it directly with Ollama to test:

```powershell
ollama run japanese-teacher-ft
```

---

## My Specs

- **GPU**: NVIDIA RTX 4080 Laptop
- **RAM**: 32GB

---

## Performance

