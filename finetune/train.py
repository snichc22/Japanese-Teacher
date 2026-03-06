from pathlib import Path

import torch.cuda
from datasets import load_dataset
from dotenv import load_dotenv
from transformers import AutoTokenizer, AutoConfig, BitsAndBytesConfig, Qwen3_5ForCausalLM
from peft import prepare_model_for_kbit_training, LoraConfig, get_peft_model
from trl import SFTTrainer, SFTConfig

load_dotenv(Path(__file__).resolve().parent.parent / ".env.teacher")

BASE_MODEL = "Qwen/Qwen3.5-9B-Base"
DATASET_PATH = "train_data.jsonl"
OUTPUT_DIR = "output"
LORA_OUTPUT = "lora_adapter"

MAX_SEQ_LENGTH = 2048
BATCH_SIZE = 4
GRADIENT_ACCUM = 8
EPOCHS = 3
LEARNING_RATE = 2e-4

def main():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True
    )

    config = AutoConfig.from_pretrained(BASE_MODEL, trust_remote_code=True)
    text_config = config.text_config

    model = Qwen3_5ForCausalLM.from_pretrained(
        BASE_MODEL,
        config=text_config,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.bfloat16
    )

    # Workaround: transformers 5.x stores _no_split_modules as a set, but
    # accelerate 1.13 chokes on it in get_balanced_memory(). Flatten to list.
    if hasattr(model, "_no_split_modules") and isinstance(model._no_split_modules, set):
        model._no_split_modules = list(model._no_split_modules)

    model = prepare_model_for_kbit_training(model)

    lora_config = LoraConfig(
        r=64,
        lora_alpha=128,
        target_modules=[
            "q_proj",       # Query attention head
            "k_proj",       # Key attention head
            "v_proj",       # Value attention head
            "o_proj",       # Output projection
            "gate_proj",    # MLP gate
            "up_proj",      # MLP up projection
            "down_proj",    # MLP down projection
        ],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    dataset = load_dataset("json", data_files=DATASET_PATH, split="train")
    print(f"Training examples: {len(dataset)}")

    training_args = SFTConfig(
        output_dir=OUTPUT_DIR,
        num_train_epochs=EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUM,
        learning_rate=LEARNING_RATE,
        weight_decay=0.01,
        warmup_steps=0.1,               # fraction of total steps used for LR warmup (transformers 5.x: float = ratio)
        lr_scheduler_type="cosine",
        logging_steps=10,
        save_steps=50,
        save_total_limit=3,
        bf16=True,                      # Use bfloat16 (RTX 30xx/40xx)
        optim="adamw_torch_fused",      # paged_adamw_8bit crashes on Windows (bitsandbytes CUDA init bug)
        gradient_checkpointing=True,
        max_grad_norm=0.3,
        report_to="none",               # "wandb" for logging
        max_length=MAX_SEQ_LENGTH,
    )

    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        args=training_args,
        train_dataset=dataset,
    )

    print(f"Starting fine-tuning...")
    trainer.train()

    model.save_pretrained(LORA_OUTPUT)
    tokenizer.save_pretrained(LORA_OUTPUT)
    print(f"Saved LoRA adapter to {LORA_OUTPUT}")
    print(f"Adapter size: {sum(f.stat().st_size for f in Path(LORA_OUTPUT).rglob('*')) / 1e6:.2f}MB")

if __name__ == "__main__":
    main()