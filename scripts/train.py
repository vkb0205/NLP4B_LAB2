import argparse
import torch
from datasets import load_dataset
from unsloth import FastLanguageModel
from unsloth.chat_templates import get_chat_template, standardize_sharegpt
from trl import SFTConfig, SFTTrainer
from transformers import DataCollatorForSeq2Seq

def parse_args():
    parser = argparse.ArgumentParser(description="Train Unsloth Model on Banking77")
    parser.add_argument("--data-file", type=str, default="sample_data/train_unsloth.jsonl")
    parser.add_argument("--model-name", type=str, default="unsloth/Llama-3.2-1B-Instruct")
    parser.add_argument("--output-dir", type=str, default="outputs")
    parser.add_argument("--save-model-dir", type=str, default="lora_model")
    return parser.parse_args()

def main():
    args = parse_args()
    max_seq_length = 2048
    dtype = Float16 # None for auto detection. Float16 for Tesla T4, V100, Bfloat16 for Ampere+

    print("1. Loading Model...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = args.model_name,
        max_seq_length = max_seq_length,
        dtype = dtype, # None for auto detection. Float16 for Tesla T4, V100, Bfloat16 for Ampere+
        load_in_4bit = True,
    )

    model = FastLanguageModel.get_peft_model(
        model,
        r = 16, # Choose any number > 0 ! Suggested 8, 16, 32, 64, 128
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj",],
        lora_alpha = 16,
        lora_dropout = 0, # Supports any, but = 0 is optimized
        bias = "none",    # Supports any, but = "none" is optimized
        # [NEW] "unsloth" uses 30% less VRAM, fits 2x larger batch sizes!
        use_gradient_checkpointing = "unsloth", # True or "unsloth" for very long context
        random_state = 3407,
        use_rslora = False,  # We support rank stabilized LoRA
        loftq_config = None, # And LoftQ
    )

    print("2. Formatting Dataset...")
    tokenizer = get_chat_template(tokenizer, chat_template="llama-3.1")

    def formatting_prompts_func(examples):
        convos = examples["conversations"]
        texts = [tokenizer.apply_chat_template(convo, tokenize=False, add_generation_prompt=False) for convo in convos]
        return { "text" : texts }

    dataset = load_dataset("json", data_files=args.data_file, split="train")
    dataset = standardize_sharegpt(dataset)
    dataset = dataset.map(formatting_prompts_func, batched=True)

    print("3. Initializing Trainer...")
    # --- YOUR PROVIDED TRAINER BLOCK ---
    trainer = SFTTrainer(
        model = model,
        tokenizer = tokenizer,
        train_dataset = dataset,
        dataset_text_field = "text",
        max_seq_length = max_seq_length,
        data_collator = DataCollatorForSeq2Seq(tokenizer = tokenizer),
        packing = False, # Can make training 5x faster for short sequences.
        args = SFTConfig(
            per_device_train_batch_size = 2,
            gradient_accumulation_steps = 4,
            warmup_steps = 5,
            num_train_epochs = 1, # Set this for 1 full training run.
            max_steps = 60,
            learning_rate = 2e-4,
            logging_steps = 1,
            optim = "adamw_8bit",
            weight_decay = 0.001,
            lr_scheduler_type = "linear",
            seed = 3407,
            output_dir = args.output_dir,
            report_to = "none", # Use TrackIO/WandB etc
        ),
    )
    # -----------------------------------

    print("4. Starting Training...")
    trainer.train()

    print("5. Saving Model...")
    model.save_pretrained(args.save_model_dir)
    tokenizer.save_pretrained(args.save_model_dir)
    print("Done!")

if __name__ == "__main__":
    main()
