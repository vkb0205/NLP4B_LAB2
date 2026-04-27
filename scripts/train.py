import argparse
import yaml
import torch
from datasets import load_dataset
from unsloth import FastLanguageModel
from unsloth.chat_templates import get_chat_template, standardize_sharegpt
from trl import SFTConfig, SFTTrainer
from transformers import DataCollatorForSeq2Seq

def parse_args():
    parser = argparse.ArgumentParser(description="Train Unsloth Model on Banking77")
    parser.add_argument("--config", type=str, default="../configs/train.yaml", help="Path to training config file")
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Load parameters from the YAML config file
    with open(args.config, 'r') as file:
        config = yaml.safe_load(file)

    max_seq_length = config.get("max_seq_length", 2048)
    dtype = None # Auto detection

    print(f"1. Loading Model: {config['model_name']}...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = config["model_name"],
        max_seq_length = max_seq_length,
        dtype = dtype, 
        load_in_4bit = True,
    )

    model = FastLanguageModel.get_peft_model(
        model,
        r = config.get("lora_r", 16), 
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                          "gate_proj", "up_proj", "down_proj"],
        lora_alpha = config.get("lora_alpha", 16),
        lora_dropout = 0, 
        bias = "none", 
        use_gradient_checkpointing = "unsloth", 
        random_state = 3407,
        use_rslora = False, 
        loftq_config = None, 
    )

    print("2. Formatting Dataset...")
    tokenizer = get_chat_template(tokenizer, chat_template="llama-3.1")

    def formatting_prompts_func(examples):
        convos = examples["conversations"]
        texts = [tokenizer.apply_chat_template(convo, tokenize=False, add_generation_prompt=False) for convo in convos]
        return { "text" : texts }

    # Load Train Data
    train_dataset = load_dataset("json", data_files=config["data_file"], split="train")
    train_dataset = standardize_sharegpt(train_dataset)
    train_dataset = train_dataset.map(formatting_prompts_func, batched=True)

    # Load Validation Data
    val_dataset = load_dataset("json", data_files=config["val_data_file"], split="train")
    val_dataset = standardize_sharegpt(val_dataset)
    val_dataset = val_dataset.map(formatting_prompts_func, batched=True)

    print("3. Initializing Trainer...")
    trainer = SFTTrainer(
        model = model,
        tokenizer = tokenizer,
        train_dataset = train_dataset,
        eval_dataset = val_dataset,          # Passed the validation dataset here
        dataset_text_field = "text",
        max_seq_length = max_seq_length,
        data_collator = DataCollatorForSeq2Seq(tokenizer = tokenizer),
        packing = False, 
        args = SFTConfig(
            per_device_train_batch_size = config["per_device_train_batch_size"],
            gradient_accumulation_steps = config["gradient_accumulation_steps"],
            warmup_steps = config["warmup_steps"],
            num_train_epochs = config["num_train_epochs"],
            learning_rate = float(config["learning_rate"]),
            logging_steps = 1,
            optim = config["optimizer"],
            weight_decay = config["weight_decay"],
            lr_scheduler_type = config["lr_scheduler_type"],
            evaluation_strategy = config.get("evaluation_strategy", "epoch"), # Added eval strategy
            seed = 3407,
            output_dir = config["output_dir"],
            report_to = "none", 
        ),
    )

    print("4. Starting Training...")
    trainer.train()

    print("5. Saving Model...")
    model.save_pretrained(config["save_model_dir"])
    tokenizer.save_pretrained(config["save_model_dir"])
    print("Done!")

if __name__ == "__main__":
    main()