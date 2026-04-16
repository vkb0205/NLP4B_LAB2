import argparse
import pandas as pd
from datasets import Dataset
from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification, 
    TrainingArguments, 
    Trainer
)
from sklearn.metrics import accuracy_score
from pathlib import Path
import torch

def parse_args():
    parser = argparse.ArgumentParser(description="Train Intent Detection Model")
    parser.add_argument("--data-dir", type=str, default="sample_data", help="Directory containing processed CSVs")
    parser.add_argument("--model-name", type=str, default="distilbert-base-uncased", help="Base model to fine-tune")
    parser.add_argument("--output-dir", type=str, default="models/intent_model", help="Where to save the model")
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size for training and evaluation")
    return parser.parse_args()

def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    acc = accuracy_score(labels, preds)
    return {"accuracy": acc}

def main():
    args = parse_args()
    data_path = Path(args.data_dir)
    
    # 1. Load the label mapping to configure the model
    label_map_df = pd.read_csv(data_path / "label_mapping.csv")
    label2id = dict(zip(label_map_df["label"], label_map_df["label_id"]))
    id2label = {id: label for label, id in label2id.items()}
    num_labels = len(label2id)

    # 2. Load the processed CSV files into Hugging Face Datasets
    train_df = pd.read_csv(data_path / "train.csv")
    test_df = pd.read_csv(data_path / "test.csv")
    
    train_dataset = Dataset.from_pandas(train_df)
    test_dataset = Dataset.from_pandas(test_df)

    # 3. Load Tokenizer and Model
    print(f"Loading model: {args.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name,
        num_labels=num_labels,
        id2label=id2label,
        label2id=label2id
    )

    # 4. Tokenize the text data
    def tokenize_function(examples):
        return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=128)

    print("Tokenizing datasets...")
    tokenized_train = train_dataset.map(tokenize_function, batched=True)
    tokenized_test = test_dataset.map(tokenize_function, batched=True)

    # Rename label_id to label so the Trainer understands it
    tokenized_train = tokenized_train.rename_column("label_id", "labels")
    tokenized_test = tokenized_test.rename_column("label_id", "labels")

    # 5. Set up Training Arguments
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        logging_dir="./logs",
        logging_steps=50,
        fp16=torch.cuda.is_available(), # Uses Colab T4's mixed precision for faster training
    )

    # 6. Initialize Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_test,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )

    # 7. Train and Save
    print("Starting training...")
    trainer.train()
    
    print(f"Saving final model to {args.output_dir}")
    trainer.save_model(args.output_dir)
    print("Training complete!")

if __name__ == "__main__":
    main()