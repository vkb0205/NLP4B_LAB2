import argparse
import re
from datasets import load_dataset
import pandas as pd
from sklearn.model_selection import train_test_split
from pathlib import Path
import json

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample-size", type=int, default=4000)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--output-dir", type=str, default="../sample_data")
    return parser.parse_args()

def clean_text(text):
    """
    Perform basic text cleaning and normalization as required by the assignment.
    """
    if not isinstance(text, str):
        return ""
    # Lowercase the text
    text = text.lower()
    # Remove special characters but keep alphanumeric and spaces
    text = re.sub(r'[^a-z0-9\s\.\?\,\']', '', text)
    # Remove extra spaces (collapse multiple spaces into one)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def main():
    args = parse_args()
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print("Loading BANKING77 dataset...")
    dataset = load_dataset("banking77")
    train_df = pd.DataFrame(dataset["train"])
    test_df = pd.DataFrame(dataset["test"])

    combined = pd.concat([train_df, test_df], ignore_index=True)

    # Map numeric labels back to strings
    label_names = dataset["train"].features["label"].names
    combined["intent"] = combined["label"].apply(lambda x: label_names[x])

    print("Applying text normalization and cleaning...")
    combined["text"] = combined["text"].apply(clean_text)

    if args.sample_size < len(combined):
        print(f"Sampling {args.sample_size} rows...")
        combined = combined.sample(n=args.sample_size, random_state=42).reset_index(drop=True)

    def create_conversation(row):
        return [
            {"from": "human", "value": f"What is the intent of this customer query?\n\nQuery: {row['text']}"},
            {"from": "gpt", "value": row['intent']}
        ]

    combined["conversations"] = combined.apply(create_conversation, axis=1)

    # 1. First split: Separate Test data
    train_temp, test_data = train_test_split(combined, test_size=args.test_size, random_state=42)
    
    # 2. Second split: Extract Validation data from the remaining Training data (e.g., 10%)
    train_data, val_data = train_test_split(train_temp, test_size=0.1, random_state=42)

    print("Saving datasets...")
    # Save as JSONL for Unsloth
    train_data[["conversations"]].to_json(output_path / "train_unsloth.jsonl", orient="records", lines=True)
    val_data[["conversations"]].to_json(output_path / "val_unsloth.jsonl", orient="records", lines=True)
    test_data[["conversations"]].to_json(output_path / "test_unsloth.jsonl", orient="records", lines=True)
    
    # Save standard CSVs for EDA (Optional but helpful)
    train_data[["text", "label", "intent"]].rename(columns={"label": "label_id"}).to_csv(output_path / "train.csv", index=False)
    val_data[["text", "label", "intent"]].rename(columns={"label": "label_id"}).to_csv(output_path / "val.csv", index=False)
    test_data[["text", "label", "intent"]].rename(columns={"label": "label_id"}).to_csv(output_path / "test.csv", index=False)

    print(f"Saved {len(train_data)} train, {len(val_data)} val, and {len(test_data)} test rows to {args.output_dir}.")

if __name__ == "__main__":
    main()