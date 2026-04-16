import argparse
from datasets import load_dataset
import pandas as pd
from sklearn.model_selection import train_test_split
from pathlib import Path
import json

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample-size", type=int, default=4000)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--output-dir", type=str, default="sample_data")
    return parser.parse_args()

def main():
    args = parse_args()
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    dataset = load_dataset("banking77")
    train_df = pd.DataFrame(dataset["train"])
    test_df = pd.DataFrame(dataset["test"])
    
    combined = pd.concat([train_df, test_df], ignore_index=True)
    
    # Map numeric labels back to strings
    label_names = dataset["train"].features["label"].names
    combined["intent"] = combined["label"].apply(lambda x: label_names[x])

    if args.sample_size < len(combined):
        combined = combined.sample(n=args.sample_size, random_state=42).reset_index(drop=True)

    # Create Unsloth ShareGPT format (List of Dicts)
    def create_conversation(row):
        return [
            {"from": "human", "value": f"What is the intent of this customer query?\n\nQuery: {row['text']}"},
            {"from": "gpt", "value": row['intent']}
        ]

    combined["conversations"] = combined.apply(create_conversation, axis=1)

    train_data, test_data = train_test_split(combined, test_size=args.test_size, random_state=42)

    # Save as JSONL (JSON Lines is much better for nested lists than CSV)
    train_data[["conversations"]].to_json(output_path / "train_unsloth.jsonl", orient="records", lines=True)
    test_data[["conversations"]].to_json(output_path / "test_unsloth.jsonl", orient="records", lines=True)
    
    print(f"Saved {len(train_data)} train and {len(test_data)} test rows.")

if __name__ == "__main__":
    main()