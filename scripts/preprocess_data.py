from datasets import load_dataset
import pandas as pd

def main():
    # Load the BANKING77 dataset from Hugging Face
    dataset = load_dataset("banking77")
    
    # Convert to pandas DataFrames for easier manipulation
    train_df = pd.DataFrame(dataset['train'])
    test_df = pd.DataFrame(dataset['test'])
    
    # Sample a manageable subset for fine-tuning
    # (e.g., taking 10% of the training data)
    sampled_train = train_df.sample(frac=0.1, random_state=42)
    
    # Save the sampled data to the sample_data directory
    sampled_train.to_csv('../sample_data/train_sample.csv', index=False)
    print("Dataset downloaded and sampled successfully!")

if __name__ == "__main__":
    main()
