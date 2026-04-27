import yaml
import json
from tqdm import tqdm
from unsloth import FastLanguageModel
from unsloth.chat_templates import get_chat_template

class IntentClassification:
    def __init__(self, model_path):
        # According to requirements, model_path points to a config file
        with open(model_path, 'r') as file:
            config = yaml.safe_load(file)
            
        checkpoint_dir = config.get("model_checkpoint", "../lora_model")
        max_seq_length = config.get("max_seq_length", 2048)
        self.test_file_path = config.get("test_file_path", "../sample_data/test_unsloth.jsonl")
        
        print(f"Loading fine-tuned model from '{checkpoint_dir}'...")
        self.model, self.tokenizer = FastLanguageModel.from_pretrained(
            model_name = checkpoint_dir,
            max_seq_length = max_seq_length,
            dtype = None,
            load_in_4bit = True,
        )
        
        # Enable Unsloth's 2x faster inference mode
        FastLanguageModel.for_inference(self.model)
        
        # Apply the same chat template used during training
        self.tokenizer = get_chat_template(self.tokenizer, chat_template="llama-3.1")

    def __call__(self, message):
        messages = [
            {"role": "user", "content": f"What is the intent of this customer query?\n\nQuery: {message}"}
        ]
        
        inputs = self.tokenizer.apply_chat_template(
            messages,
            tokenize = True,
            add_generation_prompt = True,
            return_tensors = "pt",
        ).to("cuda")
        
        outputs = self.model.generate(input_ids = inputs, max_new_tokens = 64, use_cache = True)
        
        predicted_label = self.tokenizer.decode(outputs[0][inputs.shape[1]:], skip_special_tokens=True)
        return predicted_label.strip()

# ==========================================
# Usage Example & Test Set Evaluation
# ==========================================
def evaluate_test_set(classifier, test_file):
    print(f"\n--- Evaluating Test Set Accuracy ---")
    correct = 0
    total = 0

    with open(test_file, 'r') as f:
        lines = f.readlines()

    # NOTE: To keep the video short (2-5 mins), evaluate a sample of the test set (e.g., first 50)
    # Change `lines[:50]` to `lines` to evaluate the whole set
    for line in tqdm(lines[:50], desc="Evaluating"):
        data = json.loads(line)

        # Extract the raw text and true intent from the Unsloth JSONL format
        human_prompt = data["conversations"][0]["value"]
        raw_text = human_prompt.replace("What is the intent of this customer query?\n\nQuery: ", "")
        true_intent = data["conversations"][1]["value"]

        predicted_intent = classifier(raw_text)

        # Compare prediction to the actual label
        if predicted_intent.lower() == true_intent.lower():
            correct += 1
        total += 1

    accuracy = (correct / total) * 100
    print(f"\nFinal Test Set Accuracy: {accuracy:.2f}% (Tested on {total} samples)")


if __name__ == "__main__":
    config_file_path = "configs/inference.yaml" 
    
    try:
        classifier = IntentClassification(config_file_path)
        
        print("\n=== Single Input Test ===")
        test_message = "I lost my credit card, how do I get a new one?"
        print(f"Input Message: {test_message}")
        prediction = classifier(test_message)
        print(f"Predicted Intent: {prediction}\n")
        
        # Run test set evaluation for the video demo
        evaluate_test_set(classifier, classifier.test_file_path)
        
    except FileNotFoundError:
        print(f"Please ensure the config file exists at {config_file_path}")