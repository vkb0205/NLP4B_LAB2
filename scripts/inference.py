import yaml
from unsloth import FastLanguageModel
from unsloth.chat_templates import get_chat_template

class IntentClassification:
    def __init__(self, model_path):
        # According to requirements, model_path points to a config file
        with open(model_path, 'r') as file:
            config = yaml.safe_load(file)
            
        # Extract the actual checkpoint directory from the config
        checkpoint_dir = config.get("model_checkpoint", "lora_model")
        max_seq_length = config.get("max_seq_length", 2048)
        
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
        # Format the query exactly how the model was trained
        messages = [
            {"role": "user", "content": f"What is the intent of this customer query?\n\nQuery: {message}"}
        ]
        
        inputs = self.tokenizer.apply_chat_template(
            messages,
            tokenize = True,
            add_generation_prompt = True,
            return_tensors = "pt",
        ).to("cuda")
        
        # Generate the prediction
        outputs = self.model.generate(input_ids = inputs, max_new_tokens = 64, use_cache = True)
        
        # Decode only the newly generated text (the predicted intent)
        predicted_label = self.tokenizer.decode(outputs[0][inputs.shape[1]:], skip_special_tokens=True)
        return predicted_label.strip()

# ==========================================
# Short usage example required by the prompt
# ==========================================
if __name__ == "__main__":
    # Ensure you have a configs/inference.yaml file created before running this
    config_file_path = "../configs/inference.yaml" 
    
    try:
        # Initialize the classifier
        classifier = IntentClassification(config_file_path)
        
        # Test a single input message
        test_message = "I lost my credit card, how do I get a new one?"
        print(f"\nInput Message: {test_message}")
        
        # Call the instance directly to get the prediction
        prediction = classifier(test_message)
        print(f"Predicted Intent: {prediction}")
        
    except FileNotFoundError:
        print(f"Please create the configuration file at {config_file_path} to run this example.")