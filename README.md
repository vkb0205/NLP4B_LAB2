# Project 2: Fine-Tuning Intent Detection Model with BANKING77 Dataset
## 🎥 Video Demonstration
**[Link to Video Demonstration on Google Drive]**  
---

## 📂 Repository Structure
The project follows the required directory structure:
```text
banking-intent-unsloth/
|-- scripts/
|   |-- train.py                 # Fine-tuning script using Unsloth
|   |-- inference.py             # OOP Inference class and evaluation
|   |-- preprocess_data.py       # Data downloading, cleaning, and formatting
|   |-- eda.py                 # Exploratory Data Analysis (Optional)
|
|-- configs/
|   |-- train.yaml               # Training hyperparameters configuration
|   |-- inference.yaml           # Inference configurations
|
|-- sample_data/                 # Generated datasets (train, val, test)
|-- train.sh                     # Bash script to run preprocessing and training
|-- inference.sh                 # Bash script to test the trained model
|-- requirements.txt             # Python dependencies
|-- README.md                    # Project documentation
```

---

## ⚙️ Environment Setup & Installation

To set up the environment and run this project, it is highly recommended to use **Google Colab (T4 GPU)** or a local Linux machine with an NVIDIA GPU.

**1. Clone the repository:**
```bash
git clone https://github.com/[YOUR_USERNAME]/NLP4B_LAB2.git
cd NLP4B_LAB2
```

**2. Install required dependencies:**
Unsloth requires specific versions of PyTorch, xformers, and trl. Run the following commands:
```bash
pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
pip install --no-deps "xformers<0.0.27" "trl<0.9.0" peft accelerate bitsandbytes
pip install -r requirements.txt
```

---

## 🚀 How to Run the Model

### 1. Download Data, Preprocess, and Train
You can execute the entire pipeline (data downloading, text normalization, dataset splitting, and model fine-tuning) using a single bash script:
```bash
chmod +x train.sh
bash train.sh
```
*Note: This script calls `scripts/preprocess_data.py` to prepare the data, followed by `scripts/train.py` using configurations from `configs/train.yaml`. The fine-tuned model will be saved in the `saved_model` directory.*

### 2. Run Inference and Evaluate
After training is complete, you can test the model's prediction on a custom input and evaluate its accuracy on the test set:
```bash
chmod +x inference.sh
bash inference.sh
```
*Note: This script executes `scripts/inference.py` and loads settings from `configs/inference.yaml`.*

---

## 📊 Hyperparameters and Configuration

All hyperparameters are properly extracted into `configs/train.yaml` to avoid hardcoding. Below is the detailed explanation of the configurations used for fine-tuning the model:

### Base Model
* **Model:** `unsloth/Llama-3.2-1B-Instruct`
* **Maximum Sequence Length:** `2048` (Optimal for capturing short-to-medium banking queries efficiently without Out-Of-Memory errors).

### Training Hyperparameters
* **Batch Size:** `2` (per device)
* **Gradient Accumulation Steps:** `4` (Yields an effective batch size of 8, balancing memory usage and stable gradient updates).
* **Learning Rate:** `2e-4` (`0.0002`) - A standard and stable learning rate for LoRA fine-tuning.
* **Optimizer:** `adamw_8bit` (Provides memory-efficient optimization, allowing the model to train smoothly on a 16GB T4 GPU).
* **Number of Epochs:** `3` (Sufficient for the model to converge on the subset data without severe overfitting).
* **Warmup Steps:** `5` (Gradually increases the learning rate at the beginning of training to prevent sudden gradient spikes).
* **LR Scheduler Type:** `linear`

### Regularization and LoRA Techniques
* **Weight Decay:** `0.001` (Applies slight L2 regularization to prevent weights from growing too large).
* **LoRA Rank (r):** `16` (Determines the dimensionality of the low-rank updates. 16 provides a great balance between trainable parameter count and model performance).

### Data Preprocessing & Augmentation
* **Text Normalization:** Applied via `clean_text` in `preprocess_data.py`. The text is lowercased, non-alphanumeric characters (except basic punctuation) are stripped, and excessive whitespace is collapsed to remove noise.
* **Data Formatting:** Converted into a ShareGPT-style sequence classification format, explicitly prompting the model: *"What is the intent of this customer query? Query: {text}"*.
* **Validation Split:** A 10% validation split was created from the training data to evaluate learning progress (`eval_strategy: "epoch"`).