https://colab.research.google.com/drive/1RkFTnsVXxXOCMzbFIYlj6K3qc5FLSRGr?usp=sharing


from google.colab import drive
drive.mount('/content/drive')

-------------------------------

# Install required libraries
!pip install transformers datasets pandas wandb -q

------------------------------

# Import required libraries
import pandas as pd
from datasets import Dataset, DatasetDict
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer, EarlyStoppingCallback, TrainerCallback
import wandb
from sklearn.model_selection import train_test_split
import torch

# Authenticate Hugging Face
from huggingface_hub import login

# Log in to Hugging Face with your token
HF_TOKEN = "your hugging face token"  # Replace with your Hugging Face token
login(HF_TOKEN)

# Authenticate Weights & Biases
wandb.login()

# Load the dataset
data_path = "/content/drive/MyDrive/DataForFineTuning.csv"
df = pd.read_csv(data_path)

# Check if the data is loaded correctly
print("Sample Data:")
print(df.head())

# Ensure the dataset has the correct columns
assert "question" in df.columns and "answer" in df.columns, "The CSV must have 'question' and 'answer' columns."

# Combine questions and answers into a single text column for fine-tuning
df["input_text"] = "Question: " + df["question"] + " Answer: " + df["answer"]

# Split the data into training and evaluation sets
train_texts, eval_texts = train_test_split(df["input_text"], test_size=0.2, random_state=42)

# Convert to Hugging Face DatasetDict
dataset = DatasetDict({
    "train": Dataset.from_dict({"input_text": train_texts}),
    "eval": Dataset.from_dict({"input_text": eval_texts}),
})

# Load the tokenizer and model
model_name = "meta-llama/Llama-3.2-1B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name, use_auth_token=HF_TOKEN)

# Add a padding token if not present
if tokenizer.pad_token is None:
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})

# Tokenize the dataset with reduced max_length for memory optimization
def tokenize_function(examples):
    encodings = tokenizer(examples["input_text"], truncation=True, padding="max_length", max_length=128)
    # Shift the input by one position to create labels for the model
    encodings["labels"] = encodings["input_ids"].copy()
    return encodings

tokenized_dataset = dataset.map(tokenize_function, batched=True)

# Load the model
model = AutoModelForCausalLM.from_pretrained(model_name, use_auth_token=HF_TOKEN)

# Resize the model embeddings if new tokens were added
model.resize_token_embeddings(len(tokenizer))

# Check if GPU is available
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)  # Move model to GPU if available

# Custom callback to log training and validation loss after every epoch
class LossLoggerCallback(TrainerCallback):
    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        if metrics and 'eval_loss' in metrics:
            print(f"Epoch {int(state.epoch)}\tTraining Loss: {metrics.get('train_loss', 'N/A')}\tValidation Loss: {metrics['eval_loss']}")


# Define training arguments with optimizations for faster training
training_args = TrainingArguments(
    output_dir="/content/fine_tuned_Llama-3.2-1B-Instruct",  # Directory to save the model
    evaluation_strategy="epoch",  # Evaluate at the end of each epoch
    save_strategy="epoch",  # Save model checkpoints at the end of each epoch
    learning_rate=2e-5,
    per_device_train_batch_size=4,  # Increased batch size to utilize more GPU memory
    per_device_eval_batch_size=4,   # Adjust batch size for evaluation
    gradient_accumulation_steps=2,  # Reduced accumulation steps for faster updates
    num_train_epochs=3,
    weight_decay=0.01,
    save_total_limit=2,  # Limit the number of saved checkpoints (optional)
    fp16=True,  # Use mixed precision for lower memory usage and faster training on GPU
    report_to="wandb",  # Enable WandB reporting
    logging_dir="./logs",
    logging_steps=10,
    lr_scheduler_type="linear",  # Linear learning rate decay
    push_to_hub=False,
    load_best_model_at_end=True,  # Ensure the best model is loaded
    metric_for_best_model="eval_loss",  # Use evaluation loss to determine the best model
    greater_is_better=False  # Lower loss is better
)


# Define the Trainer with EarlyStoppingCallback and LossLoggerCallback
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["eval"],
    tokenizer=tokenizer,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=1), LossLoggerCallback()],  # Add loss logger
)

# Fine-tune the model
trainer.train()

# Save the fine-tuned model dirrectly to google drive
model_dir = "/content/drive/MyDrive/models/fine_tuned_Llama-3.2-1B-Instruct"

# Save the model and tokenizer directly to Google Drive
model.save_pretrained(model_dir)
tokenizer.save_pretrained(model_dir)

print(f"Model saved to: {model_dir}")
