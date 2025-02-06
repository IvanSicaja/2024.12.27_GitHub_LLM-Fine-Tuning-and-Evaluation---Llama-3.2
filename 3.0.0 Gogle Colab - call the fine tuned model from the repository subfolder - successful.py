import os
import spacy
import pandas as pd
import torch
from huggingface_hub import login
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM

# Configure Environment for Windows Symlink Compatibility
os.environ['HF_HUB_DISABLE_SYMLINKS_WARNING'] = '1'  # Disable symlink warnings for Windows

# Hugging Face Authentication
login("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")  # Replace with your Hugging Face token

# Load spaCy for text preprocessing
nlp = spacy.load("en_core_web_sm")

# Define repository and subfolder path
repo_name = "ivansicaja/Fine_Tuned_Llama-3.2-1B-Instruct_ROS_PLC_Arduino"
subfolder = "v_1.0.0/model_files"

# Load tokenizer from the specified subfolder
tokenizer = AutoTokenizer.from_pretrained(repo_name, token="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", subfolder=subfolder)

# Check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the fine-tuned model from the specified subfolder
model = AutoModelForCausalLM.from_pretrained(repo_name, token="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", subfolder=subfolder).to(device)

# Set up the text generation pipeline
generator = pipeline("text-generation", model=model, tokenizer=tokenizer, pad_token_id=tokenizer.eos_token_id, device=0 if device == torch.device("cuda") else -1)

# Function to preprocess text
def preprocess_text(text):
    doc = nlp(text)
    tokens = [token.lemma_.lower() for token in doc if not token.is_stop and not token.is_punct]
    return " ".join(tokens)

# Function to generate response based on query input
def generate_response(query):
    processed_query = preprocess_text(query)
    contextual_input = f"User query: {processed_query}\nAnswer:"

    response = generator(
        contextual_input,
        max_new_tokens=512,
        num_return_sequences=1,
        truncation=True,
        do_sample=True,
        temperature=0.7
    )

    generated_text = response[0]["generated_text"]
    if "Answer:" in generated_text:
        generated_text = generated_text.split("Answer:")[-1].strip()
    else:
        generated_text = generated_text.strip()
    return generated_text

# Main loop for chatbot interaction
if __name__ == "__main__":
    print("Chatbot ready! Type your questions below.")

    while True:
        print("-----------------------------------------------------------------")
        user_query = input("You: ")
        chatbot_response = generate_response(user_query)
        print("[CHATBOT]:", chatbot_response)
