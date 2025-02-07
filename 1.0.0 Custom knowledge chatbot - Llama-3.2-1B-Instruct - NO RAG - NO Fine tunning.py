import os
import spacy
import pandas as pd
import torch
from huggingface_hub import login
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
import keyboard  # To detect escape key press

# --------------------------------------------------------------- MODEL DEFINITION --------------------------------------------------------------------------------------
# --- Configure Environment for Windows Symlink Compatibility ---
os.environ['HF_HUB_DISABLE_SYMLINKS_WARNING'] = '1'  # Disable symlink warnings for Windows

# --- Hugging Face Authentication ---
login("your hugging face token")  # Replace with your Hugging Face token

# --- Setup NLP and LLaMA Model ---
# Load spaCy for text preprocessing
nlp = spacy.load("en_core_web_sm")

# Load LLaMA model and tokenizer from Hugging Face
llama_model_name = "meta-llama/Llama-3.2-1B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(llama_model_name, token="your hugging face token")

# Check if GPU is available and load the model on GPU if possible
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = AutoModelForCausalLM.from_pretrained(llama_model_name, token="your hugging face token").to(device)

# Set up the text generation pipeline with LLaMA model on the specified device
generator = pipeline("text-generation", model=model, tokenizer=tokenizer, pad_token_id=tokenizer.eos_token_id, device=0 if device == torch.device("cuda") else -1)

# --------------------------------------------------------------- RAG KNOWLEDGE BASE --------------------------------------------------------------------------------------
# --- Load Knowledge Base ---
csv_path = "1.0 Datasets/DataForFineTunning.csv"  # Path to your CSV file
df = pd.read_csv(csv_path)

# --------------------------------------------------------------- FUNCTIONS DEFINITIONS --------------------------------------------------------------------------------------
# Step 1: Define function to preprocess text
def preprocess_text(text):
    doc = nlp(text)
    tokens = [token.lemma_.lower() for token in doc if not token.is_stop and not token.is_punct]
    return " ".join(tokens)

# Step 2: Define function to generate response based on query input (no retrieval step)
def generate_response(query):
    # Preprocess user query
    processed_query = preprocess_text(query)

    # Create context input (without conversation history)
    contextual_input = f"User query: {processed_query}\nAnswer:"

    print("Contextual Input to Generator:", contextual_input)  # Debugging print

    # Generate the response with optimized parameters
    response = generator(
        contextual_input,
        max_new_tokens=512,
        num_return_sequences=1,
        truncation=True,
        do_sample=True,  # Disabling sampling for faster, deterministic output
        temperature=0.7  # Lower temperature for faster response
    )

    # Capture generated answer after "Answer:" prompt
    generated_text = response[0]["generated_text"]
    if "Answer:" in generated_text:
        generated_text = generated_text.split("Answer:")[-1].strip()
    else:
        generated_text = generated_text.strip()

    return generated_text

# --------------------------------------------------------------- MAIN LOOP FOR ASKING QUESTIONS --------------------------------------------------------------------------------------
if __name__ == "__main__":
    print("Chatbot ready! Type your questions. Press 'Esc' to exit.")
    while True:
        if keyboard.is_pressed("esc"):
            print("\nExiting chatbot. Goodbye!")
            break

        print("-----------------------------------------------------------------")
        user_query = input("You: ")
        print("User Query:", user_query)
        chatbot_response = generate_response(user_query)
        print("[FINAL CHATBOT ANSWER]:", chatbot_response)
