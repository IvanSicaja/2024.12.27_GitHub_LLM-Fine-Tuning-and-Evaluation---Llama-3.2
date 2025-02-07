import gc
import torch

# Clear GPU memory cache
torch.cuda.empty_cache()

# Run garbage collection to free up system memory
gc.collect()

----------------------

# Install required libraries
!pip
install
torch
transformers
huggingface_hub
nltk
rouge - score
bert_score

---------------------
import torch
import math
import nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer
from transformers import AutoModelForCausalLM, AutoTokenizer
from huggingface_hub import login
from nltk.translate.meteor_score import meteor_score
from bert_score import score as bert_score
import transformers

# Suppress Hugging Face warnings
transformers.logging.set_verbosity_error()

# Download necessary NLTK data
nltk.download('punkt_tab')
nltk.download('wordnet')

# Log in with your Hugging Face token (Replace with your actual token)
login("your hugging face token")  # Replace with your token

# Select device (use GPU if available)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def clean_generated_text(question, generated_text):
    """Removes the question from the generated text."""
    if generated_text.lower().startswith(question.lower()):
        return generated_text[len(question):].strip()
    return generated_text

def compute_perplexity(model, tokenizer, text):
    encodings = tokenizer(text, return_tensors="pt")
    input_ids = encodings.input_ids.to(device)

    with torch.no_grad():
        outputs = model(input_ids, labels=input_ids)
        loss = outputs.loss
        perplexity = math.exp(loss.item())

    return perplexity

def compute_bleu(reference, candidate):
    reference_tokens = nltk.word_tokenize(reference.lower())
    candidate_tokens = nltk.word_tokenize(candidate.lower())
    smoothie = SmoothingFunction().method1
    return sentence_bleu([reference_tokens], candidate_tokens, smoothing_function=smoothie)

def compute_rouge(reference, candidate):
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    scores = scorer.score(reference, candidate)
    return {
        "ROUGE-1": scores["rouge1"].fmeasure,
        "ROUGE-2": scores["rouge2"].fmeasure,
        "ROUGE-L": scores["rougeL"].fmeasure
    }

def compute_meteor(reference, candidate):
    reference_tokens = nltk.word_tokenize(reference.lower())
    candidate_tokens = nltk.word_tokenize(candidate.lower())
    return meteor_score([reference_tokens], candidate_tokens)

def compute_bert_score(reference, candidate):
    P, R, F1 = bert_score([candidate], [reference], lang="en")
    return F1.mean().item()

example_text = [
    ("What is a PLC and how is it used in automation?", "A PLC (Programmable Logic Controller) is a digital computer used for automation of industrial processes, such as control of machinery or factory assembly lines."),
    ("How can Arduino be used for automation projects?", "Arduino is an open-source microcontroller platform used for building electronics projects, including automation systems, by interfacing with sensors, actuators, and other hardware."),
    ("What is ROS and why is it important in robotics?", "ROS (Robot Operating System) is an open-source framework that provides tools and libraries for building robotic applications, enabling hardware abstraction, device control, and communication between different components."),
    ("How does ROS2 differ from ROS1?", "ROS2 improves upon ROS1 by offering real-time capabilities, better security, and support for multi-robot systems. It uses DDS for communication, making it more scalable and robust for industrial applications."),
    ("What is the role of a ROS node in a robotic system?", "A ROS node is a process that performs computation and communicates with other nodes in a ROS-based robotic system. Nodes publish and subscribe to topics to exchange data, facilitating modular and distributed system design."),
    ("How can ROS be used for SLAM?", "ROS provides packages like Gmapping, Cartographer, and RTAB-Map to implement SLAM (Simultaneous Localization and Mapping), enabling robots to map unknown environments while tracking their own position.")
]


def evaluate_model(model_name):
    print(f"\nLoading model: {model_name}...\n")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name).to(device)

    total_ppl, total_bleu, total_meteor, total_bert = 0, 0, 0, 0
    total_rouge1, total_rouge2, total_rougeL = 0, 0, 0
    num_samples = len(example_text)

    for i, (question, answer) in enumerate(example_text):
        inputs = tokenizer(question, return_tensors="pt").to(device)
        output_ids = model.generate(
            **inputs,
            max_new_tokens=512,  # Increase token limit
            num_return_sequences=1,
            do_sample=True,
            temperature=0.7,  # Increase diversity
            top_p=0.9,  # Nucleus sampling
            top_k=50,  # Prevent repetition
            repetition_penalty=1.2,  # Penalize repetitive phrases
            pad_token_id=tokenizer.eos_token_id,
            early_stopping=True  # Stop at a reasonable point
        )
        generated_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
        generated_answer = clean_generated_text(question, generated_text)

        ppl = compute_perplexity(model, tokenizer, generated_answer)
        bleu = compute_bleu(answer, generated_answer)
        rouge = compute_rouge(answer, generated_answer)
        meteor = compute_meteor(answer, generated_answer)
        bert = compute_bert_score(answer, generated_answer)

        total_ppl += ppl
        total_bleu += bleu
        total_rouge1 += rouge["ROUGE-1"]
        total_rouge2 += rouge["ROUGE-2"]
        total_rougeL += rouge["ROUGE-L"]
        total_meteor += meteor
        total_bert += bert

        print(f"Q{i+1}: {question}\n")
        print(f"Expected Answer: {answer}\n")
        print(f"Generated Answer: {generated_answer}\n")
        print(f"Perplexity = {ppl:.2f}, BLEU = {bleu:.2f}, ROUGE-1 = {rouge['ROUGE-1']:.2f}, ROUGE-2 = {rouge['ROUGE-2']:.2f}, ROUGE-L = {rouge['ROUGE-L']:.2f}, METEOR = {meteor:.2f}, BERTScore = {bert:.2f}\n")
        print("-------------------\n")

    avg_ppl = total_ppl / num_samples
    avg_bleu = total_bleu / num_samples
    avg_rouge1 = total_rouge1 / num_samples
    avg_rouge2 = total_rouge2 / num_samples
    avg_rougeL = total_rougeL / num_samples
    avg_meteor = total_meteor / num_samples
    avg_bert = total_bert / num_samples

    print(f"\n==== AVERAGE EVALUATION METRICS ({model_name}) ====\n")
    print(f"Avg Perplexity: {avg_ppl:.2f}")
    print(f"Avg BLEU: {avg_bleu:.2f}")
    print(f"Avg ROUGE-1: {avg_rouge1:.2f}")
    print(f"Avg ROUGE-2: {avg_rouge2:.2f}")
    print(f"Avg ROUGE-L: {avg_rougeL:.2f}")
    print(f"Avg METEOR: {avg_meteor:.2f}")
    print(f"Avg BERTScore: {avg_bert:.2f}")
    print("==========================================\n")


    ----------------------

    # Evaluate both models
    evaluate_model("meta-llama/Llama-3.2-1B-Instruct")
    evaluate_model("ivansicaja/Fine_Tuned_Llama-3.2-1B-Instruct_ROS_PLC_Arduino_v_1.3.5")