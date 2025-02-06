from google.colab import drive
drive.mount('/content/drive')

-------------------------------

# Install necessary libraries
!pip install huggingface_hub

-------------------------------

# Log in to Hugging Face
from huggingface_hub import login

# Use your Hugging Face token here
HF_TOKEN = "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"  # Replace with your actual Hugging Face token
login(HF_TOKEN)

# Define the model directory on Google Drive
model_dir = "/content/drive/MyDrive/models/fine_tuned_Llama-3.2-1B-Instruct"  # Path to your fine-tuned model

# Define the Hugging Face repository where you want to upload the model
repo_name = "ivansicaja/Fine_Tuned_Llama-3.2-1B-Instruct_ROS_PLC_Arduino"

# Initialize Hugging Face API
from huggingface_hub import HfApi
api = HfApi()

# Clone the Hugging Face repository to a temporary directory (Colab)
from huggingface_hub import Repository

# Define a temporary directory for cloning the repo
temp_dir = "/content/fine_tuned_model_repo"
repo = Repository(local_dir=temp_dir, clone_from=repo_name)

# No need to create the repository, since it already exists

-------------------------------

import shutil
import os

# Ensure the 'model_files' folder doesn't already exist
model_files_dir = f"{temp_dir}/model_files"
if os.path.exists(model_files_dir):
    shutil.rmtree(model_files_dir)  # Remove the existing folder if it exists

# Copy model files from your Google Drive to the cloned repository directory
shutil.copytree(model_dir, model_files_dir)

# Change to the cloned repository directory
os.chdir(temp_dir)

# Set up Git user name and email (required by Git for committing)
!git config --global user.email "ivansicaja.poslovno@gmail.com"  # Replace with your email
!git config --global user.name "IvanSicaja"  # Replace with your GitHub username

# Create a version directory to tag the model with a version
version = "v_1.0.0"

# Ensure the version directory does not already exist, and remove if necessary
version_dir = os.path.join(temp_dir, version)
if os.path.exists(version_dir):
    shutil.rmtree(version_dir)  # Remove the existing folder if it exists

# Create the version directory (this will also remove any pre-existing folder)
os.makedirs(version_dir)

# Move the model files into the version folder
shutil.move(model_files_dir, version_dir)

# Change to the version directory
os.chdir(version_dir)

# Push the model to Hugging Face with version in the commit message
repo.push_to_hub(commit_message=f"Initial commit of Fine-Tuned Llama-3.2-1B-Instruct model v_1.0.0")

print(f"Model successfully uploaded to: https://huggingface.co/{repo_name}")
