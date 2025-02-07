https://colab.research.google.com/drive/1Xf_pppe18uXEOsfj6oxCdSb9lSGa_7wR#scrollTo=KjSwDXK8TAze

from google.colab import drive
drive.mount('/content/drive')

-------------------------------

# Install necessary libraries
!pip install huggingface_hub


-------------------------------

# Import necessary libraries
from huggingface_hub import login, HfApi
import os
import shutil

# Login to Hugging Face with your token
HF_TOKEN = "your hugging face token"
login(HF_TOKEN)

# Define model directory and repo name
model_dir = "/content/drive/MyDrive/models/fine_tuned_Llama-3.2-1B-Instruct"
repo_name = "ivansicaja/Fine_Tuned_Llama-3.2-1B-Instruct_ROS_PLC_Arduino_v_1.3.5"

# Initialize HfApi object
api = HfApi()

# Check if the repository exists
repo_exists = api.repo_info(repo_name)

if repo_exists:
    print(f"The repository {repo_name} already exists. Skipping creation.")
else:
    print(f"The repository {repo_name} does not exist. Creating it now.")
    # Create the repository if it does not exist
    api.create_repo(repo_name, private=True)  # Set private=False if you want the repo public

# Use shutil to copy the files from Google Drive to the local environment for easier upload
shutil.copytree(model_dir, "/content/fine_tuned_Llama")

# Navigate to the model directory
os.chdir("/content/fine_tuned_Llama")

# Upload the model to Hugging Face
from huggingface_hub import upload_folder

# Specify the repo_name to push to Hugging Face
upload_folder(
    folder_path=".",  # Path to the model directory
    repo_id=repo_name,  # Hugging Face repo name
    token=HF_TOKEN,  # Your Hugging Face token
    repo_type="model",  # You are uploading a model
)