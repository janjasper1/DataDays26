from datasets import load_dataset
from huggingface_hub import login
from dotenv import load_dotenv
import os

load_dotenv(Path(__file__).parents[2] / ".env" if "__file__" in dir() else Path("../../.env"))

hf_token = os.getenv("HF_TOKEN")
if not hf_token:
    raise EnvironmentError("HF_TOKEN nicht gefunden – bitte in der .env-Datei setzen.")
login(token=hf_token)
# 1. Download/Load the dataset
print("Fetching Amaan/DataDays...")
dataset = load_dataset("Amaan/DataDays")

# 2. Look at what's inside
print("\n--- Dataset Info ---")
print(dataset)


# 3. Access a specific row (e.g., first row of the 'train' split)
# Note: Check the output of step 2 to see if your data uses 'train'
if 'train' in dataset:
    print("\n--- First Entry Sample ---")
    print(dataset['train'][0])

