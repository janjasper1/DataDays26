from datasets import load_dataset
from huggingface_hub import login

login(token="hf_YourTokenHere")  # Replace with your actual Hugging Face token

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

