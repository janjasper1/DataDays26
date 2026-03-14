from datasets import load_dataset
from huggingface_hub import login
from dotenv import load_dotenv
import os
from pathlib import Path
from backend.data_cleaning.cleaning_data_skript import clean_data

main_dir = Path(__file__).resolve().parent
env_candidates = [main_dir / ".env", main_dir.parent / ".env"]

for env_path in env_candidates:
    if env_path.exists():
        load_dotenv(env_path, override=True)
        break

hf_token = os.getenv("HF_TOKEN")
if not hf_token:
    checked = ", ".join(str(p) for p in env_candidates)
    raise EnvironmentError(f"HF_TOKEN nicht gefunden. Gepruefte .env-Pfade: {checked}")
login(token=hf_token.strip())
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

if 'train' not in dataset:
    raise KeyError("Dataset enthaelt keinen 'train'-Split.")

train_df = dataset['train'].to_pandas()
clean_df = clean_data(train_df)

print(clean_df)

### input / output for ml training


### model training
