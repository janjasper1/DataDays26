from datasets import load_dataset
from huggingface_hub import login
from dotenv import load_dotenv
import os
from pathlib import Path
import torch
import pandas as pd
from sklearn.preprocessing import StandardScaler
from backend.data_cleaning.cleaning_data_skript import clean_data
from backend.train.train_model import run_training_pipeline

load_dotenv()

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
y = clean_df['num_bicycles_available']

X = clean_df.drop(columns=['num_bicycles_available', 'station_id', 'name'])

### model training
X_model = X.apply(pd.to_numeric, errors='coerce').fillna(0.0)
y_model = pd.to_numeric(y, errors='coerce').fillna(0.0)

# Starte das Training mit der Pipeline
print("Starte Training...")
result = run_training_pipeline(clean_df, steps=1200, lr=0.002, visualize=True, hour=8.0)
print(f"Training abgeschlossen! Finaler Loss: {result['losses'][-1]:.2f}")

# Optional: Speichere das Modell
import torch
torch.save(result['model'].state_dict(), 'trained_model.pth')
print("Modell gespeichert als 'trained_model.pth'")


scaler = StandardScaler()
X_tensor = torch.tensor(scaler.fit_transform(X_model.values), dtype=torch.float32)
y_tensor = torch.tensor(y_model.values, dtype=torch.float32)

train_result = train_model(X_tensor, y_tensor, steps=1000, lr=0.005)
print(f"Training fertig. Final loss: {train_result['losses'][-1]:.2f}")

