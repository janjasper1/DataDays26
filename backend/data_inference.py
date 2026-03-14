import pickle
import torch
import torch.nn as nn
import pyro
import pyro.distributions as dist
from pyro.infer import Predictive
from pyro.infer.autoguide import AutoNormal
from pyro.nn import PyroModule, PyroSample
import pandas as pd
import numpy as np
from datasets import load_dataset
from huggingface_hub import login
from dotenv import load_dotenv
import os
from pathlib import Path
import json
from shapely.wkt import loads

# --- 1. Load Environment and Authenticate ---
main_dir = Path(__file__).resolve().parent
env_candidates = [main_dir / ".env", main_dir.parent / ".env"]
for env_path in env_candidates:
    if env_path.exists():
        load_dotenv(env_path, override=True)
        break

hf_token = os.getenv("HF_TOKEN")
if hf_token:
    login(token=hf_token.strip())

# --- 2. DeepNBGLM Model Class (Copied from train_model.py) ---
class DeepNBGLM(PyroModule):
    def __init__(self, input_dim):
        super().__init__()
        self.fc1 = PyroModule[nn.Linear](input_dim, 128)
        self.fc1.weight = PyroSample(dist.Normal(0., 1.).expand([128, input_dim]).to_event(2))
        self.fc1.bias = PyroSample(dist.Normal(0., 1.).expand([128]).to_event(1))

        self.fc2 = PyroModule[nn.Linear](128, 64)
        self.fc2.weight = PyroSample(dist.Normal(0., 1.).expand([64, 128]).to_event(2))
        self.fc2.bias = PyroSample(dist.Normal(0., 1.).expand([64]).to_event(1))

        self.fc3 = PyroModule[nn.Linear](64, 1)
        self.fc3.weight = PyroSample(dist.Normal(0., 1.).expand([1, 64]).to_event(2))
        self.fc3.bias = PyroSample(dist.Normal(0., 1.).expand([1]).to_event(1))

        self.relu = nn.ReLU()

    def forward(self, x, y=None):
        h = self.relu(self.fc1(x))
        h = self.relu(self.fc2(h))
        mu = torch.exp(torch.clamp(self.fc3(h).squeeze(), max=20.0))
        r = pyro.sample("r", dist.HalfNormal(10.0))

        with pyro.plate("data", x.shape[0]):
            pyro.sample("obs", dist.NegativeBinomial(total_count=r, probs=r/(r+mu)), obs=y)


FEATURES = [
    "h_sin",
    "h_cos",
    "lat",
    "lon",
    "capacity",
    "is_virtual_station",
    "realtime_data_outdated",
]


def preprocess_data(df: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray]:
    processed = df.copy()

    if "geometry" in processed.columns:
        processed["geometry"] = processed["geometry"].apply(
            lambda value: loads(value) if isinstance(value, str) else value
        )
        processed["lat"] = processed["geometry"].apply(
            lambda geometry: geometry.y if geometry is not None else np.nan
        )
        processed["lon"] = processed["geometry"].apply(
            lambda geometry: geometry.x if geometry is not None else np.nan
        )
    elif {"Breitengrad", "Laengengrad"}.issubset(processed.columns):
        processed["lat"] = pd.to_numeric(processed["Breitengrad"], errors="coerce")
        processed["lon"] = pd.to_numeric(processed["Laengengrad"], errors="coerce")
    elif {"lat", "lon"}.issubset(processed.columns):
        processed["lat"] = pd.to_numeric(processed["lat"], errors="coerce")
        processed["lon"] = pd.to_numeric(processed["lon"], errors="coerce")
    else:
        raise ValueError("No spatial columns available for prediction.")

    time_column = "last_reported" if "last_reported" in processed.columns else "timestamp"
    processed["timestamp"] = pd.to_datetime(processed[time_column], errors="coerce")
    processed["Stunde"] = processed["timestamp"].dt.hour
    processed["Wochentag"] = processed["timestamp"].dt.dayofweek
    hours_into_week = processed["Wochentag"] * 24 + processed["Stunde"]
    processed["h_sin"] = np.sin(2 * np.pi * hours_into_week / 168)
    processed["h_cos"] = np.cos(2 * np.pi * hours_into_week / 168)

    mapping = {
        "True": 1,
        "False": 0,
        "true": 1,
        "false": 0,
        "T": 1,
        "F": 0,
        "t": 1,
        "f": 0,
        "1": 1,
        "0": 0,
        True: 1,
        False: 0,
        1: 1,
        0: 0,
    }
    if "is_virtual_station" in processed.columns:
        processed["is_virtual_station"] = processed["is_virtual_station"].map(mapping).fillna(0.0).astype(float)
    else:
        processed["is_virtual_station"] = 0.0
    if "realtime_data_outdated" in processed.columns:
        processed["realtime_data_outdated"] = processed["realtime_data_outdated"].map(mapping).fillna(0.0).astype(float)
    else:
        processed["realtime_data_outdated"] = 0.0

    processed["capacity"] = pd.to_numeric(processed.get("capacity", 0.0), errors="coerce").fillna(0.0)
    processed = processed.dropna(subset=["timestamp", "lat", "lon"] + FEATURES).copy()

    return processed, processed[FEATURES].to_numpy(dtype=float)

# --- 3. Load Trained Model (Mock for now, since file doesn't exist) ---
model_path = main_dir / "bayesian_demand_model.pkl"
if model_path.exists():
    with open(model_path, "rb") as f:
        manifold_state = pickle.load(f)
    scaler_mean = np.asarray(manifold_state["scaler_mean"], dtype=float)
    scaler_scale = np.asarray(manifold_state["scaler_scale"], dtype=float)
    pyro_param_store_state = manifold_state["pyro_param_store"]
    model_state_dict = manifold_state.get("model_state_dict")
    guide_state_dict = manifold_state.get("guide_state_dict")
    print("Model loaded successfully.")
else:
    scaler_mean = None
    scaler_scale = None
    pyro_param_store_state = {}
    model_state_dict = None
    guide_state_dict = None
    print("Model file not found. Using fallback scaler derived from train split.")

# --- 4. Load and Preprocess Data ---
print("Loading dataset...")
train_df = load_dataset("Amaan/DataDays", split="train").to_pandas()
test_df = load_dataset("Amaan/DataDays", split="test").to_pandas()

train_processed, train_features = preprocess_data(train_df)
test_processed, test_features = preprocess_data(test_df)

if scaler_mean is None or scaler_scale is None:
    scaler_mean = train_features.mean(axis=0)
    scaler_scale = train_features.std(axis=0)
    scaler_scale = np.where(scaler_scale == 0, 1.0, scaler_scale)

X_test_scaled = (test_features - scaler_mean) / scaler_scale
X_test = torch.tensor(X_test_scaled, dtype=torch.float32)

# Restore trained variational state after guide creation.
pyro.clear_param_store()
model = DeepNBGLM(len(FEATURES))
guide = AutoNormal(model)
if model_state_dict is not None:
    model.load_state_dict(model_state_dict, strict=False)
if guide_state_dict is not None:
    guide.load_state_dict(guide_state_dict, strict=False)
if pyro_param_store_state:
    pyro.get_param_store().set_state(pyro_param_store_state)

# --- 5. Generate Predictions ---
print("Generating predictions...")
predictive = Predictive(model, guide=guide, num_samples=100)
predictions = predictive(X_test)

# Extract mean prediction (expected demand change)
demand_predictions = predictions['obs'].mean(dim=0).detach().numpy()

# --- 6. Export to JSON ---
dashboard_data = []
for i, row in test_processed.reset_index(drop=True).iterrows():
    dashboard_data.append({
        "lat": row['lat'],
        "lon": row['lon'],
        "name": row.get('name', row.get('name_left', 'Unknown')),
        "demand_prediction": float(demand_predictions[i]),
        "timestamp": row['timestamp'].isoformat() if pd.notna(row['timestamp']) else None
    })

output_path = main_dir / "dashboard_data.json"
with open(output_path, "w") as f:
    json.dump(dashboard_data, f, indent=4)

print(f"Predictions exported to {output_path}")