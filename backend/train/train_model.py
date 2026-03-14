import torch
import torch.nn as nn
import pyro
import pyro.distributions as dist
from pyro.infer import SVI, Trace_ELBO
from pyro.infer.autoguide import AutoNormal
from pyro.optim import Adam
from pyro.nn import PyroModule, PyroSample
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from typing import Dict, Any



def identify_temporal_key(df: pd.DataFrame) -> str:
    candidates = ['timestamp', 'datetime', 'date', 'time', 'created_at', 'updated_at']
    for candidate in candidates:
        for col in df.columns:
            if candidate in str(col).lower():
                if pd.to_datetime(df[col].dropna().head(1), errors='coerce').notna().all():
                    return col
    raise ValueError("No valid temporal vector found.")


def extract_spatial_coordinates(df: pd.DataFrame) -> pd.DataFrame:
    if {'lon', 'lat'}.issubset(df.columns):
        return df
    if {'Laengengrad', 'Breitengrad'}.issubset(df.columns):
        df['lon'] = pd.to_numeric(df['Laengengrad'], errors='coerce')
        df['lat'] = pd.to_numeric(df['Breitengrad'], errors='coerce')
        return df

    geom_candidates = ['location', 'geometry', 'point', 'coordinates']
    found_geom = next((col for col in df.columns if any(c in col.lower() for c in geom_candidates)), None)
    if found_geom:
        coords = df[found_geom].astype(str).str.extract(r'\(([^ ]+) ([^ ]+)\)')
        df['lon'] = coords[0].astype(float)
        df['lat'] = coords[1].astype(float)
        return df
    raise ValueError("Spatial dimensions undefined.")


def identify_target_key(df: pd.DataFrame) -> str:
    candidates = ['bikes', 'available', 'count', 'demand', 'target']
    valid_domain = [col for col in df.select_dtypes(include=['number']).columns if col not in ['lat', 'lon']]
    for candidate in candidates:
        for col in valid_domain:
            if candidate in str(col).lower(): return col
    return valid_domain[0]


def prepare_training_data(raw_df: pd.DataFrame) -> Dict[str, Any]:
    df = raw_df.copy()

    time_col = identify_temporal_key(df)
    df['timestamp'] = pd.to_datetime(df[time_col], errors='coerce')
    df = df.dropna(subset=['timestamp']).copy()
    df = df.sort_values('timestamp').reset_index(drop=True)

    df = extract_spatial_coordinates(df)
    target_key = identify_target_key(df)

    # Sort strictly by space and time to guarantee valid differential operations
    df = df.sort_values(['lat', 'lon', 'timestamp']).reset_index(drop=True)

    df['delta_Y'] = df.groupby(['lat', 'lon'])[target_key].shift(1) - df[target_key]
    df['delta_Y'] = df['delta_Y'].clip(lower=0).fillna(0)

    # Project discrete temporal coordinates onto the continuous S1 quotient space (tau = 168)
    df['Stunde'] = df['timestamp'].dt.hour
    df['Wochentag'] = df['timestamp'].dt.dayofweek
    hours_into_week = df['Wochentag'] * 24 + df['Stunde']
    df['h_sin'] = np.sin(2 * np.pi * hours_into_week / 168)
    df['h_cos'] = np.cos(2 * np.pi * hours_into_week / 168)

    # Boolean mapping injection
    mapping = {'True': 1, 'False': 0, 'true': 1, 'false': 0, 'T': 1, 'F': 0, 't': 1, 'f': 0, '1': 1, '0': 0, True: 1, False: 0, 1: 1, 0: 0}
    if 'is_virtual_station' in df.columns:
        df['is_virtual_station'] = df['is_virtual_station'].map(mapping).astype(float).fillna(0)
    else:
        df['is_virtual_station'] = 0.0
    if 'realtime_data_outdated' in df.columns:
        df['realtime_data_outdated'] = df['realtime_data_outdated'].map(mapping).astype(float).fillna(0)
    else:
        df['realtime_data_outdated'] = 0.0

    # Construct the design matrix X and response vector Y
    features = ['h_sin', 'h_cos', 'lat', 'lon', 'capacity', 'is_virtual_station', 'realtime_data_outdated']
    # Drop NaN values created by the shift operator and clean continuous time
    df = df.dropna(subset=['delta_Y'] + features)

    x_unscaled = df[features].values.astype(float)
    scaler = StandardScaler()
    x_tensor = torch.tensor(scaler.fit_transform(x_unscaled), dtype=torch.float)
    y_tensor = torch.tensor(df['delta_Y'].values, dtype=torch.float)
    return {
        'df': df,
        'X': x_tensor,
        'Y': y_tensor,
        'scaler': scaler,
        'target_key': target_key,
    }


# --- 3. High-Capacity Model (128-64 Architecture) ---
class DeepNBGLM(PyroModule):
    def __init__(self, input_dim):
        super().__init__()
        # Layers expanded to allow localized curvature and break parallel artifacts
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
        # Exponential link function, clamped to enforce bounded intensity gradients
        mu = torch.exp(torch.clamp(self.fc3(h).squeeze(), max=20.0))
        r = pyro.sample("r", dist.HalfNormal(10.0))

        with pyro.plate("data", x.shape[0]):
            pyro.sample("obs", dist.NegativeBinomial(total_count=r, probs=r / (r + mu)), obs=y)


def train_model(x: torch.Tensor, y: torch.Tensor, steps: int = 1200, lr: float = 0.002,
                log_every: int = 200) -> Dict[str, Any]:
    pyro.clear_param_store()
    model = DeepNBGLM(x.shape[1])
    guide = AutoNormal(model)
    svi = SVI(model, guide, Adam({"lr": lr}), loss=Trace_ELBO())

    losses = []
    print(f"Optimizing manifold with support size N={x.shape[0]}...")
    for step in range(steps):
        loss = svi.step(x, y)
        losses.append(loss)
        if step % log_every == 0:
            print(f"Step {step} | Loss: {loss:.2f}")

    return {
        'model': model,
        'guide': guide,
        'losses': losses,
    }


def visualize_results(df: pd.DataFrame, scaler: StandardScaler, hour: float = 8.0) -> None:
    import matplotlib.pyplot as plt
    import contextily as ctx

    stations = df[['lon', 'lat']].drop_duplicates()
    h_sin = np.sin(2 * np.pi * hour / 24) * np.ones(len(stations))
    h_cos = np.cos(2 * np.pi * hour / 24) * np.ones(len(stations))

    X_syn = np.column_stack([h_sin, h_cos, stations['lon'].values, stations['lat'].values])
    X_tensor = torch.tensor(scaler.transform(X_syn), dtype=torch.float32, requires_grad=True)

    # Extract MAP estimates
    w1, b1 = pyro.param("AutoNormal.locs.fc1.weight").detach(), pyro.param("AutoNormal.locs.fc1.bias").detach()
    w2, b2 = pyro.param("AutoNormal.locs.fc2.weight").detach(), pyro.param("AutoNormal.locs.fc2.bias").detach()
    w3, b3 = pyro.param("AutoNormal.locs.fc3.weight").detach(), pyro.param("AutoNormal.locs.fc3.bias").detach()

    h1 = torch.relu(torch.matmul(X_tensor, w1.t()) + b1)
    h2 = torch.relu(torch.matmul(h1, w2.t()) + b2)
    mu = torch.exp(torch.matmul(h2, w3.t()) + b3).squeeze()

    mu.sum().backward()
    grads = X_tensor.grad.numpy()
    g_lon = grads[:, 2] / scaler.scale_[2]
    g_lat = grads[:, 3] / scaler.scale_[3]

    fig, ax = plt.subplots(figsize=(12, 10))
    sc = ax.scatter(stations['lon'], stations['lat'], c=mu.detach().numpy(),
                    cmap='viridis', s=100, alpha=0.8, edgecolors='white', zorder=3)
    plt.colorbar(sc, label=r'Demand Intensity $\mu$')

    ax.quiver(stations['lon'].values, stations['lat'].values, g_lon, g_lat,
              color='white', scale=50, width=0.003, alpha=1.0, zorder=4)

    ctx.add_basemap(ax, crs='EPSG:4326', source=ctx.providers.CartoDB.Positron)
    ax.set_title(f"Mannheim Station Demand Flow at {hour}:00")
    plt.show()


def run_training_pipeline(raw_df: pd.DataFrame, steps: int = 1200, lr: float = 0.002,
                          visualize: bool = False, hour: float = 8.0) -> Dict[str, Any]:
    data = prepare_training_data(raw_df)
    trained = train_model(data['X'], data['Y'], steps=steps, lr=lr)

    if visualize:
        visualize_results(data['df'], data['scaler'], hour=hour)

    return {
        **data,
        **trained,
    }
