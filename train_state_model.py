import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from state_model import CropStateModel


DATA = "data/model/model_daily.csv"

device = (
    "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)

print("Using device:", device)


# --------------------------------------------------
# Load
# --------------------------------------------------

df = pd.read_csv(DATA, parse_dates=["date"])
df = df.sort_values("date").reset_index(drop=True)


# --------------------------------------------------
# Construct observation-informed state
# --------------------------------------------------

df["state_0"] = (
    df[["NDVI_z", "EVI_z", "FPAR_z", "NDWI_z"]]
    .mean(axis=1)
)

df["state_1"] = (
    df[["VV_z", "VH_z", "VV_VH_z"]]
    .mean(axis=1)
)

df["state_2"] = (
    df["NDWI_z"] - df["temperature_C_z"]
)

df["state_3"] = df["NDVI_z"].diff()


state_cols = [
    "state_0",
    "state_1",
    "state_2",
    "state_3"
]

forcing_cols = [
    "temperature_C_z",
    "precipitation_mm_z",
    "solar_radiation_MJ_z"
]


# --------------------------------------------------
# Select only actual satellite observations
# --------------------------------------------------

# We require both S1 and S2 at the beginning
# and end of a transition.
#
# This avoids fabricating observations on cloudy/
# satellite-missing days.

df["complete_obs"] = (
    df["NDVI"].notna()
    & df["VV"].notna()
)


obs = df[df["complete_obs"]].copy()

print("Complete multimodal observations:", len(obs))


# --------------------------------------------------
# Build observation-to-observation transitions
# --------------------------------------------------

X_state = []
X_force = []
Y_state = []


for i in range(len(obs) - 1):

    current = obs.iloc[i]
    future = obs.iloc[i + 1]

    # Current state
    x = current[state_cols].values.astype(float)

    # Future state
    y = future[state_cols].values.astype(float)

    # Weather forcing averaged across the interval
    current_force = current[forcing_cols].values.astype(float)
    future_force = future[forcing_cols].values.astype(float)

    forcing = (
        current_force + future_force
    ) / 2.0

    if (
        np.isfinite(x).all()
        and np.isfinite(y).all()
        and np.isfinite(forcing).all()
    ):
        X_state.append(x)
        X_force.append(forcing)
        Y_state.append(y)


X_state = np.asarray(X_state)
X_force = np.asarray(X_force)
Y_state = np.asarray(Y_state)


print("Training transitions:", len(X_state))


if len(X_state) == 0:
    raise RuntimeError(
        "No valid observation-to-observation transitions were found."
    )


# --------------------------------------------------
# PyTorch tensors
# --------------------------------------------------

X_state = torch.tensor(
    X_state,
    dtype=torch.float32
)

X_force = torch.tensor(
    X_force,
    dtype=torch.float32
)

Y_state = torch.tensor(
    Y_state,
    dtype=torch.float32
)


dataset = TensorDataset(
    X_state,
    X_force,
    Y_state
)

loader = DataLoader(
    dataset,
    batch_size=min(32, len(dataset)),
    shuffle=True
)


# --------------------------------------------------
# Model
# --------------------------------------------------

model = CropStateModel().to(device)

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=1e-3
)

loss_fn = nn.MSELoss()


# --------------------------------------------------
# Training
# --------------------------------------------------

epochs = 200

for epoch in range(epochs):

    total_loss = 0.0

    for state, forcing, target in loader:

        state = state.to(device)
        forcing = forcing.to(device)
        target = target.to(device)

        prediction = model(
            state,
            forcing
        )

        loss = loss_fn(
            prediction,
            target
        )

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    if (epoch + 1) % 20 == 0:

        avg_loss = total_loss / len(loader)

        print(
            f"Epoch {epoch + 1:03d} "
            f"| Loss: {avg_loss:.6f}"
        )


# --------------------------------------------------
# Save
# --------------------------------------------------

torch.save(
    model.state_dict(),
    "state_model.pt"
)

print("\nTraining complete.")
print("Saved: state_model.pt")
