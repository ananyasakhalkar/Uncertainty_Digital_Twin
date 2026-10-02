import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from pathlib import Path


# =========================================================
# CONFIG
# =========================================================

BASE = Path.home() / "Downloads" / "Crop_Digital_Twin"

DATA_FILE = BASE / "GLOBAL_ALL25_MODEL_READY.csv"
MODEL_DIR = BASE / "models"
MODEL_DIR.mkdir(exist_ok=True)

MODEL_FILE = MODEL_DIR / "GLOBAL_5CROP_TEMPORAL_TWIN.pt"

DEVICE = torch.device(
    "mps" if torch.backends.mps.is_available() else "cpu"
)

print("DEVICE:", DEVICE)


# =========================================================
# LOAD DATA
# =========================================================

df = pd.read_csv(DATA_FILE)
df["date"] = pd.to_datetime(df["date"])

df = df.sort_values(
    ["site_id", "date"]
).reset_index(drop=True)


# =========================================================
# FEATURES
# =========================================================

state_features = [
    "NDVI",
    "EVI",
    "NDWI",
    "VV",
    "VH",
    "VV_VH"
]

weather_features = [
    "temperature_C",
    "precipitation_mm",
    "solar_MJ_m2"
]


# =========================================================
# TEMPORAL HOLDOUT
# =========================================================

train_df = df[
    (df["date"] >= "2019-01-01") &
    (df["date"] <= "2023-12-01")
].copy()

test_df = df[
    (df["date"] >= "2024-01-01") &
    (df["date"] <= "2024-10-01")
].copy()


print("\n========================================")
print("GLOBAL 5-CROP TEMPORAL TWIN")
print("========================================")

print("\nTRAIN ROWS:", len(train_df))
print("TEST ROWS:", len(test_df))

print("\nTRAIN SITES:", train_df["site_id"].nunique())
print("TRAIN CROPS:", train_df["crop"].nunique())


# =========================================================
# STANDARDIZATION
# FIT ONLY ON TRAINING DATA
# =========================================================

state_mean = train_df[state_features].mean()
state_std = train_df[state_features].std().replace(0, 1)

weather_mean = train_df[weather_features].mean()
weather_std = train_df[weather_features].std().replace(0, 1)


def standardize(data):
    data = data.copy()

    for c in state_features:
        data[c] = (
            data[c] - state_mean[c]
        ) / state_std[c]

    for c in weather_features:
        data[c] = (
            data[c] - weather_mean[c]
        ) / weather_std[c]

    return data


train_df = standardize(train_df)
test_df = standardize(test_df)


# =========================================================
# CROP ENCODING
# =========================================================

crop_names = sorted(
    train_df["crop"].dropna().unique()
)

crop_to_id = {
    crop: i
    for i, crop in enumerate(crop_names)
}

print("\nCROP ENCODING:")
print(crop_to_id)


# =========================================================
# MODEL
# =========================================================

class CropTwin(nn.Module):

    def __init__(
        self,
        state_dim,
        weather_dim,
        crop_dim,
        hidden_dim=32
    ):
        super().__init__()

        self.crop_embedding = nn.Embedding(
            crop_dim,
            4
        )

        self.transition = nn.Sequential(

            nn.Linear(
                state_dim +
                weather_dim +
                4,
                hidden_dim
            ),

            nn.Tanh(),

            nn.Linear(
                hidden_dim,
                hidden_dim
            ),

            nn.Tanh(),

            nn.Linear(
                hidden_dim,
                state_dim
            )
        )

    def forward(
        self,
        state,
        weather,
        crop
    ):

        crop_emb = self.crop_embedding(crop)

        x = torch.cat(
            [
                state,
                weather,
                crop_emb
            ],
            dim=-1
        )

        delta = self.transition(x)

        # Small learned state update
        delta = 0.10 * torch.tanh(delta)

        next_state = state + delta

        return next_state


model = CropTwin(
    state_dim=len(state_features),
    weather_dim=len(weather_features),
    crop_dim=len(crop_to_id),
    hidden_dim=32
).to(DEVICE)


# =========================================================
# BUILD VALID MONTHLY TRANSITIONS
# =========================================================

def build_transitions(data):

    transitions = []

    for site_id, group in data.groupby("site_id"):

        group = group.sort_values("date").reset_index(drop=True)

        for i in range(len(group) - 1):

            current = group.iloc[i]
            nxt = group.iloc[i + 1]

            # Require genuinely consecutive months.
            expected_next = (
                current["date"] +
                pd.offsets.MonthBegin(1)
            )

            if nxt["date"] != expected_next:
                continue

            # The transition model requires a complete
            # state at both ends and complete weather
            # forcing at the current month.
            required_current = (
                state_features +
                weather_features
            )

            required_next = state_features

            if current[required_current].isna().any():
                continue

            if nxt[required_next].isna().any():
                continue

            transitions.append(
                {
                    "site_id": site_id,
                    "crop": current["crop"],
                    "date": current["date"],

                    "state": current[state_features].values.astype(
                        np.float32
                    ),

                    "weather": current[weather_features].values.astype(
                        np.float32
                    ),

                    "target": nxt[state_features].values.astype(
                        np.float32
                    )
                }
            )

    return transitions


transitions = build_transitions(train_df)

print("\nVALID MONTHLY TRANSITIONS:", len(transitions))

if len(transitions) == 0:
    raise RuntimeError(
        "No valid monthly transitions were found."
    )


# =========================================================
# TRAINING ARRAYS
# =========================================================

states_np = np.stack(
    [x["state"] for x in transitions]
)

weather_np = np.stack(
    [x["weather"] for x in transitions]
)

targets_np = np.stack(
    [x["target"] for x in transitions]
)

crop_np = np.array(
    [
        crop_to_id[x["crop"]]
        for x in transitions
    ],
    dtype=np.int64
)


states = torch.tensor(
    states_np,
    dtype=torch.float32,
    device=DEVICE
)

weather = torch.tensor(
    weather_np,
    dtype=torch.float32,
    device=DEVICE
)

targets = torch.tensor(
    targets_np,
    dtype=torch.float32,
    device=DEVICE
)

crops = torch.tensor(
    crop_np,
    dtype=torch.long,
    device=DEVICE
)


# =========================================================
# TRAINING
# =========================================================

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=1e-3
)

loss_fn = nn.MSELoss()

epochs = 500

print("\nTRAINING...")

for epoch in range(epochs):

    model.train()

    optimizer.zero_grad()

    predicted = model(
        states,
        weather,
        crops
    )

    loss = loss_fn(
        predicted,
        targets
    )

    loss.backward()

    optimizer.step()

    if epoch == 0 or (epoch + 1) % 50 == 0:

        print(
            f"Epoch {epoch + 1:04d} "
            f"Loss {loss.item():.6f}"
        )


# =========================================================
# SAVE MODEL
# =========================================================

torch.save(
    {
        "model_state_dict":
            model.state_dict(),

        "state_mean":
            state_mean.to_dict(),

        "state_std":
            state_std.to_dict(),

        "weather_mean":
            weather_mean.to_dict(),

        "weather_std":
            weather_std.to_dict(),

        "crop_to_id":
            crop_to_id,

        "state_features":
            state_features,

        "weather_features":
            weather_features,

        "train_start":
            "2019-01-01",

        "train_end":
            "2023-12-01",

        "model_type":
            "shared_global_five_crop_temporal_twin"
    },
    MODEL_FILE
)


print("\n========================================")
print("MODEL SAVED")
print("========================================")

print(MODEL_FILE)
