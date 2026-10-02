import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

# ============================================================
# CONFIG
# ============================================================

BASE = os.path.expanduser(
    "~/Downloads/Crop_Digital_Twin"
)

DATA = os.path.join(
    BASE,
    "data/global/US_global_panel_2019_2024.csv"
)

MODEL = os.path.join(
    BASE,
    "models/US_multisite_temporal_twin.pt"
)

OUTPUT = os.path.join(
    BASE,
    "data/global/learned_process_uncertainty.csv"
)

STATE_FEATURES = [
    "NDVI",
    "EVI",
    "NDWI",
    "VV",
    "VH",
    "VV_VH"
]

WEATHER_FEATURES = [
    "temperature_2m",
    "precipitation",
    "solar_radiation"
]

TRAIN_END = "2023-12-01"


# ============================================================
# MODEL
# ============================================================

class CropTwin(nn.Module):

    def __init__(
        self,
        state_dim=6,
        weather_dim=3,
        crop_dim=2,
        hidden_dim=32
    ):
        super().__init__()

        self.crop_embedding = nn.Embedding(
            crop_dim,
            4
        )

        self.transition = nn.Sequential(
            nn.Linear(
                state_dim + weather_dim + 4,
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

        delta = 0.10 * torch.tanh(delta)

        return state + delta


# ============================================================
# LOAD DATA
# ============================================================

print("Loading data...")

df = pd.read_csv(DATA)

df["date"] = pd.to_datetime(
    df["date"]
)

for col in STATE_FEATURES + WEATHER_FEATURES:

    df[col] = pd.to_numeric(
        df[col],
        errors="coerce"
    )

df = df.sort_values(
    [
        "site_id",
        "date"
    ]
).reset_index(drop=True)

df = df.dropna(
    subset=STATE_FEATURES + WEATHER_FEATURES
).copy()

print(
    "Rows:",
    len(df)
)

print(
    "Sites:",
    df["site_id"].nunique()
)


# ============================================================
# CROP ENCODING
# ============================================================

crops = sorted(
    df["crop"].dropna().unique()
)

crop_to_id = {
    crop: i
    for i, crop in enumerate(crops)
}

df["crop_id"] = df["crop"].map(
    crop_to_id
)

print(
    "Crop mapping:",
    crop_to_id
)


# ============================================================
# TRAINING STANDARDIZATION
# ============================================================

train_df = df[
    df["date"] <= pd.Timestamp(TRAIN_END)
].copy()

state_mean = train_df[
    STATE_FEATURES
].mean()

state_std = train_df[
    STATE_FEATURES
].std()

weather_mean = train_df[
    WEATHER_FEATURES
].mean()

weather_std = train_df[
    WEATHER_FEATURES
].std()

state_std = state_std.replace(
    0,
    1.0
)

weather_std = weather_std.replace(
    0,
    1.0
)


def standardize_state(x):

    return (
        x - state_mean.values
    ) / state_std.values


def standardize_weather(x):

    return (
        x - weather_mean.values
    ) / weather_std.values


# ============================================================
# DEVICE
# ============================================================

device = torch.device(
    "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)

print(
    "Device:",
    device
)


# ============================================================
# LOAD TRAINED MODEL
# ============================================================

model = CropTwin(
    state_dim=len(STATE_FEATURES),
    weather_dim=len(WEATHER_FEATURES),
    crop_dim=len(crops),
    hidden_dim=32
)

checkpoint = torch.load(
    MODEL,
    map_location=device
)

if (
    isinstance(checkpoint, dict)
    and "model_state_dict" in checkpoint
):

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

else:

    model.load_state_dict(
        checkpoint
    )

model = model.to(device)

model.eval()


# ============================================================
# COLLECT HISTORICAL TRANSITION RESIDUALS
#
# For every valid consecutive monthly transition:
#
# predicted next state - observed next state
#
# These residuals describe model/process uncertainty.
# ============================================================

records = []

print(
    "\nCalculating historical transition residuals..."
)

for site_id, site_df in df.groupby(
    "site_id"
):

    site_df = (
        site_df
        .sort_values("date")
        .reset_index(drop=True)
    )

    for i in range(
        1,
        len(site_df)
    ):

        previous = site_df.iloc[i - 1]
        current = site_df.iloc[i]

        # Require consecutive months
        expected_date = (
            previous["date"]
            + pd.offsets.MonthBegin(1)
        )

        if current["date"] != expected_date:
            continue

        # Only historical training period
        if current["date"] > pd.Timestamp(
            TRAIN_END
        ):
            continue

        previous_state = standardize_state(
            previous[
                STATE_FEATURES
            ].values.astype(float)
        )

        previous_weather = standardize_weather(
            previous[
                WEATHER_FEATURES
            ].values.astype(float)
        )

        current_state = standardize_state(
            current[
                STATE_FEATURES
            ].values.astype(float)
        )

        state_tensor = torch.tensor(
            previous_state,
            dtype=torch.float32,
            device=device
        ).unsqueeze(0)

        weather_tensor = torch.tensor(
            previous_weather,
            dtype=torch.float32,
            device=device
        ).unsqueeze(0)

        crop_tensor = torch.tensor(
            [int(previous["crop_id"])],
            dtype=torch.long,
            device=device
        )

        with torch.no_grad():

            prediction = model(
                state_tensor,
                weather_tensor,
                crop_tensor
            )

        prediction = (
            prediction
            .cpu()
            .numpy()[0]
        )

        residual = (
            current_state
            - prediction
        )

        for j, feature in enumerate(
            STATE_FEATURES
        ):

            records.append({

                "site_id": site_id,

                "date": current["date"],

                "crop": current["crop"],

                "feature": feature,

                "prediction": prediction[j],

                "observation": current_state[j],

                "residual": residual[j],

                "abs_residual":
                    abs(residual[j]),

                "squared_residual":
                    residual[j] ** 2

            })


residuals = pd.DataFrame(
    records
)


# ============================================================
# ESTIMATE PROCESS UNCERTAINTY
# ============================================================

summary = []

for feature, g in residuals.groupby(
    "feature"
):

    residual_std = g[
        "residual"
    ].std(
        ddof=1
    )

    residual_mean = g[
        "residual"
    ].mean()

    mae = g[
        "abs_residual"
    ].mean()

    rmse = np.sqrt(
        g[
            "squared_residual"
        ].mean()
    )

    summary.append({

        "feature": feature,

        "n_transitions": len(g),

        "mean_residual":
            residual_mean,

        "process_std":
            residual_std,

        "process_mae":
            mae,

        "process_rmse":
            rmse

    })


summary = pd.DataFrame(
    summary
)


# ============================================================
# SAVE
# ============================================================

summary.to_csv(
    OUTPUT,
    index=False
)


print(
    "\n===================================="
)

print(
    "LEARNED PROCESS UNCERTAINTY"
)

print(
    "===================================="
)

print(
    summary.round(4)
)

print(
    "\nSaved:"
)

print(
    OUTPUT
)
