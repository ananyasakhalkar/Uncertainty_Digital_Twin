import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from pathlib import Path


# =========================================================
# PATHS
# =========================================================

BASE = Path.home() / "Downloads" / "Crop_Digital_Twin"

DATA = BASE / "data" / "global"

MODEL_FILE = (
    BASE /
    "models" /
    "US_multisite_temporal_twin.pt"
)

OUT_FILE = (
    DATA /
    "US_free_running_2024_predictions.csv"
)


# =========================================================
# DEVICE
# =========================================================

DEVICE = torch.device(
    "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)

print("DEVICE:", DEVICE)


# =========================================================
# MODEL
# =========================================================

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


# =========================================================
# LOAD DATA
# =========================================================

df = pd.read_csv(
    DATA / "US_global_panel_2019_2024.csv"
)

df["date"] = pd.to_datetime(
    df["date"],
    errors="coerce"
)

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
    "temperature_2m",
    "precipitation",
    "solar_radiation"
]


# =========================================================
# FORCE MODEL VARIABLES TO NUMERIC
# =========================================================

for c in state_features + weather_features:

    df[c] = pd.to_numeric(
        df[c],
        errors="coerce"
    )


# =========================================================
# LOAD CHECKPOINT
# =========================================================

checkpoint = torch.load(
    MODEL_FILE,
    map_location=DEVICE
)

state_mean = pd.Series(
    checkpoint["state_mean"]
)

state_std = pd.Series(
    checkpoint["state_std"]
).replace(0, 1)

weather_mean = pd.Series(
    checkpoint["weather_mean"]
)

weather_std = pd.Series(
    checkpoint["weather_std"]
).replace(0, 1)

crop_to_id = checkpoint["crop_to_id"]


# =========================================================
# STANDARDIZE
# =========================================================

for c in state_features:

    df[c] = (
        df[c] - state_mean[c]
    ) / state_std[c]


for c in weather_features:

    df[c] = (
        df[c] - weather_mean[c]
    ) / weather_std[c]


# =========================================================
# CREATE MODEL
# =========================================================

model = CropTwin(
    state_dim=6,
    weather_dim=3,
    crop_dim=len(crop_to_id)
).to(DEVICE)

model.load_state_dict(
    checkpoint["model_state_dict"]
)

model.eval()


# =========================================================
# 2024 FREE-RUNNING FORECAST
# =========================================================

results = []


for site_id, group in df.groupby("site_id"):

    group = (
        group
        .sort_values("date")
        .reset_index(drop=True)
    )

    crop = group["crop"].iloc[0]

    crop_id = crop_to_id[crop]


    # -----------------------------------------------------
    # TRAIN PERIOD
    # -----------------------------------------------------

    train_part = group[
        group["date"] < "2024-01-01"
    ].copy()


    # -----------------------------------------------------
    # TEST PERIOD
    # -----------------------------------------------------

    test_part = group[
        group["date"] >= "2024-01-01"
    ].copy()


    # Need a valid observed state to initialize
    train_part = train_part.dropna(
        subset=state_features
    )


    # Need valid weather for prediction
    test_part = test_part.dropna(
        subset=weather_features
    )


    if len(train_part) == 0:
        print(
            "Skipping site with no initial state:",
            site_id
        )
        continue


    # -----------------------------------------------------
    # INITIAL STATE
    # Last observed state before 2024
    # -----------------------------------------------------

    current_state = torch.tensor(
        train_part[
            state_features
        ].iloc[-1].values,
        dtype=torch.float32,
        device=DEVICE
    ).unsqueeze(0)


    crop_tensor = torch.tensor(
        [crop_id],
        dtype=torch.long,
        device=DEVICE
    )


    # -----------------------------------------------------
    # FREE-RUNNING LOOP
    # -----------------------------------------------------

    for _, row in test_part.iterrows():

        weather_values = (
            row[
                weather_features
            ]
            .astype(float)
            .values
        )


        weather = torch.tensor(
            weather_values,
            dtype=torch.float32,
            device=DEVICE
        ).unsqueeze(0)


        with torch.no_grad():

            predicted_state = model(
                current_state,
                weather,
                crop_tensor
            )


        predicted = (
            predicted_state
            .cpu()
            .numpy()[0]
        )


        # -------------------------------------------------
        # OBSERVED STATE
        # -------------------------------------------------

        actual = row[
            state_features
        ].astype(float).values


        result = {
            "site_id": site_id,
            "crop": crop,
            "date": row["date"]
        }


        for i, feature in enumerate(
            state_features
        ):

            result[
                feature + "_pred"
            ] = predicted[i]

            result[
                feature + "_obs"
            ] = actual[i]


        results.append(result)


        # -------------------------------------------------
        # IMPORTANT:
        # FREE-RUNNING MODEL USES ITS OWN
        # PREVIOUS PREDICTION AS THE NEXT STATE.
        # -------------------------------------------------

        current_state = predicted_state


# =========================================================
# CREATE RESULTS TABLE
# =========================================================

results = pd.DataFrame(results)


# =========================================================
# SAVE PREDICTIONS
# =========================================================

results.to_csv(
    OUT_FILE,
    index=False
)


# =========================================================
# METRICS
# =========================================================

print("\n========================================")
print("FREE-RUNNING 2024 RESULTS")
print("========================================")


for feature in state_features:

    pred = results[
        feature + "_pred"
    ]

    obs = results[
        feature + "_obs"
    ]


    mask = (
        pred.notna() &
        obs.notna()
    )


    if mask.sum() == 0:

        print(
            f"{feature:8s} "
            "NO VALID OBSERVATIONS"
        )

        continue


    rmse = np.sqrt(
        np.mean(
            (
                pred[mask] -
                obs[mask]
            ) ** 2
        )
    )


    mae = np.mean(
        np.abs(
            pred[mask] -
            obs[mask]
        )
    )


    print(
        f"{feature:8s} "
        f"RMSE = {rmse:.4f} "
        f"MAE = {mae:.4f}"
    )


# =========================================================
# SUMMARY
# =========================================================

print(
    "\nRows evaluated:",
    len(results)
)

print(
    "Sites evaluated:",
    results["site_id"].nunique()
)

print(
    "Crops evaluated:",
    results["crop"].unique()
)


print("\nSaved:")
print(OUT_FILE)
