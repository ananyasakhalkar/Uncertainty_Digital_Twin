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
    "US_assimilated_2024_predictions.csv"
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
# NUMERIC CONVERSION
# =========================================================

for c in state_features + weather_features:

    df[c] = pd.to_numeric(
        df[c],
        errors="coerce"
    )


# =========================================================
# LOAD MODEL
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
# ENKF PARAMETERS
# =========================================================

ENSEMBLE_SIZE = 50

PROCESS_NOISE = 0.05

OBSERVATION_NOISE = 0.15


# =========================================================
# OBSERVATION MASK
# =========================================================

def valid_observation(row):

    values = row[
        state_features
    ].values.astype(float)

    return np.isfinite(values)


# =========================================================
# RESULTS
# =========================================================

results = []


# =========================================================
# SITE LOOP
# =========================================================

for site_id, group in df.groupby("site_id"):

    group = (
        group
        .sort_values("date")
        .reset_index(drop=True)
    )

    crop = group["crop"].iloc[0]

    crop_id = crop_to_id[crop]


    # -----------------------------------------------------
    # TRAIN / TEST
    # -----------------------------------------------------

    train_part = group[
        group["date"] < "2024-01-01"
    ].copy()

    test_part = group[
        group["date"] >= "2024-01-01"
    ].copy()


    train_part = train_part.dropna(
        subset=state_features
    )


    if len(train_part) == 0:
        continue


    # -----------------------------------------------------
    # INITIAL STATE
    # -----------------------------------------------------

    initial_state = train_part[
        state_features
    ].iloc[-1].values.astype(float)


    # -----------------------------------------------------
    # ENSEMBLE
    # -----------------------------------------------------

    ensemble = (
        initial_state[None, :]
        +
        np.random.normal(
            0,
            0.05,
            size=(
                ENSEMBLE_SIZE,
                len(state_features)
            )
        )
    )


    crop_tensor = torch.full(
        (ENSEMBLE_SIZE,),
        crop_id,
        dtype=torch.long,
        device=DEVICE
    )


    # -----------------------------------------------------
    # TIME LOOP
    # -----------------------------------------------------

    for _, row in test_part.iterrows():

        weather_values = row[
            weather_features
        ].values.astype(float)


        if not np.all(
            np.isfinite(weather_values)
        ):
            continue


        weather_tensor = torch.tensor(
            np.repeat(
                weather_values[None, :],
                ENSEMBLE_SIZE,
                axis=0
            ),
            dtype=torch.float32,
            device=DEVICE
        )


        state_tensor = torch.tensor(
            ensemble,
            dtype=torch.float32,
            device=DEVICE
        )


        # =================================================
        # FORECAST STEP
        # =================================================

        with torch.no_grad():

            predicted = model(
                state_tensor,
                weather_tensor,
                crop_tensor
            )


        predicted = (
            predicted
            .cpu()
            .numpy()
        )


        # -------------------------------------------------
        # ADD PROCESS NOISE
        # -------------------------------------------------

        predicted += np.random.normal(
            0,
            PROCESS_NOISE,
            predicted.shape
        )


        # =================================================
        # ASSIMILATION STEP
        # =================================================

        observation = row[
            state_features
        ].values.astype(float)


        valid = np.isfinite(
            observation
        )


        if valid.any():

            observed_values = observation[
                valid
            ]


            ensemble_values = predicted[
                :,
                valid
            ]


            ensemble_mean = (
                ensemble_values.mean(axis=0)
            )


            # Observation perturbations
            obs_perturbations = np.random.normal(
                0,
                OBSERVATION_NOISE,
                size=(
                    ENSEMBLE_SIZE,
                    len(observed_values)
                )
            )


            perturbed_observations = (
                observed_values[None, :]
                +
                obs_perturbations
            )


            # Covariance
            X = (
                ensemble_values -
                ensemble_mean
            )


            Y = X.copy()


            Pxy = (
                X.T @ Y
            ) / max(
                ENSEMBLE_SIZE - 1,
                1
            )


            Pyy = (
                Y.T @ Y
            ) / max(
                ENSEMBLE_SIZE - 1,
                1
            )


            R = (
                OBSERVATION_NOISE ** 2
            ) * np.eye(
                len(observed_values)
            )


            K = Pxy @ np.linalg.pinv(
                Pyy + R
            )


            innovation = (
                perturbed_observations -
                ensemble_values
            )


            update = (
                innovation @ K.T
            )


            predicted[
                :,
                valid
            ] += update


        # -------------------------------------------------
        # ASSIMILATED STATE
        # -------------------------------------------------

        ensemble = predicted


        assimilated_mean = (
            ensemble.mean(axis=0)
        )


        assimilated_std = (
            ensemble.std(axis=0)
        )


        # -------------------------------------------------
        # SAVE
        # -------------------------------------------------

        result = {
            "site_id": site_id,
            "crop": crop,
            "date": row["date"]
        }


        for i, feature in enumerate(
            state_features
        ):

            result[
                feature + "_assim"
            ] = assimilated_mean[i]

            result[
                feature + "_obs"
            ] = observation[i]

            result[
                feature + "_uncertainty"
            ] = assimilated_std[i]


        results.append(result)


# =========================================================
# RESULTS DATAFRAME
# =========================================================

results = pd.DataFrame(results)


# =========================================================
# SAVE
# =========================================================

results.to_csv(
    OUT_FILE,
    index=False
)


# =========================================================
# METRICS
# =========================================================

print("\n========================================")
print("ASSIMILATED 2024 RESULTS")
print("========================================")


for feature in state_features:

    pred = results[
        feature + "_assim"
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


print(
    "\nRows evaluated:",
    len(results)
)

print(
    "Sites evaluated:",
    results["site_id"].nunique()
)

print("\nSaved:")
print(OUT_FILE)
