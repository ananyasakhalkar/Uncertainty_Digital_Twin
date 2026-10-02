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
    "US_prior_posterior_2024.csv"
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
# LOAD TRAINING CHECKPOINT
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
# MODEL
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
# ENKF SETTINGS
# =========================================================

ENSEMBLE_SIZE = 50

PROCESS_NOISE = 0.05

OBSERVATION_NOISE = 0.15


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


    # Need a valid initial state
    train_part = train_part.dropna(
        subset=state_features
    )


    if len(train_part) == 0:
        continue


    # -----------------------------------------------------
    # INITIAL STATE
    # -----------------------------------------------------

    initial_state = (
        train_part[
            state_features
        ]
        .iloc[-1]
        .values
        .astype(float)
    )


    # -----------------------------------------------------
    # ENSEMBLE INITIALIZATION
    # -----------------------------------------------------

    rng = np.random.default_rng(
        seed=42
    )


    ensemble = (
        initial_state[None, :]
        +
        rng.normal(
            0,
            PROCESS_NOISE,
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
    # 2024 MONTH LOOP
    # -----------------------------------------------------

    for _, row in test_part.iterrows():

        weather_values = row[
            weather_features
        ].values.astype(float)


        # Cannot make a forecast without weather
        if not np.all(
            np.isfinite(weather_values)
        ):
            continue


        # =================================================
        # PRIOR FORECAST
        # =================================================

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


        with torch.no_grad():

            prior_tensor = model(
                state_tensor,
                weather_tensor,
                crop_tensor
            )


        prior = (
            prior_tensor
            .cpu()
            .numpy()
        )


        # Add process uncertainty
        prior += rng.normal(
            0,
            PROCESS_NOISE,
            prior.shape
        )


        prior_mean = prior.mean(
            axis=0
        )

        prior_std = prior.std(
            axis=0
        )


        # =================================================
        # CURRENT OBSERVATION
        # =================================================

        observation = row[
            state_features
        ].values.astype(float)


        valid = np.isfinite(
            observation
        )


        # =================================================
        # PRIOR ERROR
        # =================================================

        for i, feature in enumerate(
            state_features
        ):

            results.append({
                "site_id": site_id,
                "crop": crop,
                "date": row["date"],

                "feature": feature,

                "prior": prior_mean[i],

                "posterior": np.nan,

                "observation": observation[i],

                "prior_uncertainty":
                    prior_std[i],

                "posterior_uncertainty":
                    np.nan
            })


        # =================================================
        # ENKF UPDATE
        # =================================================

        if valid.any():

            observed_values = observation[
                valid
            ]

            ensemble_values = prior[
                :,
                valid
            ]


            ensemble_mean = (
                ensemble_values.mean(
                    axis=0
                )
            )


            # Observation perturbations
            obs_perturbations = rng.normal(
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


            # Ensemble anomalies
            X = (
                ensemble_values -
                ensemble_mean
            )


            # Observation-space anomalies
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


            K = (
                Pxy @
                np.linalg.pinv(
                    Pyy + R
                )
            )


            innovation = (
                perturbed_observations -
                ensemble_values
            )


            update = (
                innovation @ K.T
            )


            posterior = prior.copy()


            posterior[
                :,
                valid
            ] += update


        else:

            posterior = prior.copy()


        # =================================================
        # POSTERIOR STATISTICS
        # =================================================

        posterior_mean = (
            posterior.mean(axis=0)
        )

        posterior_std = (
            posterior.std(axis=0)
        )


        # =================================================
        # STORE POSTERIOR
        # =================================================

        start_index = (
            len(results) -
            len(state_features)
        )


        for i, feature in enumerate(
            state_features
        ):

            results[
                start_index + i
            ]["posterior"] = (
                posterior_mean[i]
            )

            results[
                start_index + i
            ]["posterior_uncertainty"] = (
                posterior_std[i]
            )


        # =================================================
        # IMPORTANT
        #
        # The posterior becomes the starting state
        # for the NEXT month's forecast.
        # =================================================

        ensemble = posterior


# =========================================================
# RESULTS TABLE
# =========================================================

results = pd.DataFrame(
    results
)


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
print("PRIOR vs POSTERIOR 2024")
print("========================================")


summary = []


for feature in state_features:

    subset = results[
        results["feature"] == feature
    ].copy()


    prior_mask = (
        subset["prior"].notna() &
        subset["observation"].notna()
    )


    posterior_mask = (
        subset["posterior"].notna() &
        subset["observation"].notna()
    )


    prior_rmse = np.sqrt(
        np.mean(
            (
                subset.loc[
                    prior_mask,
                    "prior"
                ]
                -
                subset.loc[
                    prior_mask,
                    "observation"
                ]
            ) ** 2
        )
    )


    posterior_rmse = np.sqrt(
        np.mean(
            (
                subset.loc[
                    posterior_mask,
                    "posterior"
                ]
                -
                subset.loc[
                    posterior_mask,
                    "observation"
                ]
            ) ** 2
        )
    )


    improvement = (
        100 *
        (
            prior_rmse -
            posterior_rmse
        )
        /
        prior_rmse
    )


    print(
        f"{feature:8s} "
        f"Prior RMSE = {prior_rmse:.4f} "
        f"Posterior RMSE = {posterior_rmse:.4f} "
        f"Reduction = {improvement:.2f}%"
    )


    summary.append({
        "feature": feature,
        "prior_rmse": prior_rmse,
        "posterior_rmse": posterior_rmse,
        "rmse_reduction_percent":
            improvement
    })


summary = pd.DataFrame(
    summary
)


SUMMARY_FILE = (
    DATA /
    "US_prior_posterior_summary_2024.csv"
)


summary.to_csv(
    SUMMARY_FILE,
    index=False
)


# =========================================================
# UNCERTAINTY
# =========================================================

print("\n========================================")
print("MEAN ENSEMBLE UNCERTAINTY")
print("========================================")


for feature in state_features:

    subset = results[
        results["feature"] == feature
    ]

    prior_unc = subset[
        "prior_uncertainty"
    ].mean()

    posterior_unc = subset[
        "posterior_uncertainty"
    ].mean()

    print(
        f"{feature:8s} "
        f"Prior = {prior_unc:.4f} "
        f"Posterior = {posterior_unc:.4f}"
    )


print("\nRows:", len(results))

print(
    "Sites:",
    results["site_id"].nunique()
)

print("\nSaved:")
print(OUT_FILE)

print(SUMMARY_FILE)
