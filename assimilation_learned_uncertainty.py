import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

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

PROCESS = os.path.join(
    BASE,
    "data/global/learned_process_uncertainty.csv"
)

OUT = os.path.join(
    BASE,
    "data/global/US_learned_uncertainty_2024.csv"
)

SUMMARY = os.path.join(
    BASE,
    "data/global/US_learned_uncertainty_summary_2024.csv"
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
TEST_START = "2024-01-01"

N_ENSEMBLE = 50


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
    ["site_id", "date"]
).reset_index(drop=True)

df = df.dropna(
    subset=STATE_FEATURES + WEATHER_FEATURES
).copy()


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
# LOAD MODEL
# ============================================================

device = torch.device(
    "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)

print("Device:", device)

model = CropTwin(
    state_dim=6,
    weather_dim=3,
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
# LOAD LEARNED PROCESS UNCERTAINTY
# ============================================================

process_df = pd.read_csv(PROCESS)

process_map = dict(
    zip(
        process_df["feature"],
        process_df["process_std"]
    )
)

process_std = np.array([
    process_map[f]
    for f in STATE_FEATURES
])

print("\nLearned process standard deviations:")

for feature, value in zip(
    STATE_FEATURES,
    process_std
):

    print(
        f"{feature}: {value:.4f}"
    )


# ============================================================
# PROPER NEXT-MONTH FORECAST
#
# At month t:
#
# posterior state from t-1
#       ↓
# forecast month t
#       ↓
# evaluate against observation t
#
# Observation t is NOT used before evaluation.
#
# Then observation t is assimilated using a simple
# observation-space update.
# ============================================================

records = []


for site_id, site_df in df.groupby(
    "site_id"
):

    site_df = (
        site_df
        .sort_values("date")
        .reset_index(drop=True)
    )

    # Start from first available state
    # before 2024.

    previous_state = None

    previous_date = None

    previous_crop = None

    for i in range(
        len(site_df)
    ):

        row = site_df.iloc[i]

        current_date = row["date"]

        # ----------------------------------------------------
        # Before 2024, simply propagate using observations.
        #
        # This establishes the state entering 2024.
        # ----------------------------------------------------

        if current_date < pd.Timestamp(
            TEST_START
        ):

            previous_state = standardize_state(
                row[
                    STATE_FEATURES
                ].values.astype(float)
            )

            previous_date = current_date

            previous_crop = int(
                row["crop_id"]
            )

            continue

        # ----------------------------------------------------
        # Require consecutive monthly observations
        # ----------------------------------------------------

        if previous_state is None:
            continue

        expected_date = (
            previous_date
            + pd.offsets.MonthBegin(1)
        )

        if current_date != expected_date:
            continue

        # ----------------------------------------------------
        # Previous-month weather
        # ----------------------------------------------------

        previous_row = site_df.iloc[i - 1]

        previous_weather = standardize_weather(
            previous_row[
                WEATHER_FEATURES
            ].values.astype(float)
        )

        # ----------------------------------------------------
        # Current observation
        # ----------------------------------------------------

        current_obs = standardize_state(
            row[
                STATE_FEATURES
            ].values.astype(float)
        )

        # ----------------------------------------------------
        # Create ensemble around previous posterior
        # ----------------------------------------------------

        ensemble = np.repeat(
            previous_state[None, :],
            N_ENSEMBLE,
            axis=0
        )

        # Initial small state spread
        ensemble += np.random.normal(
            0,
            0.01,
            size=ensemble.shape
        )

        state_tensor = torch.tensor(
            ensemble,
            dtype=torch.float32,
            device=device
        )

        weather_tensor = torch.tensor(
            previous_weather,
            dtype=torch.float32,
            device=device
        ).unsqueeze(0).repeat(
            N_ENSEMBLE,
            1
        )

        crop_tensor = torch.tensor(
            [previous_crop] * N_ENSEMBLE,
            dtype=torch.long,
            device=device
        )

        # ----------------------------------------------------
        # Forecast
        # ----------------------------------------------------

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

        # ----------------------------------------------------
        # ADD LEARNED PROCESS NOISE
        # ----------------------------------------------------

        noise = np.random.normal(
            0,
            process_std[None, :],
            size=predicted.shape
        )

        predicted += noise

        prior_mean = predicted.mean(
            axis=0
        )

        prior_std = predicted.std(
            axis=0,
            ddof=1
        )

        error = (
            prior_mean
            - current_obs
        )

        # ----------------------------------------------------
        # Simple observation update
        #
        # Use the ensemble Kalman update with observation
        # noise estimated from the historical process scale.
        # ----------------------------------------------------

        obs_noise = np.maximum(
            0.10,
            process_std * 0.25
        )

        for j in range(
            len(STATE_FEATURES)
        ):

            variance = (
                prior_std[j] ** 2
            )

            observation_variance = (
                obs_noise[j] ** 2
            )

            gain = (
                variance /
                (
                    variance
                    + observation_variance
                )
            )

            predicted[:, j] += (
                gain *
                (
                    current_obs[j]
                    - predicted[:, j]
                )
            )

        posterior_mean = predicted.mean(
            axis=0
        )

        posterior_std = predicted.std(
            axis=0,
            ddof=1
        )

        # ----------------------------------------------------
        # Store
        # ----------------------------------------------------

        for j, feature in enumerate(
            STATE_FEATURES
        ):

            records.append({

                "site_id": site_id,

                "crop": row["crop"],

                "date": current_date,

                "feature": feature,

                "prior_mean":
                    prior_mean[j],

                "prior_std":
                    prior_std[j],

                "observation":
                    current_obs[j],

                "prior_abs_error":
                    abs(error[j]),

                "posterior_mean":
                    posterior_mean[j],

                "posterior_std":
                    posterior_std[j],

                "posterior_abs_error":
                    abs(
                        posterior_mean[j]
                        - current_obs[j]
                    )

            })

        # ----------------------------------------------------
        # Posterior becomes next state
        # ----------------------------------------------------

        previous_state = posterior_mean

        previous_date = current_date

        previous_crop = int(
            row["crop_id"]
        )


# ============================================================
# RESULTS
# ============================================================

results = pd.DataFrame(
    records
)

results.to_csv(
    OUT,
    index=False
)


# ============================================================
# SUMMARY
# ============================================================

summary_rows = []

for feature, g in results.groupby(
    "feature"
):

    prior_rmse = np.sqrt(
        np.mean(
            g["prior_abs_error"] ** 2
        )
    )

    posterior_rmse = np.sqrt(
        np.mean(
            g["posterior_abs_error"] ** 2
        )
    )

    prior_std = g[
        "prior_std"
    ].mean()

    posterior_std = g[
        "posterior_std"
    ].mean()

    prior_coverage = (
        g["prior_abs_error"]
        <= g["prior_std"]
    ).mean()

    summary_rows.append({

        "feature": feature,

        "n": len(g),

        "prior_RMSE":
            prior_rmse,

        "posterior_RMSE":
            posterior_rmse,

        "RMSE_reduction_percent":
            (
                100 *
                (
                    prior_rmse
                    - posterior_rmse
                )
                / prior_rmse
            ),

        "mean_prior_std":
            prior_std,

        "mean_posterior_std":
            posterior_std,

        "prior_1std_coverage":
            prior_coverage

    })


summary = pd.DataFrame(
    summary_rows
)

summary.to_csv(
    SUMMARY,
    index=False
)


print(
    "\n======================================"
)

print(
    "LEARNED-UNCERTAINTY 2024 RESULTS"
)

print(
    "======================================"
)

print(
    summary.round(4)
)

print(
    "\nSaved:"
)

print(OUT)
print(SUMMARY)
