import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

# ============================================================
# CONFIG
# ============================================================

BASE = os.path.expanduser("~/Downloads/Crop_Digital_Twin")

DATA = os.path.join(
    BASE, "data/global/US_global_panel_2019_2024.csv"
)

MODEL = os.path.join(
    BASE, "models/US_multisite_temporal_twin.pt"
)

OUT = os.path.join(
    BASE, "data/global/US_uncertainty_calibration_2024.csv"
)

SUMMARY = os.path.join(
    BASE, "data/global/US_uncertainty_calibration_summary.csv"
)

N_ENSEMBLE = 50
PROCESS_NOISE = 0.05

STATE_FEATURES = [
    "NDVI",
    "EVI",
    "NDWI",
    "VV",
    "VH",
    "VV_VH",
]

WEATHER_FEATURES = [
    "temperature_2m",
    "precipitation",
    "solar_radiation",
]

TRAIN_END = "2023-12-01"
TEST_START = "2024-01-01"


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

        self.crop_embedding = nn.Embedding(crop_dim, 4)

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

    def forward(self, state, weather, crop):

        crop_emb = self.crop_embedding(crop)

        x = torch.cat(
            [state, weather, crop_emb],
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

df["date"] = pd.to_datetime(df["date"])

for col in STATE_FEATURES + WEATHER_FEATURES:
    df[col] = pd.to_numeric(
        df[col],
        errors="coerce"
    )

df = df.sort_values(
    ["site_id", "date"]
).reset_index(drop=True)

# Keep only rows with complete state and weather
df = df.dropna(
    subset=STATE_FEATURES + WEATHER_FEATURES
).copy()

print("Rows:", len(df))
print("Sites:", df["site_id"].nunique())


# ============================================================
# CROP ENCODING
# ============================================================

crops = sorted(df["crop"].dropna().unique())

crop_to_id = {
    crop: i
    for i, crop in enumerate(crops)
}

df["crop_id"] = df["crop"].map(crop_to_id)

print("Crop mapping:", crop_to_id)


# ============================================================
# TRAINING STANDARDIZATION
# ============================================================

train_df = df[
    df["date"] <= pd.Timestamp(TRAIN_END)
].copy()

state_mean = train_df[STATE_FEATURES].mean()
state_std = train_df[STATE_FEATURES].std()

weather_mean = train_df[WEATHER_FEATURES].mean()
weather_std = train_df[WEATHER_FEATURES].std()

state_std = state_std.replace(0, 1.0)
weather_std = weather_std.replace(0, 1.0)


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
    state_dim=len(STATE_FEATURES),
    weather_dim=len(WEATHER_FEATURES),
    crop_dim=len(crops),
    hidden_dim=32
)

checkpoint = torch.load(
    MODEL,
    map_location=device
)

if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
    model.load_state_dict(
        checkpoint["model_state_dict"]
    )
else:
    model.load_state_dict(checkpoint)

model = model.to(device)
model.eval()


# ============================================================
# HELPER
# ============================================================

def tensor(x):

    return torch.tensor(
        x,
        dtype=torch.float32,
        device=device
    )


# ============================================================
# UNCERTAINTY CALIBRATION
#
# IMPORTANT:
#
# At month t:
#
# 1. Generate prior ensemble
# 2. Calculate prior mean
# 3. Calculate prior std
# 4. Compare prior mean with ACTUAL observation
#
# The observation is NOT used before this comparison.
# ============================================================

records = []

test_start = pd.Timestamp(TEST_START)

print("\nRunning uncertainty experiment...")


for site_id, site_df in df.groupby("site_id"):

    site_df = site_df.sort_values("date").reset_index(drop=True)

    # Need complete chronological monthly transitions
    for i in range(1, len(site_df)):

        prev = site_df.iloc[i - 1]
        current = site_df.iloc[i]

        # Only evaluate 2024
        if current["date"] < test_start:
            continue

        # Require exactly one month between observations
        expected_date = (
            prev["date"] + pd.offsets.MonthBegin(1)
        )

        if current["date"] != expected_date:
            continue

        prev_state = standardize_state(
            prev[STATE_FEATURES].values.astype(float)
        )

        prev_weather = standardize_weather(
            prev[WEATHER_FEATURES].values.astype(float)
        )

        current_obs = standardize_state(
            current[STATE_FEATURES].values.astype(float)
        )

        # ----------------------------------------------------
        # Create ensemble
        # ----------------------------------------------------

        ensemble = np.repeat(
            prev_state[None, :],
            N_ENSEMBLE,
            axis=0
        )

        # Small initial spread
        ensemble += np.random.normal(
            0,
            PROCESS_NOISE,
            size=ensemble.shape
        )

        weather_tensor = tensor(
            prev_weather
        ).unsqueeze(0).repeat(
            N_ENSEMBLE,
            1
        )

        crop_tensor = torch.tensor(
            [int(prev["crop_id"])] * N_ENSEMBLE,
            dtype=torch.long,
            device=device
        )

        state_tensor = tensor(ensemble)

        # ----------------------------------------------------
        # Prior forecast
        # ----------------------------------------------------

        with torch.no_grad():

            predicted = model(
                state_tensor,
                weather_tensor,
                crop_tensor
            )

        predicted = predicted.cpu().numpy()

        # ----------------------------------------------------
        # Ensemble statistics
        # ----------------------------------------------------

        prior_mean = predicted.mean(axis=0)

        prior_std = predicted.std(
            axis=0,
            ddof=1
        )

        # ----------------------------------------------------
        # Forecast error
        # ----------------------------------------------------

        error = prior_mean - current_obs

        abs_error = np.abs(error)

        squared_error = error ** 2

        # ----------------------------------------------------
        # Store each variable
        # ----------------------------------------------------

        for j, feature in enumerate(STATE_FEATURES):

            records.append({

                "site_id": site_id,

                "crop": current["crop"],

                "date": current["date"],

                "feature": feature,

                "prior_mean": prior_mean[j],

                "prior_std": prior_std[j],

                "observation": current_obs[j],

                "error": error[j],

                "abs_error": abs_error[j],

                "squared_error": squared_error[j],

            })


# ============================================================
# SAVE RAW RESULTS
# ============================================================

results = pd.DataFrame(records)

results.to_csv(
    OUT,
    index=False
)

print("\nSaved:")
print(OUT)

print(
    "Forecast records:",
    len(results)
)


# ============================================================
# 1. SPEARMAN CORRELATION
#
# Does higher predicted uncertainty correspond
# to larger absolute forecast errors?
# ============================================================

summary_rows = []

for feature, g in results.groupby("feature"):

    if len(g) >= 3:

        spearman = g[
            "prior_std"
        ].corr(
            g["abs_error"],
            method="spearman"
        )

    else:

        spearman = np.nan

    # --------------------------------------------------------
    # Coverage of ±1 standard deviation
    # --------------------------------------------------------

    within_1std = (
        g["abs_error"] <= g["prior_std"]
    ).mean()

    # --------------------------------------------------------
    # Coverage of ±2 standard deviations
    # --------------------------------------------------------

    within_2std = (
        g["abs_error"] <= 2 * g["prior_std"]
    ).mean()

    # --------------------------------------------------------
    # Mean absolute error
    # --------------------------------------------------------

    mae = g["abs_error"].mean()

    rmse = np.sqrt(
        g["squared_error"].mean()
    )

    mean_std = g["prior_std"].mean()

    summary_rows.append({

        "feature": feature,

        "n": len(g),

        "spearman_uncertainty_error":
            spearman,

        "mean_prior_std":
            mean_std,

        "MAE":
            mae,

        "RMSE":
            rmse,

        "coverage_1std":
            within_1std,

        "coverage_2std":
            within_2std,

    })


summary = pd.DataFrame(summary_rows)


# ============================================================
# 2. UNCERTAINTY QUARTILES
#
# Divide forecasts into four groups:
#
# Q1 = lowest uncertainty
# Q4 = highest uncertainty
#
# Then check whether forecast error increases.
# ============================================================

quartile_rows = []

for feature, g in results.groupby("feature"):

    g = g.copy()

    try:

        g["uncertainty_quartile"] = pd.qcut(
            g["prior_std"],
            4,
            labels=[
                "Q1_low",
                "Q2",
                "Q3",
                "Q4_high"
            ],
            duplicates="drop"
        )

    except Exception:

        continue

    for q, qg in g.groupby(
        "uncertainty_quartile",
        observed=True
    ):

        quartile_rows.append({

            "feature": feature,

            "uncertainty_quartile": str(q),

            "n": len(qg),

            "mean_uncertainty":
                qg["prior_std"].mean(),

            "MAE":
                qg["abs_error"].mean(),

            "RMSE":
                np.sqrt(
                    qg["squared_error"].mean()
                ),

        })


quartiles = pd.DataFrame(
    quartile_rows
)


# ============================================================
# SAVE SUMMARY
# ============================================================

summary.to_csv(
    SUMMARY,
    index=False
)

quartile_path = os.path.join(
    BASE,
    "data/global/US_uncertainty_quartiles_2024.csv"
)

quartiles.to_csv(
    quartile_path,
    index=False
)


# ============================================================
# PRINT RESULTS
# ============================================================

pd.set_option(
    "display.max_columns",
    None
)

print("\n==============================")
print("UNCERTAINTY SUMMARY")
print("==============================")

print(summary.round(4))


print("\n==============================")
print("UNCERTAINTY QUARTILES")
print("==============================")

print(quartiles.round(4))


print("\nSaved:")
print(SUMMARY)

print(quartile_path)
