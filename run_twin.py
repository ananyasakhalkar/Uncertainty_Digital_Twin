import pandas as pd
import numpy as np
from pathlib import Path

from enkf import EnsembleKalmanFilter


# --------------------------------------------------
# Load data
# --------------------------------------------------

DATA = "data/model/model_daily.csv"

df = pd.read_csv(
    DATA,
    parse_dates=["date"]
)

df = df.sort_values("date").reset_index(drop=True)


# --------------------------------------------------
# Observation columns
# --------------------------------------------------

s2_cols = [
    "NDVI_z",
    "EVI_z",
    "FPAR_z",
    "NDWI_z",
    "SVI_z"
]

s1_cols = [
    "VV_z",
    "VH_z",
    "VV_VH_z"
]

forcing_cols = [
    "temperature_C_z",
    "precipitation_mm_z",
    "solar_radiation_MJ_z"
]


# --------------------------------------------------
# Simple prototype transition
# --------------------------------------------------

def transition(state, forcing):

    temperature = forcing[0]
    precipitation = forcing[1]
    radiation = forcing[2]

    next_state = state.copy()

    # Small weather-driven changes
    next_state[0] += (
        0.015 * radiation
        + 0.010 * precipitation
        - 0.010 * temperature
    )

    next_state[1] += (
        0.010 * radiation
        + 0.005 * precipitation
    )

    next_state[2] += (
        0.020 * precipitation
        - 0.010 * temperature
    )

    # Phenology progresses gradually
    next_state[3] += 0.01

    return next_state


# --------------------------------------------------
# Prototype observation mappings
# --------------------------------------------------

def s2_observation(state):

    return np.array([
        state[0],
        state[0],
        state[0],
        state[2],
        state[0]
    ])


def s1_observation(state):

    return np.array([
        state[1],
        state[1],
        state[1]
    ])


# --------------------------------------------------
# Initialize EnKF
# --------------------------------------------------

enkf = EnsembleKalmanFilter(
    state_dim=4,
    ensemble_size=50,
    process_noise=0.05,
    observation_noise_s1=0.20,
    observation_noise_s2=0.20,
    seed=42
)


# --------------------------------------------------
# Storage
# --------------------------------------------------

results = []


# --------------------------------------------------
# Daily twin loop
# --------------------------------------------------

for _, row in df.iterrows():

    forcing = np.array([
        row[c]
        for c in forcing_cols
    ])

    # Missing weather should never enter the model
    if not np.isfinite(forcing).all():
        continue

    # ------------------------------
    # 1. Predict
    # ------------------------------

    enkf.predict(
        transition,
        forcing
    )

    # ------------------------------
    # 2. S1 assimilation
    # ------------------------------

    if row["S1_available"] == 1:

        observation = np.array([
            row[c]
            for c in s1_cols
        ])

        if np.isfinite(observation).all():

            enkf.update(
                observation,
                s1_observation,
                enkf.R_s1
            )

    # ------------------------------
    # 3. S2 assimilation
    # ------------------------------

    if row["S2_available"] == 1:

        observation = np.array([
            row[c]
            for c in s2_cols
        ])

        if np.isfinite(observation).all():

            enkf.update(
                observation,
                s2_observation,
                enkf.R_s2
            )

    # ------------------------------
    # 4. Save state
    # ------------------------------

    state_mean = enkf.state_mean()
    state_std = enkf.state_std()

    results.append({
        "date": row["date"],

        "state_0": state_mean[0],
        "state_1": state_mean[1],
        "state_2": state_mean[2],
        "state_3": state_mean[3],

        "uncertainty_0": state_std[0],
        "uncertainty_1": state_std[1],
        "uncertainty_2": state_std[2],
        "uncertainty_3": state_std[3],

        "S1_available": row["S1_available"],
        "S2_available": row["S2_available"]
    })


# --------------------------------------------------
# Save
# --------------------------------------------------

results = pd.DataFrame(results)

output = Path(
    "data/model/twin_states.csv"
)

output.parent.mkdir(
    exist_ok=True
)

results.to_csv(
    output,
    index=False
)


print("Digital twin run complete.")
print("Days processed:", len(results))

print(
    "S1 assimilation days:",
    results["S1_available"].sum()
)

print(
    "S2 assimilation days:",
    results["S2_available"].sum()
)

print(
    "Output:",
    output
)
