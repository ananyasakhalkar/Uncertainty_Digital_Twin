import pandas as pd
import numpy as np

from enkf import EnsembleKalmanFilter


# ==================================================
# Load data
# ==================================================

df = pd.read_csv(
    "data/model/model_daily.csv",
    parse_dates=["date"]
)

df = df.sort_values("date").reset_index(drop=True)

test = df[
    df["date"].dt.year == 2024
].copy()


# ==================================================
# Columns
# ==================================================

forcing_cols = [
    "temperature_C_z",
    "precipitation_mm_z",
    "solar_radiation_MJ_z"
]

s1_cols = [
    "VV_z",
    "VH_z",
    "VV_VH_z"
]

s2_cols = [
    "NDVI_z",
    "EVI_z",
    "FPAR_z",
    "NDWI_z",
    "SVI_z"
]


# ==================================================
# Prototype dynamic model
# ==================================================

def transition(state, forcing):

    temperature = forcing[0]
    precipitation = forcing[1]
    radiation = forcing[2]

    next_state = state.copy()

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

    next_state[3] += 0.01

    return next_state


# ==================================================
# Observation models
# ==================================================

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


# ==================================================
# Run one twin experiment
# ==================================================

def run_twin(use_s1=True, use_s2=True):

    enkf = EnsembleKalmanFilter(
        state_dim=4,
        ensemble_size=50,
        process_noise=0.05,
        observation_noise_s1=0.20,
        observation_noise_s2=0.20,
        seed=42
    )

    errors = []

    for _, row in test.iterrows():

        forcing = row[
            forcing_cols
        ].values.astype(float)

        if not np.isfinite(forcing).all():
            continue

        # ------------------------------------------
        # Predict
        # ------------------------------------------

        enkf.predict(
            transition,
            forcing
        )

        # ------------------------------------------
        # S1 assimilation
        # ------------------------------------------

        if (
            use_s1
            and row["S1_available"] == 1
        ):

            obs = row[
                s1_cols
            ].values.astype(float)

            if np.isfinite(obs).all():

                enkf.update(
                    obs,
                    s1_observation,
                    enkf.R_s1
                )

        # ------------------------------------------
        # S2 assimilation
        # ------------------------------------------

        if (
            use_s2
            and row["S2_available"] == 1
        ):

            obs = row[
                s2_cols
            ].values.astype(float)

            if np.isfinite(obs).all():

                enkf.update(
                    obs,
                    s2_observation,
                    enkf.R_s2
                )

        # ------------------------------------------
        # Evaluate state against NDVI
        # ------------------------------------------

        if row["S2_available"] == 1:

            ndvi = row["NDVI_z"]

            if np.isfinite(ndvi):

                predicted_ndvi = (
                    enkf.state_mean()[0]
                )

                errors.append(
                    (predicted_ndvi - ndvi) ** 2
                )

    return np.sqrt(np.mean(errors))


# ==================================================
# Experiments
# ==================================================

print("2024 assimilation experiment")
print()

print(
    "S1 + weather:",
    run_twin(
        use_s1=True,
        use_s2=False
    )
)

print(
    "S1 + S2 + weather:",
    run_twin(
        use_s1=True,
        use_s2=True
    )
)
