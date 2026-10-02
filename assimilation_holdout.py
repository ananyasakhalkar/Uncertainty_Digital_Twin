import pandas as pd
import numpy as np
import torch

from state_model import CropStateModel
from observation_model import (
    S1ObservationModel,
    S2ObservationModel
)


# =========================================================
# SETTINGS
# =========================================================

DEVICE = (
    "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)

DATA = "data/model/model_daily.csv"

STATE_DIM = 4
FORCING_DIM = 3

N_ENSEMBLE = 50

PROCESS_NOISE = 0.05
OBS_NOISE_S1 = 0.15
OBS_NOISE_S2 = 0.15

SEED = 42

rng = np.random.default_rng(SEED)


# =========================================================
# LOAD 2024 DATA
# =========================================================

df = pd.read_csv(
    DATA,
    parse_dates=["date"]
)

df = df.sort_values("date").reset_index(drop=True)

test = df[
    df["date"] >= "2024-01-01"
].copy()

print("Using device:", DEVICE)
print("Temporal holdout: 2024")
print("Test days:", len(test))


# =========================================================
# LOAD FROZEN 2019-2023 MODELS
# =========================================================

transition = CropStateModel(
    state_dim=STATE_DIM,
    forcing_dim=FORCING_DIM
).to(DEVICE)

s1_model = S1ObservationModel(
    state_dim=STATE_DIM
).to(DEVICE)

s2_model = S2ObservationModel(
    state_dim=STATE_DIM
).to(DEVICE)


transition.load_state_dict(
    torch.load(
        "transition_model_holdout.pt",
        map_location=DEVICE,
        weights_only=True
    )
)

s1_model.load_state_dict(
    torch.load(
        "s1_observation_model_holdout.pt",
        map_location=DEVICE,
        weights_only=True
    )
)

s2_model.load_state_dict(
    torch.load(
        "s2_observation_model_holdout.pt",
        map_location=DEVICE,
        weights_only=True
    )
)

transition.eval()
s1_model.eval()
s2_model.eval()


# =========================================================
# ENKF FUNCTION
# =========================================================

def enkf_update(
    ensemble,
    observed,
    observation_model,
    observation_noise
):

    state_tensor = torch.tensor(
        ensemble,
        dtype=torch.float32,
        device=DEVICE
    )

    with torch.no_grad():

        predicted_obs = (
            observation_model(state_tensor)
            .cpu()
            .numpy()
        )

    n = ensemble.shape[0]

    state_mean = ensemble.mean(axis=0)

    obs_mean = predicted_obs.mean(axis=0)

    state_anomaly = (
        ensemble - state_mean
    )

    obs_anomaly = (
        predicted_obs - obs_mean
    )

    cross_covariance = (
        state_anomaly.T
        @ obs_anomaly
        / (n - 1)
    )

    observation_covariance = (
        obs_anomaly.T
        @ obs_anomaly
        / (n - 1)
    )

    observation_covariance += (
        observation_noise ** 2
        * np.eye(len(observed))
    )

    kalman_gain = (
        cross_covariance
        @ np.linalg.pinv(
            observation_covariance
        )
    )

    perturbed_observations = (
        observed
        + rng.normal(
            0,
            observation_noise,
            size=predicted_obs.shape
        )
    )

    innovation = (
        perturbed_observations
        - predicted_obs
    )

    updated_ensemble = (
        ensemble
        + innovation
        @ kalman_gain.T
    )

    return updated_ensemble


# =========================================================
# INITIAL ENSEMBLE
# =========================================================

ensemble = rng.normal(
    loc=0.0,
    scale=0.10,
    size=(N_ENSEMBLE, STATE_DIM)
)


# =========================================================
# METRICS
# =========================================================

s1_errors = []
s2_errors = []

s1_count = 0
s2_count = 0


# =========================================================
# DAILY DIGITAL-TWIN LOOP
# =========================================================

for _, row in test.iterrows():

    # -----------------------------------------------------
    # DAILY WEATHER FORCING
    # -----------------------------------------------------

    forcing = np.array(
        [
            row["temperature_C_z"],
            row["precipitation_mm_z"],
            row["solar_radiation_MJ_z"]
        ],
        dtype=np.float32
    )

    forcing_batch = np.repeat(
        forcing.reshape(1, -1),
        N_ENSEMBLE,
        axis=0
    )

    state_tensor = torch.tensor(
        ensemble,
        dtype=torch.float32,
        device=DEVICE
    )

    forcing_tensor = torch.tensor(
        forcing_batch,
        dtype=torch.float32,
        device=DEVICE
    )


    # -----------------------------------------------------
    # 1. PREDICT
    # -----------------------------------------------------

    with torch.no_grad():

        predicted_state = transition(
            state_tensor,
            forcing_tensor
        )

    ensemble = (
        predicted_state
        .cpu()
        .numpy()
    )

    ensemble += rng.normal(
        0,
        PROCESS_NOISE,
        size=ensemble.shape
    )


    # -----------------------------------------------------
    # 2. PREDICT S2 OBSERVATION
    # -----------------------------------------------------

    if row["S2_available"] == 1:

        observed_s2 = np.array(
            [
                row["NDVI_z"],
                row["EVI_z"],
                row["FPAR_z"],
                row["NDWI_z"],
                row["SVI_z"]
            ],
            dtype=np.float32
        )

        if np.isfinite(observed_s2).all():

            state_tensor = torch.tensor(
                ensemble,
                dtype=torch.float32,
                device=DEVICE
            )

            with torch.no_grad():

                predicted_s2 = (
                    s2_model(state_tensor)
                    .cpu()
                    .numpy()
                )

            mean_prediction = (
                predicted_s2.mean(axis=0)
            )

            s2_errors.extend(
                mean_prediction - observed_s2
            )

            s2_count += 1


            # -------------------------------------------------
            # 3. ASSIMILATE S2
            # -------------------------------------------------

            ensemble = enkf_update(
                ensemble,
                observed_s2,
                s2_model,
                OBS_NOISE_S2
            )


    # -----------------------------------------------------
    # 4. PREDICT S1 OBSERVATION
    # -----------------------------------------------------

    if row["S1_available"] == 1:

        observed_s1 = np.array(
            [
                row["VV_z"],
                row["VH_z"],
                row["VV_VH_z"]
            ],
            dtype=np.float32
        )

        if np.isfinite(observed_s1).all():

            state_tensor = torch.tensor(
                ensemble,
                dtype=torch.float32,
                device=DEVICE
            )

            with torch.no_grad():

                predicted_s1 = (
                    s1_model(state_tensor)
                    .cpu()
                    .numpy()
                )

            mean_prediction = (
                predicted_s1.mean(axis=0)
            )

            s1_errors.extend(
                mean_prediction - observed_s1
            )

            s1_count += 1


            # -------------------------------------------------
            # 5. ASSIMILATE S1
            # -------------------------------------------------

            ensemble = enkf_update(
                ensemble,
                observed_s1,
                s1_model,
                OBS_NOISE_S1
            )


# =========================================================
# FINAL METRICS
# =========================================================

s2_rmse = np.sqrt(
    np.mean(
        np.square(s2_errors)
    )
)

s1_rmse = np.sqrt(
    np.mean(
        np.square(s1_errors)
    )
)


# =========================================================
# RESULTS
# =========================================================

print()
print("==========================================")
print("2024 HOLDOUT: LATENT TWIN + ENKF")
print("==========================================")

print(
    "S2 observations evaluated:",
    s2_count
)

print(
    "S1 observations evaluated:",
    s1_count
)

print(
    "S2 observation-space RMSE:",
    s2_rmse
)

print(
    "S1 observation-space RMSE:",
    s1_rmse
)

print()
print("Comparison:")
print("Seasonal S2 baseline: 0.4726600875")
print("Seasonal S1 baseline: 0.5258740830")
print("Free-running S2 twin: 0.6222874")
print("Free-running S1 twin: 0.6032182")
