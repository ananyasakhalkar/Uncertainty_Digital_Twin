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

N_ENSEMBLE = 50

PROCESS_NOISE = 0.05
OBS_NOISE_S1 = 0.15
OBS_NOISE_S2 = 0.15

SEED = 42

rng = np.random.default_rng(SEED)


# =========================================================
# LOAD DATA
# =========================================================

df = pd.read_csv(
    DATA,
    parse_dates=["date"]
)

df = df.sort_values("date").reset_index(drop=True)

# Historical calibration period only
calibration = df[
    df["date"] < "2024-01-01"
].copy()

print("Using device:", DEVICE)
print("Calibration period: 2019-2023")
print("Days:", len(calibration))


# =========================================================
# LOAD FROZEN MODEL
# =========================================================

transition = CropStateModel().to(DEVICE)
s1_model = S1ObservationModel().to(DEVICE)
s2_model = S2ObservationModel().to(DEVICE)

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
# INITIAL ENSEMBLE
# =========================================================

ensemble = rng.normal(
    0,
    0.10,
    size=(N_ENSEMBLE, 4)
)


# =========================================================
# ENKF UPDATE
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

    return (
        ensemble
        + innovation
        @ kalman_gain.T
    )


# =========================================================
# STORAGE
# =========================================================

optical_scores = []
sar_scores = []


# =========================================================
# HISTORICAL LOOP
# =========================================================

for _, row in calibration.iterrows():

    # -----------------------------------------------------
    # WEATHER
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
    # PREDICT
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
    # S2
    # -----------------------------------------------------

    if row["S2_available"] == 1:

        observed = np.array(
            [
                row["NDVI_z"],
                row["EVI_z"],
                row["FPAR_z"],
                row["NDWI_z"],
                row["SVI_z"]
            ],
            dtype=np.float32
        )

        if np.isfinite(observed).all():

            state_tensor = torch.tensor(
                ensemble,
                dtype=torch.float32,
                device=DEVICE
            )

            with torch.no_grad():

                prediction = (
                    s2_model(state_tensor)
                    .cpu()
                    .numpy()
                )

            mean_prediction = prediction.mean(
                axis=0
            )

            spread = prediction.std(
                axis=0,
                ddof=1
            )

            innovation = (
                observed - mean_prediction
            )

            normalized = (
                innovation
                / np.sqrt(
                    spread ** 2
                    + OBS_NOISE_S2 ** 2
                    + 1e-8
                )
            )

            score = np.sqrt(
                np.mean(
                    normalized ** 2
                )
            )

            optical_scores.append(score)

            ensemble = enkf_update(
                ensemble,
                observed,
                s2_model,
                OBS_NOISE_S2
            )


    # -----------------------------------------------------
    # S1
    # -----------------------------------------------------

    if row["S1_available"] == 1:

        observed = np.array(
            [
                row["VV_z"],
                row["VH_z"],
                row["VV_VH_z"]
            ],
            dtype=np.float32
        )

        if np.isfinite(observed).all():

            state_tensor = torch.tensor(
                ensemble,
                dtype=torch.float32,
                device=DEVICE
            )

            with torch.no_grad():

                prediction = (
                    s1_model(state_tensor)
                    .cpu()
                    .numpy()
                )

            mean_prediction = prediction.mean(
                axis=0
            )

            spread = prediction.std(
                axis=0,
                ddof=1
            )

            innovation = (
                observed - mean_prediction
            )

            normalized = (
                innovation
                / np.sqrt(
                    spread ** 2
                    + OBS_NOISE_S1 ** 2
                    + 1e-8
                )
            )

            score = np.sqrt(
                np.mean(
                    normalized ** 2
                )
            )

            sar_scores.append(score)

            ensemble = enkf_update(
                ensemble,
                observed,
                s1_model,
                OBS_NOISE_S1
            )


# =========================================================
# CALIBRATION
# =========================================================

optical_scores = np.array(
    optical_scores
)

sar_scores = np.array(
    sar_scores
)


optical_threshold_95 = np.percentile(
    optical_scores,
    95
)

sar_threshold_95 = np.percentile(
    sar_scores,
    95
)


# =========================================================
# RESULTS
# =========================================================

print()
print("==========================================")
print("HISTORICAL DEVIATION CALIBRATION")
print("==========================================")

print(
    "S2 calibration observations:",
    len(optical_scores)
)

print(
    "S1 calibration observations:",
    len(sar_scores)
)

print()
print(
    "S2 mean deviation:",
    optical_scores.mean()
)

print(
    "S2 95th percentile:",
    optical_threshold_95
)

print()
print(
    "S1 mean deviation:",
    sar_scores.mean()
)

print(
    "S1 95th percentile:",
    sar_threshold_95
)

print()
print(
    "These 95th-percentile values are "
    "historical reference thresholds, "
    "not validated crop-stress thresholds."
)
