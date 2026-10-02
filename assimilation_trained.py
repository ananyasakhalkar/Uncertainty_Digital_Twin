import pandas as pd
import numpy as np
import torch

from state_model import CropStateModel
from observation_model import (
    S1ObservationModel,
    S2ObservationModel
)


# =========================================================
# DEVICE
# =========================================================

DEVICE = (
    "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)

DATA = "data/model/model_daily.csv"


# =========================================================
# SETTINGS
# =========================================================

STATE_DIM = 4
FORCING_DIM = 3

N_ENSEMBLE = 50

PROCESS_NOISE = 0.05
OBS_NOISE_S1 = 0.15
OBS_NOISE_S2 = 0.15

SEED = 42

rng = np.random.default_rng(SEED)


# =========================================================
# LOAD DATA
# =========================================================

df = pd.read_csv(DATA)

df["date"] = pd.to_datetime(df["date"])

test = df[
    df["date"] >= "2024-01-01"
].copy()

print("Using device:", DEVICE)
print("Temporal holdout: 2024")
print("Test days:", len(test))


# =========================================================
# LOAD TRAINED MODELS
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
        "transition_model.pt",
        map_location=DEVICE,
        weights_only=True
    )
)

s1_model.load_state_dict(
    torch.load(
        "s1_observation_model.pt",
        map_location=DEVICE,
        weights_only=True
    )
)

s2_model.load_state_dict(
    torch.load(
        "s2_observation_model.pt",
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
    0.0,
    0.10,
    size=(N_ENSEMBLE, STATE_DIM)
)


# =========================================================
# METRICS
# =========================================================

s2_errors = []

s2_count = 0
s1_count = 0


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

        predicted_observations = (
            observation_model(state_tensor)
            .cpu()
            .numpy()
        )

    n = ensemble.shape[0]

    state_mean = ensemble.mean(axis=0)

    observation_mean = (
        predicted_observations.mean(axis=0)
    )

    state_anomalies = (
        ensemble - state_mean
    )

    observation_anomalies = (
        predicted_observations
        - observation_mean
    )

    cross_covariance = (
        state_anomalies.T
        @ observation_anomalies
        / (n - 1)
    )

    observation_covariance = (
        observation_anomalies.T
        @ observation_anomalies
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
            size=predicted_observations.shape
        )
    )

    innovations = (
        perturbed_observations
        - predicted_observations
    )

    ensemble = (
        ensemble
        + innovations
        @ kalman_gain.T
    )

    return ensemble


# =========================================================
# DAILY DIGITAL-TWIN LOOP
# =========================================================

for _, row in test.iterrows():

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

        predicted_states = transition(
            state_tensor,
            forcing_tensor
        )

    ensemble = (
        predicted_states
        .cpu()
        .numpy()
    )

    ensemble += rng.normal(
        0,
        PROCESS_NOISE,
        size=ensemble.shape
    )


    # -----------------------------------------------------
    # S2 ASSIMILATION
    # -----------------------------------------------------

    if row["optical_available"] == 1:

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

        if np.all(np.isfinite(observed_s2)):

            ensemble = enkf_update(
                ensemble,
                observed_s2,
                s2_model,
                OBS_NOISE_S2
            )

            s2_count += 1

            predicted_ndvi = (
                ensemble[:, 0].mean()
            )

            s2_errors.append(
                predicted_ndvi
                - row["NDVI_z"]
            )


    # -----------------------------------------------------
    # S1 ASSIMILATION
    # -----------------------------------------------------

    if row["sar_available"] == 1:

        observed_s1 = np.array(
            [
                row["VV_z"],
                row["VH_z"],
                row["VV_VH_z"]
            ],
            dtype=np.float32
        )

        if np.all(np.isfinite(observed_s1)):

            ensemble = enkf_update(
                ensemble,
                observed_s1,
                s1_model,
                OBS_NOISE_S1
            )

            s1_count += 1


# =========================================================
# RESULTS
# =========================================================

if len(s2_errors) > 0:

    s2_rmse = np.sqrt(
        np.mean(
            np.square(s2_errors)
        )
    )

else:

    s2_rmse = np.nan


print()
print("==========================================")
print("TRAINED LATENT TWIN + ENKF")
print("==========================================")

print(
    "S2 assimilation observations:",
    s2_count
)

print(
    "S1 assimilation observations:",
    s1_count
)

print(
    "S2 RMSE:",
    s2_rmse
)
