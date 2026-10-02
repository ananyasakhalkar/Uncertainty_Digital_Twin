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
# LOAD 2024 HOLDOUT
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
print("Stress analysis period: 2024")
print("Days:", len(test))


# =========================================================
# LOAD FROZEN MODELS
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
# INITIAL ENSEMBLE
# =========================================================

ensemble = rng.normal(
    0.0,
    0.10,
    size=(N_ENSEMBLE, STATE_DIM)
)


# =========================================================
# DAILY RESULTS
# =========================================================

results = []


# =========================================================
# DIGITAL-TWIN LOOP
# =========================================================

for _, row in test.iterrows():

    date = row["date"]


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
    # PREDICT LATENT STATE
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
    # S2 INNOVATION / UNCERTAINTY
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

            mean_s2 = predicted_s2.mean(axis=0)

            std_s2 = predicted_s2.std(
                axis=0,
                ddof=1
            )

            innovation = (
                observed_s2 - mean_s2
            )

            # Normalized innovation
            normalized_innovation = (
                innovation
                / np.sqrt(
                    std_s2 ** 2
                    + OBS_NOISE_S2 ** 2
                    + 1e-8
                )
            )

            # Overall optical deviation
            optical_score = np.sqrt(
                np.mean(
                    normalized_innovation ** 2
                )
            )

            # Assimilate S2
            ensemble = enkf_update(
                ensemble,
                observed_s2,
                s2_model,
                OBS_NOISE_S2
            )

        else:

            optical_score = np.nan

    else:

        optical_score = np.nan


    # -----------------------------------------------------
    # S1 INNOVATION / UNCERTAINTY
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

            mean_s1 = predicted_s1.mean(axis=0)

            std_s1 = predicted_s1.std(
                axis=0,
                ddof=1
            )

            innovation = (
                observed_s1 - mean_s1
            )

            normalized_innovation = (
                innovation
                / np.sqrt(
                    std_s1 ** 2
                    + OBS_NOISE_S1 ** 2
                    + 1e-8
                )
            )

            sar_score = np.sqrt(
                np.mean(
                    normalized_innovation ** 2
                )
            )

            # Assimilate S1
            ensemble = enkf_update(
                ensemble,
                observed_s1,
                s1_model,
                OBS_NOISE_S1
            )

        else:

            sar_score = np.nan

    else:

        sar_score = np.nan


    # -----------------------------------------------------
    # ENSEMBLE STATE UNCERTAINTY
    # -----------------------------------------------------

    state_uncertainty = np.mean(
        np.std(
            ensemble,
            axis=0,
            ddof=1
        )
    )


    # -----------------------------------------------------
    # STORE
    # -----------------------------------------------------

    results.append(
        {
            "date": date,
            "optical_deviation": optical_score,
            "sar_deviation": sar_score,
            "state_uncertainty": state_uncertainty,
            "S2_available": row["S2_available"],
            "S1_available": row["S1_available"]
        }
    )


# =========================================================
# SAVE RESULTS
# =========================================================

results_df = pd.DataFrame(results)

results_df.to_csv(
    "stress_results_2024.csv",
    index=False
)


# =========================================================
# SUMMARY
# =========================================================

optical_values = results_df[
    "optical_deviation"
].dropna()

sar_values = results_df[
    "sar_deviation"
].dropna()


print()
print("==========================================")
print("2024 DIGITAL-TWIN UNCERTAINTY ANALYSIS")
print("==========================================")

print(
    "Optical observations:",
    len(optical_values)
)

print(
    "SAR observations:",
    len(sar_values)
)

print(
    "Mean optical deviation:",
    optical_values.mean()
)

print(
    "95th percentile optical deviation:",
    optical_values.quantile(0.95)
)

print(
    "Mean SAR deviation:",
    sar_values.mean()
)

print(
    "95th percentile SAR deviation:",
    sar_values.quantile(0.95)
)

print(
    "Mean state uncertainty:",
    results_df[
        "state_uncertainty"
    ].mean()
)

print()
print(
    "Saved:",
    "stress_results_2024.csv"
)
