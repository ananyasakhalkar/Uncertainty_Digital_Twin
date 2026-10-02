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

print("Using device:", DEVICE)


# =========================================================
# DATA
# =========================================================

df = pd.read_csv(
    "data/model/model_daily.csv",
    parse_dates=["date"]
)

df = df.sort_values("date").reset_index(drop=True)

test = df[
    df["date"] >= "2024-01-01"
].copy()

print("Temporal holdout: 2024")
print("Test days:", len(test))


# =========================================================
# COLUMNS
# =========================================================

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


# =========================================================
# MODELS
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
# INITIAL STATE
# =========================================================

state = torch.zeros(
    (1, 4),
    dtype=torch.float32,
    device=DEVICE
)


# =========================================================
# ERRORS
# =========================================================

s1_errors = []
s2_errors = []


# =========================================================
# DAILY FREE-RUNNING PREDICTION
# =========================================================

for _, row in test.iterrows():

    forcing_np = row[
        forcing_cols
    ].values.astype(np.float32)

    forcing = torch.tensor(
        forcing_np,
        dtype=torch.float32,
        device=DEVICE
    ).unsqueeze(0)

    # Predict next latent state
    with torch.no_grad():

        state = transition(
            state,
            forcing
        )

    # ---------------------------------------------
    # S2 observation prediction
    # ---------------------------------------------

    if row["S2_available"] == 1:

        observed = row[
            s2_cols
        ].values.astype(np.float32)

        if np.isfinite(observed).all():

            with torch.no_grad():

                predicted = (
                    s2_model(state)
                    .cpu()
                    .numpy()
                    .flatten()
                )

            s2_errors.extend(
                predicted - observed
            )


    # ---------------------------------------------
    # S1 observation prediction
    # ---------------------------------------------

    if row["S1_available"] == 1:

        observed = row[
            s1_cols
        ].values.astype(np.float32)

        if np.isfinite(observed).all():

            with torch.no_grad():

                predicted = (
                    s1_model(state)
                    .cpu()
                    .numpy()
                    .flatten()
                )

            s1_errors.extend(
                predicted - observed
            )


# =========================================================
# METRICS
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
print("HOLDOUT FREE-RUNNING LATENT TWIN")
print("==========================================")

print(
    "S2 observations evaluated:",
    int(test["S2_available"].sum())
)

print(
    "S1 observations evaluated:",
    int(test["S1_available"].sum())
)

print(
    "S2 observation-space RMSE:",
    s2_rmse
)

print(
    "S1 observation-space RMSE:",
    s1_rmse
)
