import pandas as pd
import numpy as np
import torch

from state_model import CropStateModel
from observation_model import S1ObservationModel, S2ObservationModel


DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"

print("Using device:", DEVICE)

df = pd.read_csv(
    "data/model/model_daily.csv",
    parse_dates=["date"]
)

df = df.sort_values("date").reset_index(drop=True)

# --------------------------------------------------
# Load windowed models
# --------------------------------------------------

transition = CropStateModel().to(DEVICE)
transition.load_state_dict(
    torch.load(
        "windowed_transition_model.pt",
        map_location=DEVICE
    )
)

s1_model = S1ObservationModel().to(DEVICE)
s1_model.load_state_dict(
    torch.load(
        "windowed_s1_model.pt",
        map_location=DEVICE
    )
)

s2_model = S2ObservationModel().to(DEVICE)
s2_model.load_state_dict(
    torch.load(
        "windowed_s2_model.pt",
        map_location=DEVICE
    )
)

transition.eval()
s1_model.eval()
s2_model.eval()


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


# --------------------------------------------------
# Evaluate 2024
# --------------------------------------------------

test = df[
    df["date"].dt.year == 2024
].copy()

state = torch.zeros(
    (1, 4),
    dtype=torch.float32,
    device=DEVICE
)

s1_errors = []
s2_errors = []


with torch.no_grad():

    for _, row in test.iterrows():

        forcing_np = row[
            forcing_cols
        ].values.astype(np.float32)

        if not np.isfinite(forcing_np).all():
            continue

        forcing = torch.tensor(
            forcing_np,
            dtype=torch.float32,
            device=DEVICE
        ).unsqueeze(0)

        state = transition(
            state,
            forcing
        )

        # S2
        if row["S2_available"] == 1:

            obs = row[
                s2_cols
            ].values.astype(np.float32)

            if np.isfinite(obs).all():

                target = torch.tensor(
                    obs,
                    dtype=torch.float32,
                    device=DEVICE
                ).unsqueeze(0)

                prediction = s2_model(state)

                mse = torch.mean(
                    (prediction - target) ** 2
                ).item()

                s2_errors.append(mse)

        # S1
        if row["S1_available"] == 1:

            obs = row[
                s1_cols
            ].values.astype(np.float32)

            if np.isfinite(obs).all():

                target = torch.tensor(
                    obs,
                    dtype=torch.float32,
                    device=DEVICE
                ).unsqueeze(0)

                prediction = s1_model(state)

                mse = torch.mean(
                    (prediction - target) ** 2
                ).item()

                s1_errors.append(mse)


print("\n2024 temporal holdout")

print("S2 observations:", len(s2_errors))
print("S1 observations:", len(s1_errors))

if s2_errors:
    print(
        "S2 MSE:",
        np.mean(s2_errors)
    )
    print(
        "S2 RMSE:",
        np.sqrt(np.mean(s2_errors))
    )

if s1_errors:
    print(
        "S1 MSE:",
        np.mean(s1_errors)
    )
    print(
        "S1 RMSE:",
        np.sqrt(np.mean(s1_errors))
    )
