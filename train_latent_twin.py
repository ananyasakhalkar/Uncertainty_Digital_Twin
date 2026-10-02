import pandas as pd
import numpy as np
import torch
import torch.nn as nn

from state_model import CropStateModel
from observation_model import (
    S1ObservationModel,
    S2ObservationModel
)


# =========================================================
# DEVICE
# =========================================================

device = (
    "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)

print("Using device:", device)


# =========================================================
# LOAD DATA
# =========================================================

df = pd.read_csv(
    "data/model/model_daily.csv",
    parse_dates=["date"]
)

df = df.sort_values("date").reset_index(drop=True)


# =========================================================
# TRUE TEMPORAL SPLIT
# =========================================================

train_df = df[
    df["date"] < "2024-01-01"
].copy()

test_df = df[
    df["date"] >= "2024-01-01"
].copy()

print(
    "Training period:",
    train_df["date"].min().date(),
    "to",
    train_df["date"].max().date()
)

print(
    "Holdout period:",
    test_df["date"].min().date(),
    "to",
    test_df["date"].max().date()
)

print(
    "Training S1 observations:",
    int(train_df["S1_available"].sum())
)

print(
    "Training S2 observations:",
    int(train_df["S2_available"].sum())
)

print(
    "Holdout S1 observations:",
    int(test_df["S1_available"].sum())
)

print(
    "Holdout S2 observations:",
    int(test_df["S2_available"].sum())
)


# =========================================================
# OBSERVATION VARIABLES
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

transition = CropStateModel().to(device)

s1_model = S1ObservationModel().to(device)

s2_model = S2ObservationModel().to(device)


# =========================================================
# OPTIMIZER
# =========================================================

optimizer = torch.optim.Adam(
    list(transition.parameters())
    + list(s1_model.parameters())
    + list(s2_model.parameters()),
    lr=1e-3
)


# =========================================================
# LOSS
# =========================================================

loss_fn = nn.MSELoss()


# =========================================================
# TRAINING
# =========================================================

epochs = 150

for epoch in range(epochs):

    # Reset latent state at beginning
    # of every training epoch.
    state = torch.zeros(
        (1, 4),
        dtype=torch.float32,
        device=device
    )

    total_loss = torch.tensor(
        0.0,
        dtype=torch.float32,
        device=device
    )

    observation_count = 0

    optimizer.zero_grad()

    for _, row in train_df.iterrows():

        # -------------------------------------------------
        # WEATHER FORCING
        # -------------------------------------------------

        forcing_np = row[
            forcing_cols
        ].values.astype(np.float32)

        if not np.isfinite(forcing_np).all():
            continue

        forcing = torch.tensor(
            forcing_np,
            dtype=torch.float32,
            device=device
        ).unsqueeze(0)


        # -------------------------------------------------
        # STATE TRANSITION
        # -------------------------------------------------

        state = transition(
            state,
            forcing
        )


        # -------------------------------------------------
        # SENTINEL-2 OBSERVATION
        # -------------------------------------------------

        if row["S2_available"] == 1:

            obs_np = row[
                s2_cols
            ].values.astype(np.float32)

            if np.isfinite(obs_np).all():

                observation = torch.tensor(
                    obs_np,
                    dtype=torch.float32,
                    device=device
                ).unsqueeze(0)

                prediction = s2_model(
                    state
                )

                loss = loss_fn(
                    prediction,
                    observation
                )

                total_loss = (
                    total_loss + loss
                )

                observation_count += 1


        # -------------------------------------------------
        # SENTINEL-1 OBSERVATION
        # -------------------------------------------------

        if row["S1_available"] == 1:

            obs_np = row[
                s1_cols
            ].values.astype(np.float32)

            if np.isfinite(obs_np).all():

                observation = torch.tensor(
                    obs_np,
                    dtype=torch.float32,
                    device=device
                ).unsqueeze(0)

                prediction = s1_model(
                    state
                )

                loss = loss_fn(
                    prediction,
                    observation
                )

                total_loss = (
                    total_loss + loss
                )

                observation_count += 1


    if observation_count == 0:
        raise RuntimeError(
            "No valid training observations found."
        )


    # -----------------------------------------------------
    # AVERAGE LOSS
    # -----------------------------------------------------

    total_loss = (
        total_loss / observation_count
    )


    # -----------------------------------------------------
    # BACKPROPAGATION
    # -----------------------------------------------------

    total_loss.backward()

    torch.nn.utils.clip_grad_norm_(
        list(transition.parameters())
        + list(s1_model.parameters())
        + list(s2_model.parameters()),
        max_norm=1.0
    )

    optimizer.step()


    # -----------------------------------------------------
    # LOGGING
    # -----------------------------------------------------

    if (epoch + 1) % 10 == 0:

        print(
            f"Epoch {epoch + 1:03d} "
            f"| Loss: {total_loss.item():.6f}"
        )


# =========================================================
# SAVE MODELS
# =========================================================

torch.save(
    transition.state_dict(),
    "transition_model_holdout.pt"
)

torch.save(
    s1_model.state_dict(),
    "s1_observation_model_holdout.pt"
)

torch.save(
    s2_model.state_dict(),
    "s2_observation_model_holdout.pt"
)


# =========================================================
# COMPLETE
# =========================================================

print()
print("Training complete.")

print(
    "Saved:",
    "transition_model_holdout.pt"
)

print(
    "Saved:",
    "s1_observation_model_holdout.pt"
)

print(
    "Saved:",
    "s2_observation_model_holdout.pt"
)
