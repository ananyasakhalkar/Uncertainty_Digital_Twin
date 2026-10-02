import pandas as pd
import numpy as np
import torch
import torch.nn as nn

from state_model import CropStateModel
from observation_model import S1ObservationModel, S2ObservationModel


DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"

print("Using device:", DEVICE)

# --------------------------------------------------
# Load data
# --------------------------------------------------

df = pd.read_csv(
    "data/model/model_daily.csv",
    parse_dates=["date"]
)

df = df.sort_values("date").reset_index(drop=True)

# Training period only
train = df[df["date"].dt.year <= 2023].copy()

print("Training period:")
print(train["date"].min().date(), "to", train["date"].max().date())

print("S1 observations:", train["S1_available"].sum())
print("S2 observations:", train["S2_available"].sum())


# --------------------------------------------------
# Columns
# --------------------------------------------------

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
# Models
# --------------------------------------------------

transition = CropStateModel().to(DEVICE)

s1_model = S1ObservationModel().to(DEVICE)

s2_model = S2ObservationModel().to(DEVICE)


optimizer = torch.optim.Adam(
    list(transition.parameters())
    + list(s1_model.parameters())
    + list(s2_model.parameters()),
    lr=5e-4
)

loss_fn = nn.MSELoss()


# --------------------------------------------------
# Windowed training
# --------------------------------------------------

WINDOW = 30
EPOCHS = 100

for epoch in range(EPOCHS):

    total_loss = 0.0
    count = 0

    # Start a fresh state for each window.
    for start in range(
        0,
        len(train) - WINDOW,
        WINDOW
    ):

        window = train.iloc[
            start:start + WINDOW
        ]

        state = torch.zeros(
            (1, 4),
            dtype=torch.float32,
            device=DEVICE
        )

        optimizer.zero_grad()

        window_loss = None
        observations = 0

        for _, row in window.iterrows():

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

            # Predict latent state
            state = transition(
                state,
                forcing
            )

            # S2 observation
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

                    loss = loss_fn(
                        prediction,
                        target
                    )

                    window_loss = (
                        loss
                        if window_loss is None
                        else window_loss + loss
                    )

                    observations += 1

            # S1 observation
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

                    loss = loss_fn(
                        prediction,
                        target
                    )

                    window_loss = (
                        loss
                        if window_loss is None
                        else window_loss + loss
                    )

                    observations += 1

        if window_loss is not None:

            window_loss = (
                window_loss / observations
            )

            window_loss.backward()

            torch.nn.utils.clip_grad_norm_(
                list(transition.parameters())
                + list(s1_model.parameters())
                + list(s2_model.parameters()),
                1.0
            )

            optimizer.step()

            total_loss += window_loss.item()
            count += 1

    if (epoch + 1) % 10 == 0:

        print(
            f"Epoch {epoch + 1:03d} "
            f"| Window loss: "
            f"{total_loss / max(count, 1):.6f}"
        )


# --------------------------------------------------
# Save
# --------------------------------------------------

torch.save(
    transition.state_dict(),
    "windowed_transition_model.pt"
)

torch.save(
    s1_model.state_dict(),
    "windowed_s1_model.pt"
)

torch.save(
    s2_model.state_dict(),
    "windowed_s2_model.pt"
)

print("\nWindowed training complete.")
print("Saved windowed model files.")
