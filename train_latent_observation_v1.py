import pandas as pd
import numpy as np
import torch
import torch.nn as nn

DEVICE = torch.device(
    "mps" if torch.backends.mps.is_available() else "cpu"
)

DATA = "GLOBAL_ALL25_MODEL_READY.csv"

STATE = ["NDVI","EVI","NDWI","VV","VH","VV_VH"]
S1 = ["VV","VH","VV_VH"]
S2 = ["NDVI","EVI","NDWI"]

df = pd.read_csv(DATA)
df["date"] = pd.to_datetime(df["date"])

train = df[
    (df["date"] >= "2019-01-01") &
    (df["date"] <= "2022-12-01")
].copy()

valid = df[
    (df["date"] >= "2023-01-01") &
    (df["date"] <= "2023-12-01")
].copy()

state_mean = train[STATE].mean()
state_std = train[STATE].std()

train[STATE] = (
    train[STATE] - state_mean
) / state_std

valid[STATE] = (
    valid[STATE] - state_mean
) / state_std

# Only complete observations are used for this first
# observation-model experiment.
train_complete = train.dropna(subset=STATE).copy()
valid_complete = valid.dropna(subset=STATE).copy()

crop_to_id = {
    crop: i
    for i, crop in enumerate(
        sorted(train["crop"].unique())
    )
}


class LatentObservationModel(nn.Module):

    def __init__(self):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Linear(6, 32),
            nn.Tanh(),
            nn.Linear(32, 4)
        )

        self.s1_head = nn.Sequential(
            nn.Linear(4, 16),
            nn.Tanh(),
            nn.Linear(16, 3)
        )

        self.s2_head = nn.Sequential(
            nn.Linear(4, 16),
            nn.Tanh(),
            nn.Linear(16, 3)
        )

    def forward(self, x):
        z = self.encoder(x)

        s1 = self.s1_head(z)
        s2 = self.s2_head(z)

        return z, s1, s2


model = LatentObservationModel().to(DEVICE)

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=1e-3,
    weight_decay=1e-4
)

loss_fn = nn.MSELoss()

X_train = torch.tensor(
    train_complete[STATE].values,
    dtype=torch.float32
).to(DEVICE)

X_valid = torch.tensor(
    valid_complete[STATE].values,
    dtype=torch.float32
).to(DEVICE)

s1_idx = [STATE.index(x) for x in S1]
s2_idx = [STATE.index(x) for x in S2]

Y1_train = X_train[:, s1_idx]
Y2_train = X_train[:, s2_idx]

Y1_valid = X_valid[:, s1_idx]
Y2_valid = X_valid[:, s2_idx]

best_val = float("inf")
best_state = None
patience = 60
bad = 0

for epoch in range(1, 501):

    model.train()

    optimizer.zero_grad()

    _, pred_s1, pred_s2 = model(X_train)

    loss_s1 = loss_fn(
        pred_s1,
        Y1_train
    )

    loss_s2 = loss_fn(
        pred_s2,
        Y2_train
    )

    loss = loss_s1 + loss_s2

    loss.backward()

    torch.nn.utils.clip_grad_norm_(
        model.parameters(),
        1.0
    )

    optimizer.step()

    model.eval()

    with torch.no_grad():

        _, v1, v2 = model(X_valid)

        val = (
            loss_fn(v1, Y1_valid)
            +
            loss_fn(v2, Y2_valid)
        ).item()

    if val < best_val:

        best_val = val
        best_state = {
            k: v.detach().cpu().clone()
            for k, v in model.state_dict().items()
        }
        bad = 0

    else:
        bad += 1

    if epoch == 1 or epoch % 50 == 0:

        print(
            f"Epoch {epoch:3d} | "
            f"train {loss.item():.6f} | "
            f"val {val:.6f}"
        )

    if bad >= patience:

        print(
            f"Early stopping at epoch {epoch}"
        )
        break


model.load_state_dict(best_state)

checkpoint = {
    "model_type":
        "latent_observation_v1",

    "model_state_dict":
        model.state_dict(),

    "state_features":
        STATE,

    "s1_features":
        S1,

    "s2_features":
        S2,

    "latent_dim":
        4,

    "state_mean":
        state_mean.to_dict(),

    "state_std":
        state_std.to_dict(),

    "crop_to_id":
        crop_to_id,

    "train_period":
        "2019-2022",

    "validation_period":
        "2023",

    "best_validation_loss":
        best_val
}

torch.save(
    checkpoint,
    "models/GLOBAL_LATENT_OBSERVATION_V1.pt"
)

print("\n======================================")
print("LATENT OBSERVATION V1")
print("======================================")
print("Train rows:", len(train_complete))
print("Validation rows:", len(valid_complete))
print("Latent dimension: 4")
print("Best validation loss:", best_val)
print(
    "Saved: models/GLOBAL_LATENT_OBSERVATION_V1.pt"
)
