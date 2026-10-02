import pandas as pd
import numpy as np
import torch
import torch.nn as nn

DATA = "GLOBAL_ALL25_MODEL_READY.csv"
MODEL = "models/GLOBAL_LATENT_OBSERVATION_V1.pt"

STATE = ["NDVI","EVI","NDWI","VV","VH","VV_VH"]
S1 = ["VV","VH","VV_VH"]
S2 = ["NDVI","EVI","NDWI"]

DEVICE = torch.device(
    "mps" if torch.backends.mps.is_available() else "cpu"
)

ckpt = torch.load(
    MODEL,
    map_location="cpu",
    weights_only=False
)

state_mean = pd.Series(ckpt["state_mean"])
state_std = pd.Series(ckpt["state_std"])


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
        return z, self.s1_head(z), self.s2_head(z)


model = LatentObservationModel().to(DEVICE)

model.load_state_dict(
    ckpt["model_state_dict"]
)

model.eval()

df = pd.read_csv(DATA)
df["date"] = pd.to_datetime(df["date"])

valid = df[
    (df["date"] >= "2023-01-01") &
    (df["date"] <= "2023-12-01")
].copy()

valid = valid.dropna(
    subset=STATE
).copy()

X = (
    (valid[STATE] - state_mean)
    / state_std
).values.astype(np.float32)

X_t = torch.tensor(
    X,
    dtype=torch.float32,
    device=DEVICE
)

s1_idx = [STATE.index(x) for x in S1]
s2_idx = [STATE.index(x) for x in S2]

with torch.no_grad():

    z, pred_s1, pred_s2 = model(X_t)

z = z.cpu().numpy()
pred_s1 = pred_s1.cpu().numpy()
pred_s2 = pred_s2.cpu().numpy()

true_s1 = X[:,s1_idx]
true_s2 = X[:,s2_idx]

print("\n======================================")
print("LATENT OBSERVATION V1 EVALUATION")
print("======================================")

print("Validation rows:", len(valid))
print("Latent dimensions:", z.shape[1])

print("\nS1 RMSE:")

for i,f in enumerate(S1):

    rmse = np.sqrt(
        np.mean(
            (pred_s1[:,i] - true_s1[:,i])**2
        )
    )

    print(
        f"{f}: {rmse:.6f}"
    )

print("\nS2 RMSE:")

for i,f in enumerate(S2):

    rmse = np.sqrt(
        np.mean(
            (pred_s2[:,i] - true_s2[:,i])**2
        )
    )

    print(
        f"{f}: {rmse:.6f}"
    )

print("\nMean S1 RMSE:")
print(
    np.sqrt(
        np.mean(
            (pred_s1 - true_s1)**2
        )
    )
)

print("\nMean S2 RMSE:")
print(
    np.sqrt(
        np.mean(
            (pred_s2 - true_s2)**2
        )
    )
)

print("\nLatent state statistics:")

for i in range(4):

    print(
        f"z{i+1}: "
        f"mean={z[:,i].mean():.4f}, "
        f"std={z[:,i].std():.4f}, "
        f"min={z[:,i].min():.4f}, "
        f"max={z[:,i].max():.4f}"
    )
