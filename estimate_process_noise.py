import pandas as pd
import numpy as np
import torch
import torch.nn as nn

DATA = "GLOBAL_ALL25_MODEL_READY.csv"
MODEL = "models/GLOBAL_5CROP_TEMPORAL_TWIN.pt"

STATE = ["NDVI","EVI","NDWI","VV","VH","VV_VH"]
WEATHER = ["temperature_C","precipitation_mm","solar_MJ_m2"]

ckpt = torch.load(MODEL, map_location="cpu", weights_only=False)

state_mean = pd.Series(ckpt["state_mean"])
state_std = pd.Series(ckpt["state_std"])
weather_mean = pd.Series(ckpt["weather_mean"])
weather_std = pd.Series(ckpt["weather_std"])
crop_to_id = ckpt["crop_to_id"]

class Twin(nn.Module):
    def __init__(self):
        super().__init__()
        self.crop_embedding = nn.Embedding(5, 4)
        self.transition = nn.Sequential(
            nn.Linear(13, 32),
            nn.Tanh(),
            nn.Linear(32, 32),
            nn.Tanh(),
            nn.Linear(32, 6)
        )

    def forward(self, state, weather, crop):
        emb = self.crop_embedding(crop)
        x = torch.cat([state, weather, emb], dim=1)
        return self.transition(x)

model = Twin()
model.load_state_dict(ckpt["model_state_dict"])
model.eval()

df = pd.read_csv(DATA)
df["date"] = pd.to_datetime(df["date"])
df = df.sort_values(["site_id","date"])

residuals = []

with torch.no_grad():

    for site, g in df.groupby("site_id"):

        g = g.sort_values("date").reset_index(drop=True)

        for i in range(1, len(g)):

            prev = g.iloc[i-1]
            cur = g.iloc[i]

            if cur["date"] != prev["date"] + pd.DateOffset(months=1):
                continue

            if (
                prev[STATE].isna().any() or
                cur[STATE].isna().any() or
                cur[WEATHER].isna().any()
            ):
                continue

            s_prev = (
                (prev[STATE] - state_mean) / state_std
            ).values.astype(np.float32)

            s_cur = (
                (cur[STATE] - state_mean) / state_std
            ).values.astype(np.float32)

            w = (
                (cur[WEATHER] - weather_mean) / weather_std
            ).values.astype(np.float32)

            crop = torch.tensor(
                [crop_to_id[cur["crop"]]],
                dtype=torch.long
            )

            pred_delta = model(
                torch.tensor(s_prev).unsqueeze(0),
                torch.tensor(w).unsqueeze(0),
                crop
            ).numpy()[0]

            pred = s_prev + pred_delta

            residuals.append(pred - s_cur)

R = np.asarray(residuals)

print("\n==========================================")
print("EMPIRICAL PROCESS NOISE")
print("==========================================")

print("Valid transitions:", len(R))

for i, feature in enumerate(STATE):
    sd = np.std(R[:, i], ddof=1)
    var = sd**2

    print(
        f"{feature:7s}  "
        f"std={sd:.6f}  "
        f"variance={var:.6f}"
    )

print("\nDiagonal process-noise covariance:")
print(np.diag(np.var(R, axis=0, ddof=1)))

np.save(
    "GLOBAL_EMPIRICAL_PROCESS_NOISE.npy",
    np.var(R, axis=0, ddof=1)
)

print("\nSaved: GLOBAL_EMPIRICAL_PROCESS_NOISE.npy")
