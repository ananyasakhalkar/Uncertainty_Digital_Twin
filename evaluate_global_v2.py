import pandas as pd
import numpy as np
import torch
import torch.nn as nn

DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

DATA = "GLOBAL_ALL25_MODEL_READY.csv"
MODEL = "models/GLOBAL_5CROP_TEMPORAL_TWIN_V2.pt"

STATE = ["NDVI","EVI","NDWI","VV","VH","VV_VH"]
WEATHER = ["temperature_C","precipitation_mm","solar_MJ_m2"]

ckpt = torch.load(MODEL, map_location="cpu", weights_only=False)

state_mean = pd.Series(ckpt["state_mean"])
state_std = pd.Series(ckpt["state_std"])
weather_mean = pd.Series(ckpt["weather_mean"])
weather_std = pd.Series(ckpt["weather_std"])

crop_to_id = ckpt["crop_to_id"]

class TwinV2(nn.Module):
    def __init__(self):
        super().__init__()
        self.crop_embedding = nn.Embedding(5, 4)
        self.transition = nn.Sequential(
            nn.Linear(21, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 6)
        )

    def forward(self, state, delta_prev, weather, crop, sin_m, cos_m):
        emb = self.crop_embedding(crop)
        x = torch.cat([
            state, delta_prev, weather,
            emb, sin_m, cos_m
        ], dim=1)
        return self.transition(x)

model = TwinV2().to(DEVICE)
model.load_state_dict(ckpt["model_state_dict"])
model.eval()

df = pd.read_csv(DATA)
df["date"] = pd.to_datetime(df["date"])
df = df.sort_values(["site_id","date"])

# Only genuine 2024 transitions
test = df[
    (df["date"] >= "2024-01-01") &
    (df["date"] <= "2024-10-01")
].copy()

results = []

with torch.no_grad():

    for site, g in df.groupby("site_id"):

        g = g.sort_values("date").reset_index(drop=True)

        for i in range(1, len(g)):

            prev = g.iloc[i-1]
            cur = g.iloc[i]

            if cur["date"] < pd.Timestamp("2024-01-01"):
                continue
            if cur["date"] > pd.Timestamp("2024-10-01"):
                continue

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

            # Approximate previous delta from the two available states
            # when the preceding transition is available.
            if i < 2:
                continue

            prev2 = g.iloc[i-2]

            if (
                prev2["date"] != prev["date"] - pd.DateOffset(months=1)
                or prev2[STATE].isna().any()
            ):
                continue

            s_prev2 = (
                (prev2[STATE] - state_mean) / state_std
            ).values.astype(np.float32)

            delta_prev = s_prev - s_prev2

            weather = (
                (cur[WEATHER] - weather_mean) / weather_std
            ).values.astype(np.float32)

            month = cur["date"].month
            sin_m = np.sin(2*np.pi*month/12)
            cos_m = np.cos(2*np.pi*month/12)

            state_t = torch.tensor(
                s_prev, dtype=torch.float32
            ).unsqueeze(0).to(DEVICE)

            delta_t = torch.tensor(
                delta_prev, dtype=torch.float32
            ).unsqueeze(0).to(DEVICE)

            weather_t = torch.tensor(
                weather, dtype=torch.float32
            ).unsqueeze(0).to(DEVICE)

            crop_t = torch.tensor(
                [crop_to_id[cur["crop"]]],
                dtype=torch.long
            ).to(DEVICE)

            sin_t = torch.tensor(
                [[sin_m]], dtype=torch.float32
            ).to(DEVICE)

            cos_t = torch.tensor(
                [[cos_m]], dtype=torch.float32
            ).to(DEVICE)

            pred_delta = model(
                state_t,
                delta_t,
                weather_t,
                crop_t,
                sin_t,
                cos_t
            ).cpu().numpy()[0]

            pred = s_prev + pred_delta

            error = pred - s_cur

            for j, feature in enumerate(STATE):
                results.append({
                    "site_id": site,
                    "crop": cur["crop"],
                    "country": cur["country"],
                    "date": cur["date"],
                    "feature": feature,
                    "prediction": pred[j],
                    "observation": s_cur[j],
                    "error": error[j],
                    "squared_error": error[j]**2
                })

res = pd.DataFrame(results)

print("\n==========================================")
print("GLOBAL V2 2024 ONE-STEP EVALUATION")
print("==========================================")

print("Transitions:", res.date.nunique())
print("Sites:", res.site_id.nunique())
print("Rows:", len(res))

print("\nFeature RMSE:")
rmse = (
    res.groupby("feature")["squared_error"]
       .mean()
       .pow(0.5)
)
print(rmse)

print("\nOverall RMSE:")
print(np.sqrt(res["squared_error"].mean()))

print("\nCrop RMSE:")
print(
    res.groupby("crop")["squared_error"]
       .mean()
       .pow(0.5)
)

res.to_csv("GLOBAL_V2_2024_ONE_STEP_RESULTS.csv", index=False)

print("\nSaved: GLOBAL_V2_2024_ONE_STEP_RESULTS.csv")
