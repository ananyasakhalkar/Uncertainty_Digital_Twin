import pandas as pd
import numpy as np
import torch
import torch.nn as nn

DEVICE = torch.device(
    "mps" if torch.backends.mps.is_available() else "cpu"
)

DATA = "GLOBAL_ALL25_MODEL_READY.csv"
MODEL = "models/GLOBAL_5CROP_WEATHER_V3.pt"

STATE = ["NDVI","EVI","NDWI","VV","VH","VV_VH"]
WEATHER = ["temperature_C","precipitation_mm","solar_MJ_m2"]

ckpt = torch.load(
    MODEL,
    map_location="cpu",
    weights_only=False
)

state_mean = pd.Series(ckpt["state_mean"])
state_std = pd.Series(ckpt["state_std"])
weather_mean = pd.Series(ckpt["weather_mean"])
weather_std = pd.Series(ckpt["weather_std"])
crop_to_id = ckpt["crop_to_id"]


class WeatherTwinV3(nn.Module):

    def __init__(self):
        super().__init__()

        self.crop_embedding = nn.Embedding(5, 4)

        self.transition = nn.Sequential(
            nn.Linear(19, 32),
            nn.Tanh(),
            nn.Linear(32, 32),
            nn.Tanh(),
            nn.Linear(32, 6)
        )

    def forward(self, state, weather, crop):
        emb = self.crop_embedding(crop)

        x = torch.cat([
            state,
            weather,
            emb
        ], dim=1)

        return self.transition(x)


model = WeatherTwinV3().to(DEVICE)
model.load_state_dict(
    ckpt["model_state_dict"]
)
model.eval()

df = pd.read_csv(DATA)
df["date"] = pd.to_datetime(df["date"])
df = df.sort_values(["site_id","date"])

results = []

with torch.no_grad():

    for site, g in df.groupby("site_id"):

        g = g.sort_values("date").reset_index(drop=True)

        for i in range(2, len(g)):

            prev = g.iloc[i-1]
            cur = g.iloc[i]
            prev2 = g.iloc[i-2]

            if cur["date"] < pd.Timestamp("2024-01-01"):
                continue

            if cur["date"] > pd.Timestamp("2024-10-01"):
                continue

            if (
                prev["date"] != cur["date"]
                - pd.DateOffset(months=1)
                or
                prev2["date"] != cur["date"]
                - pd.DateOffset(months=2)
            ):
                continue

            if (
                prev[STATE].isna().any()
                or cur[STATE].isna().any()
                or cur[WEATHER].isna().any()
                or prev[WEATHER].isna().any()
                or prev2[WEATHER].isna().any()
            ):
                continue

            s_prev = (
                (prev[STATE] - state_mean)
                / state_std
            ).values.astype(np.float32)

            s_cur = (
                (cur[STATE] - state_mean)
                / state_std
            ).values.astype(np.float32)

            w0 = (
                (cur[WEATHER] - weather_mean)
                / weather_std
            ).values.astype(np.float32)

            w1 = (
                (prev[WEATHER] - weather_mean)
                / weather_std
            ).values.astype(np.float32)

            w2 = (
                (prev2[WEATHER] - weather_mean)
                / weather_std
            ).values.astype(np.float32)

            weather_history = np.concatenate([
                w0, w1, w2
            ])

            state_t = torch.tensor(
                s_prev,
                dtype=torch.float32
            ).unsqueeze(0).to(DEVICE)

            weather_t = torch.tensor(
                weather_history,
                dtype=torch.float32
            ).unsqueeze(0).to(DEVICE)

            crop_t = torch.tensor(
                [crop_to_id[cur["crop"]]],
                dtype=torch.long
            ).to(DEVICE)

            pred_delta = model(
                state_t,
                weather_t,
                crop_t
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
                    "squared_error": error[j] ** 2
                })

res = pd.DataFrame(results)

print("\n==========================================")
print("GLOBAL WEATHER V3 2024 EVALUATION")
print("==========================================")

print("Rows:", len(res))
print("Sites:", res.site_id.nunique())

print("\nFeature RMSE:")
print(
    res.groupby("feature")["squared_error"]
       .mean()
       .pow(0.5)
)

print("\nOverall RMSE:")
print(
    np.sqrt(res["squared_error"].mean())
)

print("\nCrop RMSE:")
print(
    res.groupby("crop")["squared_error"]
       .mean()
       .pow(0.5)
)

res.to_csv(
    "GLOBAL_WEATHER_V3_2024_RESULTS.csv",
    index=False
)

print(
    "\nSaved: "
    "GLOBAL_WEATHER_V3_2024_RESULTS.csv"
)
