import pandas as pd
import numpy as np
import torch
import torch.nn as nn

DEVICE = torch.device(
    "mps" if torch.backends.mps.is_available() else "cpu"
)

DATA = "GLOBAL_ALL25_MODEL_READY.csv"
MODEL = "models/GLOBAL_LATENT_TRANSITION_V1.pt"
OBS_MODEL = "models/GLOBAL_LATENT_OBSERVATION_V1.pt"

STATE = ["NDVI","EVI","NDWI","VV","VH","VV_VH"]
S1 = ["VV","VH","VV_VH"]
S2 = ["NDVI","EVI","NDWI"]
WEATHER = ["temperature_C","precipitation_mm","solar_MJ_m2"]


obs_ckpt = torch.load(
    OBS_MODEL,
    map_location="cpu",
    weights_only=False
)

trans_ckpt = torch.load(
    MODEL,
    map_location="cpu",
    weights_only=False
)

state_mean = pd.Series(
    trans_ckpt["state_mean"]
)
state_std = pd.Series(
    trans_ckpt["state_std"]
)

weather_mean = np.array(
    [trans_ckpt["weather_mean"][f] for f in WEATHER],
    dtype=np.float32
)

weather_std = np.array(
    [trans_ckpt["weather_std"][f] for f in WEATHER],
    dtype=np.float32
)

crop_to_id = trans_ckpt["crop_to_id"]


class Encoder(nn.Module):

    def __init__(self):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(6,32),
            nn.Tanh(),
            nn.Linear(32,4)
        )

    def forward(self,x):
        return self.net(x)


class ObservationModel(nn.Module):

    def __init__(self):
        super().__init__()

        self.encoder = Encoder()

        self.s1_head = nn.Sequential(
            nn.Linear(4,16),
            nn.Tanh(),
            nn.Linear(16,3)
        )

        self.s2_head = nn.Sequential(
            nn.Linear(4,16),
            nn.Tanh(),
            nn.Linear(16,3)
        )

    def forward(self,x):

        z = self.encoder(x)

        return (
            z,
            self.s1_head(z),
            self.s2_head(z)
        )


class LatentTransition(nn.Module):

    def __init__(self):
        super().__init__()

        self.crop_embedding = nn.Embedding(5,4)

        self.net = nn.Sequential(
            nn.Linear(17,32),
            nn.Tanh(),
            nn.Linear(32,32),
            nn.Tanh(),
            nn.Linear(32,4)
        )

    def forward(self,z,weather,crop):

        emb = self.crop_embedding(crop)

        x = torch.cat(
            [z,weather,emb],
            dim=1
        )

        return self.net(x)


obs_model = ObservationModel().to(DEVICE)

obs_model.load_state_dict({
    k.replace(
        "encoder.0.", "encoder.net.0."
    ).replace(
        "encoder.2.", "encoder.net.2."
    ): v
    for k,v in obs_ckpt["model_state_dict"].items()
})

obs_model.eval()


transition = LatentTransition().to(DEVICE)

transition.load_state_dict(
    trans_ckpt["model_state_dict"]
)

transition.eval()


df = pd.read_csv(DATA)

df["date"] = pd.to_datetime(
    df["date"]
)

df = df.sort_values(
    ["site_id","date"]
).reset_index(drop=True)


results = []


with torch.no_grad():

    for site,g in df.groupby("site_id"):

        g = g.sort_values(
            "date"
        ).reset_index(drop=True)

        for i in range(2,len(g)):

            cur = g.iloc[i]
            prev = g.iloc[i-1]
            prev2 = g.iloc[i-2]

            if cur["date"] < pd.Timestamp("2023-01-01"):
                continue

            if cur["date"] > pd.Timestamp("2023-12-01"):
                continue

            if (
                prev["date"]
                != cur["date"]
                - pd.DateOffset(months=1)
                or
                prev2["date"]
                != cur["date"]
                - pd.DateOffset(months=2)
            ):
                continue

            if (
                prev[STATE].isna().any()
                or cur[STATE].isna().any()
                or prev[WEATHER].isna().any()
                or prev2[WEATHER].isna().any()
                or cur[WEATHER].isna().any()
            ):
                continue

            previous_state = (
                (
                    prev[STATE].values.astype(
                        np.float32
                    )
                    -
                    state_mean.values
                )
                /
                state_std.values
            )

            target_state = (
                (
                    cur[STATE].values.astype(
                        np.float32
                    )
                    -
                    state_mean.values
                )
                /
                state_std.values
            )

            w0 = (
                (
                    cur[WEATHER].values.astype(
                        np.float32
                    )
                    -
                    weather_mean
                )
                /
                weather_std
            )

            w1 = (
                (
                    prev[WEATHER].values.astype(
                        np.float32
                    )
                    -
                    weather_mean
                )
                /
                weather_std
            )

            w2 = (
                (
                    prev2[WEATHER].values.astype(
                        np.float32
                    )
                    -
                    weather_mean
                )
                /
                weather_std
            )

            weather_history = np.concatenate(
                [w0,w1,w2]
            )

            x = torch.tensor(
                previous_state,
                dtype=torch.float32,
                device=DEVICE
            ).unsqueeze(0)

            weather = torch.tensor(
                weather_history,
                dtype=torch.float32,
                device=DEVICE
            ).unsqueeze(0)

            crop = torch.tensor(
                [crop_to_id[cur["crop"]]],
                dtype=torch.long,
                device=DEVICE
            )

            z = obs_model.encoder(x)

            delta = transition(
                z,
                weather,
                crop
            )

            z_next = z + delta

            pred_s1 = obs_model.s1_head(
                z_next
            ).cpu().numpy()[0]

            pred_s2 = obs_model.s2_head(
                z_next
            ).cpu().numpy()[0]

            pred = np.zeros(6)

            pred[
                [STATE.index(x) for x in S1]
            ] = pred_s1

            pred[
                [STATE.index(x) for x in S2]
            ] = pred_s2

            error = pred - target_state

            for j,f in enumerate(STATE):

                results.append({
                    "site_id": site,
                    "crop": cur["crop"],
                    "country": cur["country"],
                    "date": cur["date"],
                    "feature": f,
                    "error": error[j],
                    "squared_error": error[j]**2
                })


res = pd.DataFrame(results)

print("\n======================================")
print("LATENT TRANSITION V1")
print("2023 OBSERVATION-SPACE EVALUATION")
print("======================================")

print("Rows:",len(res))
print("Sites:",res.site_id.nunique())

feature_rmse = (
    res.groupby("feature")["squared_error"]
    .mean()
    .pow(0.5)
)

print("\nFeature RMSE:")
print(feature_rmse)

overall = np.sqrt(
    res["squared_error"].mean()
)

print("\nOverall RMSE:")
print(overall)

crop_rmse = (
    res.groupby("crop")["squared_error"]
    .mean()
    .pow(0.5)
)

print("\nCrop RMSE:")
print(crop_rmse)

res.to_csv(
    "GLOBAL_LATENT_TRANSITION_V1_2023_RESULTS.csv",
    index=False
)

print(
    "\nSaved: "
    "GLOBAL_LATENT_TRANSITION_V1_2023_RESULTS.csv"
)
