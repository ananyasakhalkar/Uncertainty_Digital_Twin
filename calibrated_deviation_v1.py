import pandas as pd
import numpy as np
import torch
import torch.nn as nn

DEVICE = torch.device(
    "mps" if torch.backends.mps.is_available() else "cpu"
)

DATA = "GLOBAL_ALL25_MODEL_READY.csv"
OBS_MODEL = "models/GLOBAL_LATENT_OBSERVATION_V1.pt"
TRANS_MODEL = "models/GLOBAL_LATENT_TRANSITION_V1.pt"

STATE = ["NDVI","EVI","NDWI","VV","VH","VV_VH"]
WEATHER = [
    "temperature_C",
    "precipitation_mm",
    "solar_MJ_m2"
]

obs_ckpt = torch.load(
    OBS_MODEL,
    map_location="cpu",
    weights_only=False
)

trans_ckpt = torch.load(
    TRANS_MODEL,
    map_location="cpu",
    weights_only=False
)

state_mean = pd.Series(
    trans_ckpt["state_mean"]
)

state_std = pd.Series(
    trans_ckpt["state_std"]
)

weather_mean = np.array([
    trans_ckpt["weather_mean"][f]
    for f in WEATHER
], dtype=np.float32)

weather_std = np.array([
    trans_ckpt["weather_std"][f]
    for f in WEATHER
], dtype=np.float32)

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
        "encoder.0.",
        "encoder.net.0."
    ).replace(
        "encoder.2.",
        "encoder.net.2."
    ): v
    for k,v in obs_ckpt[
        "model_state_dict"
    ].items()
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


def norm_state(x):

    return (
        x - state_mean.values
    ) / state_std.values


def norm_weather(x):

    return (
        x - weather_mean
    ) / weather_std


def predict_one(
    cur,
    prev,
    prev2
):

    previous = norm_state(
        prev[STATE].values.astype(
            np.float32
        )
    )

    with torch.no_grad():

        x = torch.tensor(
            previous,
            dtype=torch.float32,
            device=DEVICE
        ).unsqueeze(0)

        z = obs_model.encoder(x)

    weather = np.concatenate([
        norm_weather(
            cur[WEATHER].values.astype(
                np.float32
            )
        ),
        norm_weather(
            prev[WEATHER].values.astype(
                np.float32
            )
        ),
        norm_weather(
            prev2[WEATHER].values.astype(
                np.float32
            )
        )
    ])

    w = torch.tensor(
        weather,
        dtype=torch.float32,
        device=DEVICE
    ).unsqueeze(0)

    crop = torch.tensor(
        [crop_to_id[cur["crop"]]],
        dtype=torch.long,
        device=DEVICE
    )

    with torch.no_grad():

        delta = transition(
            z,w,crop
        )

        z_next = z + delta

        s2 = obs_model.s2_head(
            z_next
        )

        s1 = obs_model.s1_head(
            z_next
        )

    pred = torch.cat(
        [s2,s1],
        dim=1
    ).cpu().numpy()[0]

    return pred


rows = []


# ---------------------------------------------------------
# 1. HISTORICAL CALIBRATION: 2019-2023
# ---------------------------------------------------------

for site,g in df.groupby("site_id"):

    g = g.sort_values(
        "date"
    ).reset_index(drop=True)

    for i in range(2,len(g)):

        cur = g.iloc[i]
        prev = g.iloc[i-1]
        prev2 = g.iloc[i-2]

        if cur["date"] > pd.Timestamp(
            "2023-12-01"
        ):
            continue

        if cur["date"] < pd.Timestamp(
            "2019-01-01"
        ):
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

        pred = predict_one(
            cur,
            prev,
            prev2
        )

        obs = norm_state(
            cur[STATE].values.astype(
                np.float32
            )
        )

        error = np.abs(
            pred - obs
        )

        for j,f in enumerate(STATE):

            rows.append({
                "period": "historical",
                "site_id": site,
                "crop": cur["crop"],
                "country": cur["country"],
                "date": cur["date"],
                "feature": f,
                "absolute_error": error[j]
            })


historical = pd.DataFrame(rows)


# ---------------------------------------------------------
# 2. HISTORICAL ERROR SCALE
# ---------------------------------------------------------

calibration = (
    historical
    .groupby("feature")["absolute_error"]
    .agg(
        median="median",
        mad=lambda x:
            np.median(
                np.abs(
                    x - np.median(x)
                )
            ),
        p90=lambda x:
            np.quantile(x,0.90),
        p95=lambda x:
            np.quantile(x,0.95)
    )
    .reset_index()
)

# Robust scale.
# 1.4826 converts MAD to a Gaussian-equivalent
# standard-deviation scale.

calibration["robust_scale"] = (
    1.4826 * calibration["mad"]
)

print(
    "\n=========================================="
)

print(
    "HISTORICAL ERROR CALIBRATION"
)

print(
    "=========================================="
)

print(calibration)


# ---------------------------------------------------------
# 3. 2024 DEVIATIONS
# ---------------------------------------------------------

results = []


for site,g in df.groupby("site_id"):

    g = g.sort_values(
        "date"
    ).reset_index(drop=True)

    for i in range(2,len(g)):

        cur = g.iloc[i]
        prev = g.iloc[i-1]
        prev2 = g.iloc[i-2]

        if cur["date"] < pd.Timestamp(
            "2024-01-01"
        ):
            continue

        if cur["date"] > pd.Timestamp(
            "2024-10-01"
        ):
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

        pred = predict_one(
            cur,
            prev,
            prev2
        )

        obs = norm_state(
            cur[STATE].values.astype(
                np.float32
            )
        )

        error = np.abs(
            pred - obs
        )

        for j,f in enumerate(STATE):

            scale = float(
                calibration.loc[
                    calibration["feature"] == f,
                    "robust_scale"
                ].iloc[0]
            )

            p95 = float(
                calibration.loc[
                    calibration["feature"] == f,
                    "p95"
                ].iloc[0]
            )

            results.append({
                "site_id": site,
                "crop": cur["crop"],
                "country": cur["country"],
                "date": cur["date"],
                "feature": f,
                "absolute_deviation": error[j],
                "historical_robust_scale": scale,
                "historical_p95_error": p95,
                "robust_deviation_score":
                    error[j] / (scale + 1e-6),
                "above_historical_p95":
                    bool(error[j] > p95)
            })


res = pd.DataFrame(results)


print(
    "\n=========================================="
)

print(
    "CALIBRATED 2024 DEVIATIONS"
)

print(
    "=========================================="
)

print(
    "Rows:",
    len(res)
)

print(
    "Sites:",
    res.site_id.nunique()
)


print(
    "\nMean robust deviation score:"
)

print(
    res.groupby("feature")
    ["robust_deviation_score"]
    .mean()
)


print(
    "\n95th percentile robust score:"
)

print(
    res.groupby("feature")
    ["robust_deviation_score"]
    .quantile(0.95)
)


print(
    "\n2024 observations above historical 95th error:"
)

print(
    res.groupby("feature")
    ["above_historical_p95"]
    .mean()
)


high = res[
    res["above_historical_p95"]
].copy()

high = high.sort_values(
    "robust_deviation_score",
    ascending=False
)


print(
    "\nTop 25 calibrated deviations:"
)

print(
    high[
        [
            "site_id",
            "crop",
            "country",
            "date",
            "feature",
            "absolute_deviation",
            "historical_p95_error",
            "robust_deviation_score"
        ]
    ]
    .head(25)
    .to_string(index=False)
)


site_summary = (
    res.groupby(
        ["site_id","crop","country"]
    )
    .agg(
        mean_deviation=(
            "absolute_deviation",
            "mean"
        ),
        mean_robust_score=(
            "robust_deviation_score",
            "mean"
        ),
        above_p95_count=(
            "above_historical_p95",
            "sum"
        )
    )
    .reset_index()
    .sort_values(
        "mean_robust_score",
        ascending=False
    )
)


print(
    "\nSite summary:"
)

print(
    site_summary.head(15)
    .to_string(index=False)
)


calibration.to_csv(
    "GLOBAL_LATENT_V1_HISTORICAL_CALIBRATION.csv",
    index=False
)

res.to_csv(
    "GLOBAL_LATENT_V1_2024_CALIBRATED_DEVIATIONS.csv",
    index=False
)

site_summary.to_csv(
    "GLOBAL_LATENT_V1_2024_CALIBRATED_SITE_SUMMARY.csv",
    index=False
)

print(
    "\nSaved calibrated deviation results."
)
