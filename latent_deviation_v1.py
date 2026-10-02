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
S1 = ["VV","VH","VV_VH"]
S2 = ["NDVI","EVI","NDWI"]
WEATHER = [
    "temperature_C",
    "precipitation_mm",
    "solar_MJ_m2"
]

N_ENSEMBLE = 50
PROCESS_NOISE = 0.05
EPS = 1e-6


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

        self.crop_embedding = nn.Embedding(
            5,4
        )

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


def predict_observations(
    latent_ensemble
):

    z = torch.tensor(
        latent_ensemble,
        dtype=torch.float32,
        device=DEVICE
    )

    with torch.no_grad():

        s1 = obs_model.s1_head(z)
        s2 = obs_model.s2_head(z)

    return np.concatenate(
        [
            s2.cpu().numpy(),
            s1.cpu().numpy()
        ],
        axis=1
    )


def propagate(
    ensemble,
    weather,
    crop_id
):

    z = torch.tensor(
        ensemble,
        dtype=torch.float32,
        device=DEVICE
    )

    w = torch.tensor(
        np.repeat(
            weather[None,:],
            len(ensemble),
            axis=0
        ),
        dtype=torch.float32,
        device=DEVICE
    )

    c = torch.tensor(
        np.repeat(
            crop_id,
            len(ensemble)
        ),
        dtype=torch.long,
        device=DEVICE
    )

    with torch.no_grad():

        delta = transition(
            z,w,c
        ).cpu().numpy()

    noise = np.random.normal(
        0,
        PROCESS_NOISE,
        size=delta.shape
    )

    return ensemble + delta + noise


rows = []


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
            or prev2[STATE].isna().any()
            or prev[WEATHER].isna().any()
            or prev2[WEATHER].isna().any()
            or cur[WEATHER].isna().any()
        ):
            continue

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

            z0 = (
                obs_model.encoder(x)
                .cpu()
                .numpy()[0]
            )

        ensemble = np.repeat(
            z0[None,:],
            N_ENSEMBLE,
            axis=0
        )

        ensemble += np.random.normal(
            0,
            PROCESS_NOISE,
            ensemble.shape
        )

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

        ensemble = propagate(
            ensemble,
            weather,
            crop_to_id[cur["crop"]]
        )

        predicted = predict_observations(
            ensemble
        )

        forecast = predicted.mean(
            axis=0
        )

        spread = predicted.std(
            axis=0
        )

        observed = norm_state(
            cur[STATE].values.astype(
                np.float32
            )
        )

        for j,f in enumerate(STATE):

            if pd.isna(cur[f]):
                continue

            abs_dev = abs(
                forecast[j] - observed[j]
            )

            standardized_dev = (
                abs_dev /
                (spread[j] + EPS)
            )

            rows.append({
                "site_id": site,
                "crop": cur["crop"],
                "country": cur["country"],
                "date": cur["date"],
                "feature": f,
                "forecast": forecast[j],
                "observation": observed[j],
                "ensemble_spread": spread[j],
                "absolute_deviation": abs_dev,
                "uncertainty_aware_deviation":
                    standardized_dev
            })


res = pd.DataFrame(rows)

print(
    "\n=========================================="
)

print(
    "LATENT V1 2024 DEVIATION ANALYSIS"
)

print(
    "=========================================="
)

print("Rows:",len(res))
print("Sites:",res.site_id.nunique())


print("\nMean absolute deviation:")

print(
    res.groupby("feature")
    ["absolute_deviation"]
    .mean()
)


print(
    "\n95th percentile absolute deviation:"
)

print(
    res.groupby("feature")
    ["absolute_deviation"]
    .quantile(0.95)
)


print(
    "\nMean uncertainty-aware deviation:"
)

print(
    res.groupby("feature")
    ["uncertainty_aware_deviation"]
    .mean()
)


print(
    "\n95th percentile uncertainty-aware deviation:"
)

thresholds = (
    res.groupby("feature")
    ["uncertainty_aware_deviation"]
    .quantile(0.95)
)

print(thresholds)


res["high_deviation"] = False

for f in STATE:

    mask = res["feature"] == f

    res.loc[
        mask,
        "high_deviation"
    ] = (
        res.loc[
            mask,
            "uncertainty_aware_deviation"
        ]
        >= thresholds[f]
    )


high = res[
    res["high_deviation"]
].copy()

high = high.sort_values(
    "uncertainty_aware_deviation",
    ascending=False
)

print(
    "\nTop 20 uncertainty-aware deviations:"
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
            "ensemble_spread",
            "uncertainty_aware_deviation"
        ]
    ].head(20).to_string(index=False)
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
        mean_uncertainty_aware_deviation=(
            "uncertainty_aware_deviation",
            "mean"
        ),
        high_deviation_count=(
            "high_deviation",
            "sum"
        )
    )
    .reset_index()
    .sort_values(
        "mean_uncertainty_aware_deviation",
        ascending=False
    )
)

print(
    "\nSite summary:"
)

print(
    site_summary.head(15).to_string(
        index=False
    )
)


res.to_csv(
    "GLOBAL_LATENT_V1_2024_DEVIATION.csv",
    index=False
)

site_summary.to_csv(
    "GLOBAL_LATENT_V1_2024_SITE_DEVIATION.csv",
    index=False
)

print(
    "\nSaved deviation results."
)
