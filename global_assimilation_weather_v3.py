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
S1 = ["VV","VH","VV_VH"]
S2 = ["NDVI","EVI","NDWI"]

N_ENSEMBLE = 50
PROCESS_NOISE = 0.05
OBS_NOISE = 0.15

ckpt = torch.load(
    MODEL,
    map_location="cpu",
    weights_only=False
)

state_mean = np.array(
    [ckpt["state_mean"][f] for f in STATE],
    dtype=np.float32
)
state_std = np.array(
    [ckpt["state_std"][f] for f in STATE],
    dtype=np.float32
)

weather_mean = np.array(
    [ckpt["weather_mean"][f] for f in WEATHER],
    dtype=np.float32
)
weather_std = np.array(
    [ckpt["weather_std"][f] for f in WEATHER],
    dtype=np.float32
)

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

df = df.sort_values(
    ["site_id","date"]
).reset_index(drop=True)


def normalize_state(x):
    return (x - state_mean) / state_std


def denormalize_state(x):
    return x * state_std + state_mean


def normalize_weather(x):
    return (x - weather_mean) / weather_std


def weather_history(g, i):

    cur = g.iloc[i]
    prev = g.iloc[i-1]
    prev2 = g.iloc[i-2]

    w0 = normalize_weather(
        cur[WEATHER].values.astype(np.float32)
    )

    w1 = normalize_weather(
        prev[WEATHER].values.astype(np.float32)
    )

    w2 = normalize_weather(
        prev2[WEATHER].values.astype(np.float32)
    )

    return np.concatenate([w0,w1,w2])


def propagate(ensemble, weather_hist, crop_id):

    n = ensemble.shape[0]

    state_t = torch.tensor(
        ensemble,
        dtype=torch.float32,
        device=DEVICE
    )

    weather_t = torch.tensor(
        np.repeat(
            weather_hist[None,:],
            n,
            axis=0
        ),
        dtype=torch.float32,
        device=DEVICE
    )

    crop_t = torch.tensor(
        np.repeat(crop_id,n),
        dtype=torch.long,
        device=DEVICE
    )

    with torch.no_grad():

        delta = model(
            state_t,
            weather_t,
            crop_t
        ).cpu().numpy()

    noise = np.random.normal(
        0,
        PROCESS_NOISE,
        size=delta.shape
    )

    return ensemble + delta + noise


def assimilate(
    ensemble,
    observation,
    obs_indices
):

    if len(obs_indices) == 0:
        return ensemble

    obs_indices = np.array(
        obs_indices,
        dtype=int
    )

    X = ensemble

    x_mean = X.mean(axis=0)

    X_anom = X - x_mean

    Y = X[:,obs_indices]

    y_mean = Y.mean(axis=0)

    Y_anom = Y - y_mean

    Pxy = (
        X_anom.T @ Y_anom
    ) / (len(X)-1)

    Pyy = (
        Y_anom.T @ Y_anom
    ) / (len(X)-1)

    R = (
        OBS_NOISE**2
        * np.eye(len(obs_indices))
    )

    K = Pxy @ np.linalg.inv(
        Pyy + R
    )

    perturbed_obs = (
        observation[obs_indices]
        + np.random.normal(
            0,
            OBS_NOISE,
            size=len(obs_indices)
        )
    )

    innovation = (
        perturbed_obs
        - Y
    )

    return X + innovation @ K.T


modes = {
    "free_running": [],
    "S1_only": S1,
    "S2_only": S2,
    "S1_plus_S2": STATE
}

all_results = []


for mode, obs_features in modes.items():

    print("\nMODE:", mode)

    for site, g in df.groupby("site_id"):

        g = g.sort_values("date").reset_index(drop=True)

        # Need two previous months for weather history.
        for i in range(2,len(g)):

            cur = g.iloc[i]
            prev = g.iloc[i-1]
            prev2 = g.iloc[i-2]

            if cur["date"] < pd.Timestamp("2024-01-01"):
                continue

            if cur["date"] > pd.Timestamp("2024-10-01"):
                continue

            if (
                prev["date"]
                != cur["date"] - pd.DateOffset(months=1)
                or
                prev2["date"]
                != cur["date"] - pd.DateOffset(months=2)
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

            # Initialize ensemble from previous observed state.
            initial = normalize_state(
                prev[STATE].values.astype(np.float32)
            )

            ensemble = np.repeat(
                initial[None,:],
                N_ENSEMBLE,
                axis=0
            )

            ensemble += np.random.normal(
                0,
                PROCESS_NOISE,
                ensemble.shape
            )

            crop_id = crop_to_id[cur["crop"]]

            wh = weather_history(g,i)

            # Forecast current month.
            ensemble = propagate(
                ensemble,
                wh,
                crop_id
            )

            forecast_mean = ensemble.mean(axis=0)

            # Current observation.
            obs_available = []

            for f in obs_features:

                if not pd.isna(cur[f]):
                    obs_available.append(
                        STATE.index(f)
                    )

            obs = normalize_state(
                cur[STATE].fillna(0).values
                .astype(np.float32)
            )

            # Score BEFORE assimilation.
            for j,f in enumerate(STATE):

                if not pd.isna(cur[f]):

                    err = (
                        forecast_mean[j]
                        - obs[j]
                    )

                    all_results.append({
                        "mode": mode,
                        "site_id": site,
                        "crop": cur["crop"],
                        "country": cur["country"],
                        "date": cur["date"],
                        "feature": f,
                        "error": float(err),
                        "squared_error": float(err**2)
                    })

            # Update ensemble.
            if mode != "free_running":

                ensemble = assimilate(
                    ensemble,
                    obs,
                    obs_available
                )


res = pd.DataFrame(all_results)

print("\n==========================================")
print("GLOBAL WEATHER V3 ASSIMILATION")
print("==========================================")

print("Rows:",len(res))
print("Sites:",res.site_id.nunique())

summary = (
    res.groupby(["mode","feature"])
       ["squared_error"]
       .mean()
       .pow(0.5)
       .reset_index(name="RMSE")
)

print("\nFeature RMSE:")
print(
    summary.pivot(
        index="mode",
        columns="feature",
        values="RMSE"
    )
)

overall = (
    res.groupby("mode")
       ["squared_error"]
       .mean()
       .pow(0.5)
       .sort_values()
)

print("\nOverall RMSE:")
print(overall)

free = overall["free_running"]

print("\nGain vs free-running:")

for mode in overall.index:

    if mode == "free_running":
        continue

    gain = (
        (free - overall[mode])
        / free
        * 100
    )

    print(
        f"{mode}: {gain:.2f}%"
    )

crop_summary = (
    res.groupby(
        ["mode","crop"]
    )["squared_error"]
    .mean()
    .pow(0.5)
    .reset_index(name="RMSE")
)

print("\nCrop RMSE:")
print(
    crop_summary.pivot(
        index="crop",
        columns="mode",
        values="RMSE"
    )
)

res.to_csv(
    "GLOBAL_WEATHER_V3_ASSIMILATION_RESULTS.csv",
    index=False
)

summary.to_csv(
    "GLOBAL_WEATHER_V3_ASSIMILATION_SUMMARY.csv",
    index=False
)

print(
    "\nSaved V3 assimilation results."
)
