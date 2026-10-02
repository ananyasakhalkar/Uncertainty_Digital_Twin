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
WEATHER = ["temperature_C","precipitation_mm","solar_MJ_m2"]

N_ENSEMBLE = 50
PROCESS_NOISE = 0.05
OBS_NOISE = 0.15


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

weather_mean = np.array(
    [
        trans_ckpt["weather_mean"][f]
        for f in WEATHER
    ],
    dtype=np.float32
)

weather_std = np.array(
    [
        trans_ckpt["weather_std"][f]
        for f in WEATHER
    ],
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

    def forward(
        self,
        z,
        weather,
        crop
    ):

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


def normalize_state(x):

    return (
        x - state_mean.values
    ) / state_std.values


def normalize_weather(x):

    return (
        x - weather_mean
    ) / weather_std


def weather_history(
    cur,
    prev,
    prev2
):

    w0 = normalize_weather(
        cur[WEATHER].values.astype(
            np.float32
        )
    )

    w1 = normalize_weather(
        prev[WEATHER].values.astype(
            np.float32
        )
    )

    w2 = normalize_weather(
        prev2[WEATHER].values.astype(
            np.float32
        )
    )

    return np.concatenate(
        [w0,w1,w2]
    )


def latent_to_observations(
    ensemble
):

    z = torch.tensor(
        ensemble,
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

    n = ensemble.shape[0]

    z = torch.tensor(
        ensemble,
        dtype=torch.float32,
        device=DEVICE
    )

    w = torch.tensor(
        np.repeat(
            weather[None,:],
            n,
            axis=0
        ),
        dtype=torch.float32,
        device=DEVICE
    )

    c = torch.tensor(
        np.repeat(
            crop_id,
            n
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

    Y = latent_to_observations(X)

    Y_obs = Y[:,obs_indices]

    y_mean = Y_obs.mean(axis=0)

    Y_anom = Y_obs - y_mean

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
        +
        np.random.normal(
            0,
            OBS_NOISE,
            size=len(obs_indices)
        )
    )

    innovation = (
        perturbed_obs
        - Y_obs
    )

    return X + innovation @ K.T


modes = {
    "free_running": [],
    "S1_only": S1,
    "S2_only": S2,
    "S1_plus_S2": S1 + S2
}

results = []


for mode, obs_features in modes.items():

    print(
        "\nMODE:",
        mode
    )

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

            prev_state = normalize_state(
                prev[STATE].values.astype(
                    np.float32
                )
            )

            # Convert the previous observed sensor state
            # into the 4-dimensional latent crop state.
            with torch.no_grad():

                prev_tensor = torch.tensor(
                    prev_state,
                    dtype=torch.float32,
                    device=DEVICE
                ).unsqueeze(0)

                prev_latent = (
                    obs_model.encoder(
                        prev_tensor
                    )
                    .cpu()
                    .numpy()[0]
                )

            ensemble = np.repeat(
                prev_latent[None,:],
                N_ENSEMBLE,
                axis=0
            )

            ensemble += np.random.normal(
                0,
                PROCESS_NOISE,
                ensemble.shape
            )

            wh = weather_history(
                cur,
                prev,
                prev2
            )

            crop_id = crop_to_id[
                cur["crop"]
            ]

            ensemble = propagate(
                ensemble,
                wh,
                crop_id
            )

            forecast_latent = (
                ensemble.mean(axis=0)
            )

            # Observation ensemble BEFORE assimilation.
            # This is the actual prior forecast used by the EnKF.
            forecast_obs_ensemble = latent_to_observations(
                ensemble
            )

            forecast_obs = (
                forecast_obs_ensemble.mean(axis=0)
            )

            forecast_obs_var = (
                forecast_obs_ensemble.var(
                    axis=0,
                    ddof=1
                )
            )

            # Observation order:
            # NDVI, EVI, NDWI, VV, VH, VV_VH

            observation = normalize_state(
                cur[STATE].fillna(0).values.astype(
                    np.float32
                )
            )

            for j,f in enumerate(STATE):

                if not pd.isna(cur[f]):

                    error = (
                        forecast_obs[j]
                        - observation[j]
                    )

                    innovation = (
                        observation[j]
                        - forecast_obs[j]
                    )

                    variance = max(
                        float(forecast_obs_var[j]),
                        1e-12
                    )

                    nis = (
                        innovation**2
                        / (
                            variance
                            + OBS_NOISE**2
                        )
                    )

                    results.append({
                        "mode": mode,
                        "site_id": site,
                        "crop": cur["crop"],
                        "country": cur["country"],
                        "date": cur["date"],
                        "feature": f,
                        "forecast": float(
                            forecast_obs[j]
                        ),
                        "observation": float(
                            observation[j]
                        ),
                        "innovation": float(
                            innovation
                        ),
                        "ensemble_variance": float(
                            forecast_obs_var[j]
                        ),
                        "NIS": float(nis),
                        "error": float(error),
                        "squared_error":
                            float(error**2)
                    })

            if mode != "free_running":

                obs_indices = [
                    STATE.index(f)
                    for f in obs_features
                    if not pd.isna(cur[f])
                ]

                ensemble = assimilate(
                    ensemble,
                    observation,
                    obs_indices
                )


res = pd.DataFrame(results)

print(
    "\n=========================================="
)

print(
    "LATENT V1 ASSIMILATION 2024"
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


feature_rmse = (
    res.groupby(
        ["mode","feature"]
    )["squared_error"]
    .mean()
    .pow(0.5)
    .reset_index(name="RMSE")
)

print("\nFeature RMSE:")

print(
    feature_rmse.pivot(
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
)

print(
    "\nOverall RMSE:"
)

print(overall)


free = overall["free_running"]

print(
    "\nGain vs free-running:"
)

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


crop_rmse = (
    res.groupby(
        ["crop","mode"]
    )["squared_error"]
    .mean()
    .pow(0.5)
    .reset_index(name="RMSE")
)

print(
    "\nCrop RMSE:"
)

print(
    crop_rmse.pivot(
        index="crop",
        columns="mode",
        values="RMSE"
    )
)


res.to_csv(
    "GLOBAL_LATENT_V1_ASSIMILATION_2024_RESULTS.csv",
    index=False
)

feature_rmse.to_csv(
    "GLOBAL_LATENT_V1_ASSIMILATION_2024_SUMMARY.csv",
    index=False
)

print(
    "\nSaved latent assimilation results."
)
