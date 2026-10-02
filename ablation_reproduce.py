import numpy as np
import pandas as pd
import torch
import torch.nn as nn

MODEL = "models/GLOBAL_LATENT_TRANSITION_V1.pt"
OBS_MODEL = "models/GLOBAL_LATENT_OBSERVATION_V1.pt"
DATA = "GLOBAL_ALL25_MODEL_READY.csv"

DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

STATE = ["NDVI","EVI","NDWI","VV","VH","VV_VH"]
S1 = ["VV","VH","VV_VH"]
S2 = ["NDVI","EVI","NDWI"]
WEATHER = ["temperature_C","precipitation_mm","solar_MJ_m2"]

# ------------------------------------------------------------
# Load
# ------------------------------------------------------------

df = pd.read_csv(DATA)
df["date"] = pd.to_datetime(df["date"])
df = df.sort_values(["site_id","date"]).reset_index(drop=True)

obs_ckpt = torch.load(OBS_MODEL, map_location="cpu", weights_only=False)
trans_ckpt = torch.load(MODEL, map_location="cpu", weights_only=False)

state_mean = np.array(
    [trans_ckpt["state_mean"][f] for f in STATE],
    dtype=np.float32
)
state_std = np.array(
    [trans_ckpt["state_std"][f] for f in STATE],
    dtype=np.float32
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

# ------------------------------------------------------------
# Exact model definitions from evaluator
# ------------------------------------------------------------

class ObservationModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(6,32),
            nn.Tanh(),
            nn.Linear(32,4)
        )
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
        return z, self.s1_head(z), self.s2_head(z)


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
        x = torch.cat([z,weather,emb],dim=1)
        return self.net(x)


obs_model = ObservationModel().to(DEVICE)
obs_model.load_state_dict(obs_ckpt["model_state_dict"])
obs_model.eval()

transition = LatentTransition().to(DEVICE)
transition.load_state_dict(trans_ckpt["model_state_dict"])
transition.eval()

# ------------------------------------------------------------
# Build EXACT valid evaluation rows
# ------------------------------------------------------------

examples = []

with torch.no_grad():

    for site, g in df.groupby("site_id"):

        g = g.sort_values("date").reset_index(drop=True)

        for i in range(2,len(g)):

            cur = g.iloc[i]
            prev = g.iloc[i-1]
            prev2 = g.iloc[i-2]

            if cur["date"] < pd.Timestamp("2024-01-01"):
                continue

            if cur["date"] > pd.Timestamp("2024-10-01"):
                continue

            if (
                prev["date"] != cur["date"] - pd.DateOffset(months=1)
                or
                prev2["date"] != cur["date"] - pd.DateOffset(months=2)
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
                    prev[STATE].values.astype(np.float32)
                    - state_mean
                )
                / state_std
            )

            target_state = (
                (
                    cur[STATE].values.astype(np.float32)
                    - state_mean
                )
                / state_std
            )

            w0 = (
                (
                    cur[WEATHER].values.astype(np.float32)
                    - weather_mean
                )
                / weather_std
            )

            w1 = (
                (
                    prev[WEATHER].values.astype(np.float32)
                    - weather_mean
                )
                / weather_std
            )

            w2 = (
                (
                    prev2[WEATHER].values.astype(np.float32)
                    - weather_mean
                )
                / weather_std
            )

            weather_history = np.concatenate([w0,w1,w2])

            examples.append({
                "site_id": site,
                "date": cur["date"],
                "crop": cur["crop"],
                "previous_state": previous_state,
                "target_state": target_state,
                "weather": weather_history
            })

print("VALID SITE-MONTHS:", len(examples))
print("SITES:", len(set(x["site_id"] for x in examples)))

# ------------------------------------------------------------
# Make predictions
# ------------------------------------------------------------

def predict(x, crop_name, weather_history):

    state = torch.tensor(
        x["previous_state"],
        dtype=torch.float32,
        device=DEVICE
    ).unsqueeze(0)

    weather = torch.tensor(
        weather_history,
        dtype=torch.float32,
        device=DEVICE
    ).unsqueeze(0)

    crop = torch.tensor(
        [crop_to_id[crop_name]],
        dtype=torch.long,
        device=DEVICE
    )

    z = obs_model.encoder(state)

    delta = transition(z, weather, crop)

    z_next = z + delta

    pred_s1 = obs_model.s1_head(z_next).detach().cpu().numpy()[0]
    pred_s2 = obs_model.s2_head(z_next).detach().cpu().numpy()[0]

    pred = np.zeros(6, dtype=np.float32)

    pred[[STATE.index(v) for v in S1]] = pred_s1
    pred[[STATE.index(v) for v in S2]] = pred_s2

    return pred


# ------------------------------------------------------------
# Deterministic weather shuffle
#
# Shuffle complete 3-month weather histories across examples.
# Crop stays unchanged unless crop_swap is requested.
# ------------------------------------------------------------

rng = np.random.default_rng(2026)

weather_pool = [
    x["weather"].copy()
    for x in examples
]

perm = rng.permutation(len(weather_pool))

# Crop swap: use the next crop in a deterministic cyclic order.
crops = [x["crop"] for x in examples]
crop_swapped = crops[1:] + crops[:1]


conditions = {
    "latent": [],
    "weather_shuffle": [],
    "crop_swap": [],
    "weather_shuffle_crop_swap": []
}

for i,x in enumerate(examples):

    conditions["latent"].append(
        (x["crop"], x["weather"])
    )

    conditions["weather_shuffle"].append(
        (x["crop"], weather_pool[perm[i]])
    )

    conditions["crop_swap"].append(
        (crop_swapped[i], x["weather"])
    )

    conditions["weather_shuffle_crop_swap"].append(
        (crop_swapped[i], weather_pool[perm[i]])
    )


# ------------------------------------------------------------
# Score
# ------------------------------------------------------------

rows = []

for condition, pairs in conditions.items():

    for x, (crop_name, weather_history) in zip(examples, pairs):

        pred = predict(
            x,
            crop_name,
            weather_history
        )

        error = pred - x["target_state"]

        for j, feature in enumerate(STATE):

            rows.append({
                "condition": condition,
                "site_id": x["site_id"],
                "date": x["date"],
                "feature": feature,
                "squared_error": float(error[j] ** 2)
            })


res = pd.DataFrame(rows)

summary = (
    res.groupby("condition")
       .agg(
           RMSE=("squared_error", lambda x: np.sqrt(x.mean())),
           N=("squared_error","size")
       )
       .reset_index()
)

baseline = float(
    summary.loc[
        summary.condition == "latent",
        "RMSE"
    ].iloc[0]
)

summary["RMSE_change_vs_latent"] = (
    summary["RMSE"] - baseline
)

summary["percent_change_vs_latent"] = (
    100 * summary["RMSE_change_vs_latent"] / baseline
)

print("\nEXACT-PROTOCOL ABLATION")
print(summary.to_string(index=False))

summary.to_csv(
    "FINAL_TEST_TIME_ABLATIONS_REPRODUCED.csv",
    index=False
)

print("\nExpected baseline from original evaluator: ~0.584825")
print("Reproduced baseline:", baseline)
