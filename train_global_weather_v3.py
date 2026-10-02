import pandas as pd
import numpy as np
import torch
import torch.nn as nn

DEVICE = torch.device(
    "mps" if torch.backends.mps.is_available() else "cpu"
)

DATA = "GLOBAL_ALL25_MODEL_READY.csv"
OUT = "models/GLOBAL_5CROP_WEATHER_V3.pt"

STATE = ["NDVI","EVI","NDWI","VV","VH","VV_VH"]
WEATHER = ["temperature_C","precipitation_mm","solar_MJ_m2"]

CROPS = {
    "Cotton": 0,
    "Maize": 1,
    "Rice": 2,
    "Soybean": 3,
    "Wheat": 4
}

df = pd.read_csv(DATA)
df["date"] = pd.to_datetime(df["date"])
df = df.sort_values(["site_id","date"])

train = df[
    (df["date"] >= "2019-01-01") &
    (df["date"] <= "2022-12-01")
].copy()

val = df[
    (df["date"] >= "2023-01-01") &
    (df["date"] <= "2023-12-01")
].copy()

# Normalization from training only
state_mean = train[STATE].mean()
state_std = train[STATE].std().replace(0, 1)

weather_mean = train[WEATHER].mean()
weather_std = train[WEATHER].std().replace(0, 1)


def build_examples(data):
    X = []
    Y = []

    for site, g in data.groupby("site_id"):

        g = g.sort_values("date").reset_index(drop=True)

        for i in range(2, len(g)):

            prev = g.iloc[i-1]
            cur = g.iloc[i]
            prev2 = g.iloc[i-2]

            # Need true monthly sequence
            if (
                prev["date"] != cur["date"] - pd.DateOffset(months=1)
                or
                prev2["date"] != cur["date"] - pd.DateOffset(months=2)
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
                (prev[STATE] - state_mean) / state_std
            ).values.astype(np.float32)

            s_cur = (
                (cur[STATE] - state_mean) / state_std
            ).values.astype(np.float32)

            # 3 months of weather
            w0 = (
                (cur[WEATHER] - weather_mean) / weather_std
            ).values.astype(np.float32)

            w1 = (
                (prev[WEATHER] - weather_mean) / weather_std
            ).values.astype(np.float32)

            w2 = (
                (prev2[WEATHER] - weather_mean) / weather_std
            ).values.astype(np.float32)

            weather_history = np.concatenate([
                w0, w1, w2
            ])

            X.append((
                s_prev,
                weather_history,
                CROPS[cur["crop"]]
            ))

            Y.append(s_cur - s_prev)

    return X, Y


X_train, Y_train = build_examples(train)
X_val, Y_val = build_examples(val)

print("DEVICE:", DEVICE)
print("TRAIN ROWS:", len(train))
print("VALIDATION ROWS:", len(val))
print("TRAIN TRANSITIONS:", len(X_train))
print("VALIDATION TRANSITIONS:", len(X_val))


class WeatherTwinV3(nn.Module):

    def __init__(self):
        super().__init__()

        self.crop_embedding = nn.Embedding(
            5, 4
        )

        self.transition = nn.Sequential(
            nn.Linear(19, 32),
            nn.Tanh(),
            nn.Linear(32, 32),
            nn.Tanh(),
            nn.Linear(32, 6)
        )

    def forward(
        self,
        state,
        weather,
        crop
    ):

        emb = self.crop_embedding(crop)

        x = torch.cat([
            state,
            weather,
            emb
        ], dim=1)

        return self.transition(x)


model = WeatherTwinV3().to(DEVICE)

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=1e-3,
    weight_decay=1e-4
)

loss_fn = nn.MSELoss()

best_val = float("inf")
best_state = None
patience = 80
bad_epochs = 0

for epoch in range(1, 801):

    model.train()
    optimizer.zero_grad()

    state = torch.tensor(
        np.stack([x[0] for x in X_train]),
        dtype=torch.float32,
        device=DEVICE
    )

    weather = torch.tensor(
        np.stack([x[1] for x in X_train]),
        dtype=torch.float32,
        device=DEVICE
    )

    crop = torch.tensor(
        [x[2] for x in X_train],
        dtype=torch.long,
        device=DEVICE
    )

    target = torch.tensor(
        np.stack(Y_train),
        dtype=torch.float32,
        device=DEVICE
    )

    pred = model(
        state,
        weather,
        crop
    )

    train_loss = loss_fn(
        pred,
        target
    )

    train_loss.backward()

    torch.nn.utils.clip_grad_norm_(
        model.parameters(),
        1.0
    )

    optimizer.step()

    # Validation
    model.eval()

    with torch.no_grad():

        vs = torch.tensor(
            np.stack([x[0] for x in X_val]),
            dtype=torch.float32,
            device=DEVICE
        )

        vw = torch.tensor(
            np.stack([x[1] for x in X_val]),
            dtype=torch.float32,
            device=DEVICE
        )

        vc = torch.tensor(
            [x[2] for x in X_val],
            dtype=torch.long,
            device=DEVICE
        )

        vy = torch.tensor(
            np.stack(Y_val),
            dtype=torch.float32,
            device=DEVICE
        )

        vp = model(
            vs,
            vw,
            vc
        )

        val_loss = loss_fn(
            vp,
            vy
        ).item()

    if val_loss < best_val:

        best_val = val_loss
        bad_epochs = 0

        best_state = {
            k: v.detach().cpu().clone()
            for k, v in model.state_dict().items()
        }

    else:
        bad_epochs += 1

    if (
        epoch == 1
        or epoch % 50 == 0
    ):
        print(
            f"Epoch {epoch:4d} | "
            f"train {train_loss.item():.6f} | "
            f"val {val_loss:.6f}"
        )

    if bad_epochs >= patience:
        print(
            f"Early stopping at epoch {epoch}"
        )
        break


model.load_state_dict(
    best_state
)

torch.save({

    "model_state_dict":
        model.state_dict(),

    "state_mean":
        state_mean.to_dict(),

    "state_std":
        state_std.to_dict(),

    "weather_mean":
        weather_mean.to_dict(),

    "weather_std":
        weather_std.to_dict(),

    "crop_to_id":
        CROPS,

    "state_features":
        STATE,

    "weather_features":
        WEATHER,

    "train_start":
        "2019-01-01",

    "train_end":
        "2022-12-01",

    "validation_period":
        "2023",

    "model_type":
        "shared_global_five_crop_weather_history_v3"

}, OUT)

print("\nBEST VALIDATION LOSS:", best_val)
print("SAVED:", OUT)
