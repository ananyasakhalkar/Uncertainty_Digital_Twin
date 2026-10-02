import pandas as pd
import numpy as np
import torch
import torch.nn as nn

DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

DATA = "GLOBAL_ALL25_MODEL_READY.csv"
OUT = "models/GLOBAL_5CROP_TEMPORAL_TWIN_V2.pt"

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
    (df["date"] <= "2023-12-01")
].copy()

# Fit normalization on training period only
state_mean = train[STATE].mean()
state_std = train[STATE].std().replace(0, 1)

weather_mean = train[WEATHER].mean()
weather_std = train[WEATHER].std().replace(0, 1)

# Build genuine consecutive monthly transitions.
# Each example contains:
# current state
# previous state change
# current weather
# crop embedding
# seasonal sin/cos
X = []
Y = []

for site, g in train.groupby("site_id"):
    g = g.sort_values("date").reset_index(drop=True)

    for i in range(1, len(g)):
        prev = g.iloc[i-1]
        cur = g.iloc[i]

        # Require true consecutive months
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

        delta = s_cur - s_prev

        w = (
            (cur[WEATHER] - weather_mean) / weather_std
        ).values.astype(np.float32)

        month = cur["date"].month
        sin_m = np.sin(2*np.pi*month/12)
        cos_m = np.cos(2*np.pi*month/12)

        crop_id = CROPS[cur["crop"]]

        X.append((
            s_prev,
            delta,
            w,
            crop_id,
            sin_m,
            cos_m
        ))
        Y.append(delta)

print("DEVICE:", DEVICE)
print("TRAIN ROWS:", len(train))
print("VALID TRANSITIONS:", len(X))

class TwinV2(nn.Module):
    def __init__(self):
        super().__init__()

        self.crop_embedding = nn.Embedding(5, 4)

        # 6 state + 6 delta + 3 weather + 4 crop + 2 seasonal
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
            state,
            delta_prev,
            weather,
            emb,
            sin_m,
            cos_m
        ], dim=1)

        return self.transition(x)

model = TwinV2().to(DEVICE)

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=1e-3,
    weight_decay=1e-5
)

loss_fn = nn.MSELoss()

best_loss = float("inf")
best_state = None

for epoch in range(1, 801):

    optimizer.zero_grad()

    batch_inputs = []
    targets = []

    for (s, d, w, c, sm, cm), y in zip(X, Y):

        batch_inputs.append(
            (
                torch.tensor(s),
                torch.tensor(d),
                torch.tensor(w),
                c,
                sm,
                cm
            )
        )
        targets.append(torch.tensor(y))

    state = torch.stack([x[0] for x in batch_inputs]).to(DEVICE)
    delta_prev = torch.stack([x[1] for x in batch_inputs]).to(DEVICE)
    weather = torch.stack([x[2] for x in batch_inputs]).to(DEVICE)

    crop = torch.tensor(
        [x[3] for x in batch_inputs],
        dtype=torch.long,
        device=DEVICE
    )

    sin_m = torch.tensor(
        [[x[4]] for x in batch_inputs],
        dtype=torch.float32,
        device=DEVICE
    )

    cos_m = torch.tensor(
        [[x[5]] for x in batch_inputs],
        dtype=torch.float32,
        device=DEVICE
    )

    target = torch.stack(targets).to(DEVICE)

    pred = model(
        state,
        delta_prev,
        weather,
        crop,
        sin_m,
        cos_m
    )

    loss = loss_fn(pred, target)

    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()

    value = loss.item()

    if value < best_loss:
        best_loss = value
        best_state = {
            k: v.detach().cpu().clone()
            for k, v in model.state_dict().items()
        }

    if epoch == 1 or epoch % 100 == 0:
        print(f"Epoch {epoch:4d} | loss {value:.6f}")

model.load_state_dict(best_state)

torch.save({
    "model_state_dict": model.state_dict(),
    "state_mean": state_mean.to_dict(),
    "state_std": state_std.to_dict(),
    "weather_mean": weather_mean.to_dict(),
    "weather_std": weather_std.to_dict(),
    "crop_to_id": CROPS,
    "state_features": STATE,
    "weather_features": WEATHER,
    "train_start": "2019-01-01",
    "train_end": "2023-12-01",
    "model_type": "shared_global_five_crop_temporal_twin_v2"
}, OUT)

print("\nBEST TRAINING LOSS:", best_loss)
print("SAVED:", OUT)
