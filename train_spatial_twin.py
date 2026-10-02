import numpy as np
import pandas as pd
import torch
import torch.nn as nn

TRAIN_FILE = "GLOBAL_SPATIAL_TRAIN.csv"
TEST_FILE = "GLOBAL_SPATIAL_TEST.csv"

MODEL_OUT = "models/GLOBAL_SPATIAL_TWIN.pt"

STATE_FEATURES = [
    "NDVI","EVI","NDWI",
    "VV","VH","VV_VH"
]

WEATHER_FEATURES = [
    "temperature_C",
    "precipitation_mm",
    "solar_MJ_m2"
]

CROP_TO_ID = {
    "Cotton": 0,
    "Maize": 1,
    "Rice": 2,
    "Soybean": 3,
    "Wheat": 4
}

DEVICE = (
    torch.device("mps")
    if torch.backends.mps.is_available()
    else torch.device("cpu")
)

print("DEVICE:", DEVICE)

train = pd.read_csv(
    TRAIN_FILE,
    parse_dates=["date"]
)

test = pd.read_csv(
    TEST_FILE,
    parse_dates=["date"]
)

train = train.sort_values(
    ["site_id","date"]
)

test = test.sort_values(
    ["site_id","date"]
)


class TransitionModel(nn.Module):

    def __init__(
        self,
        state_dim=6,
        weather_dim=3,
        n_crops=5,
        crop_embedding_dim=4,
        hidden_dim=32
    ):
        super().__init__()

        self.crop_embedding = nn.Embedding(
            n_crops,
            crop_embedding_dim
        )

        self.transition = nn.Sequential(
            nn.Linear(
                state_dim +
                weather_dim +
                crop_embedding_dim,
                hidden_dim
            ),
            nn.ReLU(),
            nn.Linear(
                hidden_dim,
                hidden_dim
            ),
            nn.ReLU(),
            nn.Linear(
                hidden_dim,
                state_dim
            )
        )

    def forward(
        self,
        state,
        weather,
        crop_id
    ):

        crop_emb = self.crop_embedding(
            crop_id
        )

        x = torch.cat(
            [
                state,
                weather,
                crop_emb
            ],
            dim=1
        )

        delta = self.transition(x)

        return (
            state +
            0.10 *
            torch.tanh(delta)
        )


# =========================================================
# STANDARDIZATION FROM TRAINING SITES ONLY
# =========================================================

state_mean_dict = (
    train[STATE_FEATURES]
    .mean()
    .to_dict()
)

state_std_dict = (
    train[STATE_FEATURES]
    .std()
    .replace(0, 1)
    .to_dict()
)

weather_mean_dict = (
    train[WEATHER_FEATURES]
    .mean()
    .to_dict()
)

weather_std_dict = (
    train[WEATHER_FEATURES]
    .std()
    .replace(0, 1)
    .to_dict()
)


# =========================================================
# BUILD CONSECUTIVE MONTHLY TRANSITIONS
# =========================================================

transitions = []

for site_id, group in train.groupby("site_id"):

    group = group.sort_values("date").reset_index(drop=True)

    crop = group["crop"].iloc[0]
    crop_id = CROP_TO_ID[crop]

    for i in range(len(group) - 1):

        a = group.iloc[i]
        b = group.iloc[i + 1]

        expected = (
            a["date"] +
            pd.DateOffset(months=1)
        )

        if b["date"] != expected:
            continue

        if a[STATE_FEATURES].isna().any():
            continue

        if b[STATE_FEATURES].isna().any():
            continue

        if a[WEATHER_FEATURES].isna().any():
            continue

        transitions.append(
            (
                a[STATE_FEATURES].to_numpy(
                    dtype=np.float32
                ),
                a[WEATHER_FEATURES].to_numpy(
                    dtype=np.float32
                ),
                b[STATE_FEATURES].to_numpy(
                    dtype=np.float32
                ),
                crop_id
            )
        )


print(
    "VALID SPATIAL-TRAIN TRANSITIONS:",
    len(transitions)
)


# =========================================================
# ARRAYS
# =========================================================

X_state = np.array(
    [x[0] for x in transitions],
    dtype=np.float32
)

X_weather = np.array(
    [x[1] for x in transitions],
    dtype=np.float32
)

Y_state = np.array(
    [x[2] for x in transitions],
    dtype=np.float32
)

X_crop = np.array(
    [x[3] for x in transitions],
    dtype=np.int64
)


# =========================================================
# NORMALIZATION
# =========================================================

state_mean = np.array(
    [
        state_mean_dict[x]
        for x in STATE_FEATURES
    ],
    dtype=np.float32
)

state_std = np.array(
    [
        state_std_dict[x]
        for x in STATE_FEATURES
    ],
    dtype=np.float32
)

weather_mean = np.array(
    [
        weather_mean_dict[x]
        for x in WEATHER_FEATURES
    ],
    dtype=np.float32
)

weather_std = np.array(
    [
        weather_std_dict[x]
        for x in WEATHER_FEATURES
    ],
    dtype=np.float32
)


X_state_n = (
    X_state -
    state_mean
) / state_std

X_weather_n = (
    X_weather -
    weather_mean
) / weather_std

Y_state_n = (
    Y_state -
    state_mean
) / state_std


# =========================================================
# TORCH
# =========================================================

X_state_t = torch.tensor(
    X_state_n,
    dtype=torch.float32,
    device=DEVICE
)

X_weather_t = torch.tensor(
    X_weather_n,
    dtype=torch.float32,
    device=DEVICE
)

Y_state_t = torch.tensor(
    Y_state_n,
    dtype=torch.float32,
    device=DEVICE
)

X_crop_t = torch.tensor(
    X_crop,
    dtype=torch.long,
    device=DEVICE
)


# =========================================================
# MODEL
# =========================================================

model = TransitionModel().to(DEVICE)

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=1e-3
)

loss_fn = nn.MSELoss()


# =========================================================
# TRAIN
# =========================================================

EPOCHS = 500

for epoch in range(EPOCHS):

    model.train()

    optimizer.zero_grad()

    pred = model(
        X_state_t,
        X_weather_t,
        X_crop_t
    )

    loss = loss_fn(
        pred,
        Y_state_t
    )

    loss.backward()

    optimizer.step()

    if epoch == 0 or (epoch + 1) % 100 == 0:

        print(
            f"Epoch {epoch+1}: "
            f"loss={loss.item():.6f}"
        )


# =========================================================
# SAVE
# =========================================================

checkpoint = {
    "model_state_dict":
        model.state_dict(),

    "state_mean":
        state_mean_dict,

    "state_std":
        state_std_dict,

    "weather_mean":
        weather_mean_dict,

    "weather_std":
        weather_std_dict,

    "crop_to_id":
        CROP_TO_ID,

    "state_features":
        STATE_FEATURES,

    "weather_features":
        WEATHER_FEATURES
}

torch.save(
    checkpoint,
    MODEL_OUT
)

print(
    "\nSaved:",
    MODEL_OUT
)


# =========================================================
# TEMPORAL TEST SET INFORMATION
# =========================================================

print(
    "\nSPATIAL TRAIN SITES:",
    train.site_id.nunique()
)

print(
    "SPATIAL TEST SITES:",
    test.site_id.nunique()
)

print(
    "SPATIAL TEST ROWS:",
    len(test)
)
