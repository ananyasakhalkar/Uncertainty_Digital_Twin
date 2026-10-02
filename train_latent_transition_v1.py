import pandas as pd
import numpy as np
import torch
import torch.nn as nn

DEVICE = torch.device(
    "mps" if torch.backends.mps.is_available() else "cpu"
)

DATA = "GLOBAL_ALL25_MODEL_READY.csv"
OBS_MODEL = "models/GLOBAL_LATENT_OBSERVATION_V1.pt"

STATE = ["NDVI","EVI","NDWI","VV","VH","VV_VH"]
WEATHER = ["temperature_C","precipitation_mm","solar_MJ_m2"]

ckpt = torch.load(
    OBS_MODEL,
    map_location="cpu",
    weights_only=False
)

state_mean = pd.Series(ckpt["state_mean"])
state_std = pd.Series(ckpt["state_std"])

df = pd.read_csv(DATA)
df["date"] = pd.to_datetime(df["date"])
df = df.sort_values(["site_id","date"])

crop_to_id = ckpt["crop_to_id"]


class Encoder(nn.Module):

    def __init__(self):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(6, 32),
            nn.Tanh(),
            nn.Linear(32, 4)
        )

    def forward(self, x):
        return self.net(x)


class LatentTransition(nn.Module):

    def __init__(self):
        super().__init__()

        self.crop_embedding = nn.Embedding(
            5, 4
        )

        self.net = nn.Sequential(
            nn.Linear(4 + 9 + 4, 32),
            nn.Tanh(),
            nn.Linear(32, 32),
            nn.Tanh(),
            nn.Linear(32, 4)
        )

    def forward(
        self,
        z,
        weather,
        crop
    ):

        emb = self.crop_embedding(crop)

        x = torch.cat(
            [z, weather, emb],
            dim=1
        )

        return self.net(x)


encoder = Encoder()

encoder.load_state_dict({
    "net." + k.replace("encoder.", ""): v
    for k,v in ckpt["model_state_dict"].items()
    if k.startswith("encoder.")
})

encoder = encoder.to(DEVICE)
encoder.eval()

for p in encoder.parameters():
    p.requires_grad = False


model = LatentTransition().to(DEVICE)

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=1e-3,
    weight_decay=1e-4
)

loss_fn = nn.MSELoss()


def norm_state(x):
    return (
        x - state_mean.values
    ) / state_std.values


def norm_weather(x, wm, ws):
    return (
        x - wm
    ) / ws


# Weather normalization is fitted only on 2019-2022.
train_weather = df[
    (df["date"] >= "2019-01-01") &
    (df["date"] <= "2022-12-01")
]

weather_mean = train_weather[
    WEATHER
].mean().values

weather_std = train_weather[
    WEATHER
].std().values


def make_samples(start, end):

    samples = []

    for site, g in df.groupby("site_id"):

        g = g.sort_values("date").reset_index(drop=True)

        for i in range(2, len(g)):

            cur = g.iloc[i]
            prev = g.iloc[i-1]
            prev2 = g.iloc[i-2]

            if not (
                cur["date"] >= start
                and cur["date"] <= end
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

            s_prev = norm_state(
                prev[STATE].values.astype(np.float32)
            )

            s_cur = norm_state(
                cur[STATE].values.astype(np.float32)
            )

            w0 = norm_weather(
                cur[WEATHER].values.astype(np.float32),
                weather_mean,
                weather_std
            )

            w1 = norm_weather(
                prev[WEATHER].values.astype(np.float32),
                weather_mean,
                weather_std
            )

            w2 = norm_weather(
                prev2[WEATHER].values.astype(np.float32),
                weather_mean,
                weather_std
            )

            samples.append(
                (
                    s_prev,
                    np.concatenate([w0,w1,w2]),
                    crop_to_id[cur["crop"]],
                    s_cur
                )
            )

    return samples


train_samples = make_samples(
    pd.Timestamp("2019-01-01"),
    pd.Timestamp("2022-12-01")
)

valid_samples = make_samples(
    pd.Timestamp("2023-01-01"),
    pd.Timestamp("2023-12-01")
)

print("TRAIN TRANSITIONS:", len(train_samples))
print("VALIDATION TRANSITIONS:", len(valid_samples))


def tensors(samples):

    s = torch.tensor(
        np.array([x[0] for x in samples]),
        dtype=torch.float32,
        device=DEVICE
    )

    w = torch.tensor(
        np.array([x[1] for x in samples]),
        dtype=torch.float32,
        device=DEVICE
    )

    c = torch.tensor(
        [x[2] for x in samples],
        dtype=torch.long,
        device=DEVICE
    )

    target = torch.tensor(
        np.array([x[3] for x in samples]),
        dtype=torch.float32,
        device=DEVICE
    )

    return s,w,c,target


s_train,w_train,c_train,y_train = tensors(
    train_samples
)

s_valid,w_valid,c_valid,y_valid = tensors(
    valid_samples
)


with torch.no_grad():

    z_train = encoder(s_train)
    z_valid = encoder(s_valid)

best = float("inf")
best_state = None
bad = 0
patience = 80

for epoch in range(1,501):

    model.train()

    optimizer.zero_grad()

    delta = model(
        z_train,
        w_train,
        c_train
    )

    pred = z_train + delta

    loss = loss_fn(
        pred,
        encoder(y_train).detach()
    )

    loss.backward()

    torch.nn.utils.clip_grad_norm_(
        model.parameters(),
        1.0
    )

    optimizer.step()

    model.eval()

    with torch.no_grad():

        val_delta = model(
            z_valid,
            w_valid,
            c_valid
        )

        val_pred = z_valid + val_delta

        val_target = encoder(
            y_valid
        )

        val_loss = loss_fn(
            val_pred,
            val_target
        ).item()

    if val_loss < best:

        best = val_loss

        best_state = {
            k:v.detach().cpu().clone()
            for k,v in model.state_dict().items()
        }

        bad = 0

    else:
        bad += 1

    if epoch == 1 or epoch % 50 == 0:

        print(
            f"Epoch {epoch:3d} | "
            f"train {loss.item():.6f} | "
            f"val {val_loss:.6f}"
        )

    if bad >= patience:

        print(
            "Early stopping at epoch",
            epoch
        )
        break


model.load_state_dict(best_state)

torch.save(
    {
        "model_type":
            "latent_transition_weather_v1",

        "model_state_dict":
            model.state_dict(),

        "encoder_state_dict":
            encoder.state_dict(),

        "state_features":
            STATE,

        "weather_features":
            WEATHER,

        "latent_dim":
            4,

        "state_mean":
            state_mean.to_dict(),

        "state_std":
            state_std.to_dict(),

        "weather_mean":
            dict(zip(
                WEATHER,
                weather_mean
            )),

        "weather_std":
            dict(zip(
                WEATHER,
                weather_std
            )),

        "crop_to_id":
            crop_to_id,

        "train_period":
            "2019-2022",

        "validation_period":
            "2023",

        "best_validation_loss":
            best
    },
    "models/GLOBAL_LATENT_TRANSITION_V1.pt"
)

print("\n======================================")
print("LATENT TRANSITION V1")
print("======================================")
print("Best validation loss:", best)
print(
    "Saved: models/GLOBAL_LATENT_TRANSITION_V1.pt"
)
