import numpy as np
import pandas as pd
import torch
import torch.nn as nn

DATA_FILE = "GLOBAL_ALL25_MODEL_READY.csv"
MODEL_FILE = "models/GLOBAL_5CROP_TEMPORAL_TWIN.pt"

N_ENSEMBLE = 50
PROCESS_NOISE = np.array([
    1.0058,
    0.9946,
    1.0892,
    1.2049,
    1.1302,
    1.1972
], dtype=np.float32)

STATE_FEATURES = [
    "NDVI","EVI","NDWI","VV","VH","VV_VH"
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

data = pd.read_csv(
    DATA_FILE,
    parse_dates=["date"]
)

checkpoint = torch.load(
    MODEL_FILE,
    map_location=DEVICE,
    weights_only=False
)


class TransitionModel(nn.Module):

    def __init__(self):
        super().__init__()

        self.crop_embedding = nn.Embedding(
            5, 4
        )

        self.transition = nn.Sequential(
            nn.Linear(13, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Linear(32, 6)
        )

    def forward(
        self,
        state,
        weather,
        crop_id
    ):

        emb = self.crop_embedding(
            crop_id
        )

        x = torch.cat(
            [state, weather, emb],
            dim=1
        )

        return (
            state +
            0.10 *
            torch.tanh(
                self.transition(x)
            )
        )


model = TransitionModel().to(DEVICE)

model.load_state_dict(
    checkpoint["model_state_dict"]
)

model.eval()


state_mean = np.array(
    [
        checkpoint["state_mean"][x]
        for x in STATE_FEATURES
    ],
    dtype=np.float32
)

state_std = np.array(
    [
        checkpoint["state_std"][x]
        for x in STATE_FEATURES
    ],
    dtype=np.float32
)

weather_mean = np.array(
    [
        checkpoint["weather_mean"][x]
        for x in WEATHER_FEATURES
    ],
    dtype=np.float32
)

weather_std = np.array(
    [
        checkpoint["weather_std"][x]
        for x in WEATHER_FEATURES
    ],
    dtype=np.float32
)


def transition_ensemble(
    ensemble,
    weather,
    crop_id
):

    n = len(ensemble)

    state = torch.tensor(
        ensemble,
        dtype=torch.float32,
        device=DEVICE
    )

    weather_array = np.repeat(
        weather[None, :],
        n,
        axis=0
    )

    weather_tensor = torch.tensor(
        weather_array,
        dtype=torch.float32,
        device=DEVICE
    )

    crop_tensor = torch.full(
        (n,),
        crop_id,
        dtype=torch.long,
        device=DEVICE
    )

    state_n = (
        state -
        torch.tensor(
            state_mean,
            dtype=torch.float32,
            device=DEVICE
        )
    ) / torch.tensor(
        state_std,
        dtype=torch.float32,
        device=DEVICE
    )

    weather_n = (
        weather_tensor -
        torch.tensor(
            weather_mean,
            dtype=torch.float32,
            device=DEVICE
        )
    ) / torch.tensor(
        weather_std,
        dtype=torch.float32,
        device=DEVICE
    )

    with torch.no_grad():

        next_n = model(
            state_n,
            weather_n,
            crop_tensor
        )

    next_state = (
        next_n *
        torch.tensor(
            state_std,
            dtype=torch.float32,
            device=DEVICE
        )
        +
        torch.tensor(
            state_mean,
            dtype=torch.float32,
            device=DEVICE
        )
    )

    return (
        next_state
        .cpu()
        .numpy()
    )


results = []


for site_id, group in data.groupby(
    "site_id"
):

    group = group.sort_values(
        "date"
    ).reset_index(drop=True)

    crop = group["crop"].iloc[0]
    crop_id = CROP_TO_ID[crop]

    # -----------------------------------------------------
    # Find latest complete state before 2024
    # -----------------------------------------------------

    pre = group[
        group["date"] <
        pd.Timestamp("2024-01-01")
    ]

    complete = pre[
        pre[STATE_FEATURES]
        .notna()
        .all(axis=1)
    ]

    if len(complete) == 0:
        continue

    start = complete.iloc[-1]

    ensemble = np.repeat(
        start[STATE_FEATURES]
        .to_numpy(
            dtype=np.float32
        )[None, :],
        N_ENSEMBLE,
        axis=0
    )

    ensemble += np.random.normal(
        0,
        PROCESS_NOISE,
        ensemble.shape
    ).astype(np.float32)

    current_date = start["date"]

    # -----------------------------------------------------
    # Propagate to Dec 2023
    # -----------------------------------------------------

    while current_date < pd.Timestamp(
        "2023-12-01"
    ):

        next_date = (
            current_date +
            pd.DateOffset(months=1)
        )

        row = group[
            group["date"] == next_date
        ]

        if len(row) == 0:
            ensemble = None
            break

        row = row.iloc[0]

        if row[
            WEATHER_FEATURES
        ].isna().any():

            ensemble = None
            break

        weather = row[
            WEATHER_FEATURES
        ].to_numpy(
            dtype=np.float32
        )

        ensemble = transition_ensemble(
            ensemble,
            weather,
            crop_id
        )

        ensemble += np.random.normal(
            0,
            PROCESS_NOISE,
            ensemble.shape
        ).astype(np.float32)

        current_date = next_date

    if ensemble is None:
        continue

    # -----------------------------------------------------
    # Hybrid Dec 2023 initialization
    # -----------------------------------------------------

    dec = group[
        group["date"] ==
        pd.Timestamp("2023-12-01")
    ]

    if len(dec) == 0:
        continue

    dec = dec.iloc[0]

    for j, feature in enumerate(
        STATE_FEATURES
    ):

        if pd.notna(dec[feature]):

            ensemble[:, j] = float(
                dec[feature]
            )

    # -----------------------------------------------------
    # 2024 evaluation
    # -----------------------------------------------------

    test = group[
        (
            group["date"] >=
            pd.Timestamp("2024-01-01")
        )
        &
        (
            group["date"] <=
            pd.Timestamp("2024-10-01")
        )
    ]

    previous_date = pd.Timestamp(
        "2023-12-01"
    )

    for _, row in test.iterrows():

        current_date = row["date"]

        expected = (
            previous_date +
            pd.DateOffset(months=1)
        )

        if current_date != expected:
            previous_date = current_date
            continue

        if row[
            WEATHER_FEATURES
        ].isna().any():

            previous_date = current_date
            continue

        # ---------------------------------------------
        # Forecast ensemble
        # ---------------------------------------------

        weather = row[
            WEATHER_FEATURES
        ].to_numpy(
            dtype=np.float32
        )

        forecast = transition_ensemble(
            ensemble,
            weather,
            crop_id
        )

        forecast += np.random.normal(
            0,
            PROCESS_NOISE,
            forecast.shape
        ).astype(np.float32)

        # ---------------------------------------------
        # Compare spread to observed error
        # ---------------------------------------------

        for j, feature in enumerate(
            STATE_FEATURES
        ):

            if pd.isna(row[feature]):
                continue

            observed = float(
                row[feature]
            )

            mean_prediction = (
                forecast[:, j].mean()
            )

            spread = (
                forecast[:, j].std(
                    ddof=1
                )
            )

            absolute_error = abs(
                mean_prediction -
                observed
            )

            results.append({
                "site_id": site_id,
                "crop": crop,
                "date": current_date,
                "feature": feature,
                "spread": spread,
                "absolute_error":
                    absolute_error
            })

        ensemble = forecast
        previous_date = current_date


df = pd.DataFrame(results)

print(
    "\n========================================"
)
print("UNCERTAINTY VALIDATION")
print("========================================"
)

print(
    "\nRows:",
    len(df)
)

print(
    "Sites:",
    df.site_id.nunique()
)


# =========================================================
# CORRELATION
# =========================================================

overall_corr = df[
    ["spread","absolute_error"]
].corr().iloc[0,1]

print(
    "\nOverall spread-error correlation:",
    round(overall_corr, 4)
)


# =========================================================
# FEATURE CORRELATION
# =========================================================

feature_corr = (
    df.groupby("feature")
    .apply(
        lambda x:
        x["spread"].corr(
            x["absolute_error"]
        ),
        include_groups=False
    )
)

print(
    "\nCorrelation by feature:"
)

print(
    feature_corr.round(4)
)


# =========================================================
# QUARTILE TEST
# =========================================================

df["spread_bin"] = pd.qcut(
    df["spread"],
    q=4,
    labels=[
        "Q1_low",
        "Q2",
        "Q3",
        "Q4_high"
    ],
    duplicates="drop"
)

quartiles = (
    df.groupby(
        "spread_bin",
        observed=False
    )
    .agg(
        mean_spread=(
            "spread",
            "mean"
        ),
        mean_absolute_error=(
            "absolute_error",
            "mean"
        ),
        N=(
            "absolute_error",
            "size"
        )
    )
)

print(
    "\nSpread quartiles:"
)

print(
    quartiles.round(4)
)


# =========================================================
# CROP CORRELATION
# =========================================================

crop_corr = (
    df.groupby("crop")
    .apply(
        lambda x:
        x["spread"].corr(
            x["absolute_error"]
        ),
        include_groups=False
    )
)

print(
    "\nCorrelation by crop:"
)

print(
    crop_corr.round(4)
)


# =========================================================
# SAVE
# =========================================================

df.to_csv(
    "GLOBAL_2024_UNCERTAINTY_RESULTS.csv",
    index=False
)

quartiles.to_csv(
    "GLOBAL_2024_UNCERTAINTY_QUARTILES.csv"
)

print(
    "\nSaved:"
)

print(
    "GLOBAL_2024_UNCERTAINTY_RESULTS.csv"
)

print(
    "GLOBAL_2024_UNCERTAINTY_QUARTILES.csv"
)
