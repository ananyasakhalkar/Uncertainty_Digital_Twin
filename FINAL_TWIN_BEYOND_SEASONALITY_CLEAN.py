import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path

# ============================================================
# FINAL FROZEN TEST
# DOES THE TWIN ADD ANYTHING BEYOND SEASONALITY?
#
# NO TRAINING
# NO IMPORTS FROM TRAINING SCRIPTS
# ============================================================

DATA = "GLOBAL_ALL25_MODEL_READY.csv"

TRANSITION_MODEL = "models/GLOBAL_LATENT_TRANSITION_V1.pt"
OBS_MODEL = "models/GLOBAL_LATENT_OBSERVATION_V1.pt"

SEED = 42
N_BOOT = 10000

STATE = [
    "NDVI",
    "EVI",
    "NDWI",
    "VV",
    "VH",
    "VV_VH",
]

WEATHER = [
    "temperature_C",
    "precipitation_mm",
    "solar_MJ_m2",
]

rng = np.random.default_rng(SEED)

print("=" * 60)
print("FINAL TEST: DOES THE TWIN ADD ANYTHING BEYOND SEASONALITY?")
print("FROZEN MODELS - NO TRAINING")
print("=" * 60)

# ------------------------------------------------------------
# Data
# ------------------------------------------------------------

data = pd.read_csv(DATA)
data["date"] = pd.to_datetime(data["date"])

# Fixed 2024 evaluation months:
# complete previous state + complete current state +
# complete 3-month weather history.
# ------------------------------------------------------------

def get_weather_history(site_df, current_date):
    vals = []

    for offset in [2, 1, 0]:

        d = current_date - pd.DateOffset(
            months=offset
        )

        row = site_df[
            site_df["date"].eq(d)
        ]

        if len(row) != 1:
            return None

        row = row.iloc[0]

        if row[WEATHER].isna().any():
            return None

        vals.extend(
            row[WEATHER]
            .values
            .astype(float)
            .tolist()
        )

    return np.asarray(
        vals,
        dtype=np.float32
    )


# ------------------------------------------------------------
# Load frozen transition checkpoint
# ------------------------------------------------------------

transition_ckpt = torch.load(
    TRANSITION_MODEL,
    map_location="cpu",
    weights_only=False
)

print(
    "Transition checkpoint keys:",
    list(transition_ckpt.keys())
)


class LatentTransition(nn.Module):

    def __init__(self):
        super().__init__()

        self.crop_embedding = nn.Embedding(
            5,
            4
        )

        self.net = nn.Sequential(
            nn.Linear(
                4 + 9 + 4,
                32
            ),
            nn.Tanh(),
            nn.Linear(
                32,
                32
            ),
            nn.Tanh(),
            nn.Linear(
                32,
                4
            )
        )

    def forward(
        self,
        z,
        weather,
        crop
    ):

        emb = self.crop_embedding(
            crop
        )

        x = torch.cat(
            [
                z,
                weather,
                emb
            ],
            dim=1
        )

        return self.net(x)


transition = LatentTransition()

transition.load_state_dict(
    transition_ckpt[
        "model_state_dict"
    ],
    strict=True
)

transition.eval()

# ------------------------------------------------------------
# Transition normalization
# ------------------------------------------------------------

state_mean = np.asarray(
    [
        transition_ckpt["state_mean"][f]
        for f in STATE
    ],
    dtype=np.float32
)

state_std = np.asarray(
    [
        transition_ckpt["state_std"][f]
        for f in STATE
    ],
    dtype=np.float32
)

weather_mean = np.asarray(
    [
        transition_ckpt["weather_mean"][f]
        for f in WEATHER
    ],
    dtype=np.float32
)

weather_std = np.asarray(
    [
        transition_ckpt["weather_std"][f]
        for f in WEATHER
    ],
    dtype=np.float32
)

state_std = np.where(
    state_std == 0,
    1,
    state_std
)

weather_std = np.where(
    weather_std == 0,
    1,
    weather_std
)

crop_to_id = transition_ckpt[
    "crop_to_id"
]

# ------------------------------------------------------------
# Load frozen observation checkpoint
# ------------------------------------------------------------

obs_ckpt = torch.load(
    OBS_MODEL,
    map_location="cpu",
    weights_only=False
)

print(
    "Observation checkpoint keys:",
    list(obs_ckpt.keys())
)


class LatentObservationModel(nn.Module):

    def __init__(self):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Linear(
                6,
                32
            ),
            nn.Tanh(),
            nn.Linear(
                32,
                4
            )
        )

        self.s1_head = nn.Sequential(
            nn.Linear(
                4,
                16
            ),
            nn.Tanh(),
            nn.Linear(
                16,
                3
            )
        )

        self.s2_head = nn.Sequential(
            nn.Linear(
                4,
                16
            ),
            nn.Tanh(),
            nn.Linear(
                16,
                3
            )
        )

    def forward(
        self,
        x
    ):

        z = self.encoder(x)

        s1 = self.s1_head(z)

        s2 = self.s2_head(z)

        return z, s1, s2


obs_model = LatentObservationModel()

obs_model.load_state_dict(
    obs_ckpt[
        "model_state_dict"
    ],
    strict=True
)

obs_model.eval()

# ------------------------------------------------------------
# Normalization helpers
# ------------------------------------------------------------

def norm_state(x):

    return (
        x - state_mean
    ) / state_std


def norm_weather(x):

    x = np.asarray(
        x,
        dtype=np.float32
    )

    return (
        x - weather_mean
    ) / weather_std


def norm_weather_history(x):

    x = np.asarray(
        x,
        dtype=np.float32
    ).reshape(3, 3)

    x = (
        x
        - weather_mean.reshape(1, 3)
    ) / weather_std.reshape(1, 3)

    return x.reshape(-1)


# ------------------------------------------------------------
# Encode previous observed 6-D state -> latent 4-D
# ------------------------------------------------------------

def encode_state(x):

    x = torch.tensor(
        x,
        dtype=torch.float32
    ).unsqueeze(0)

    with torch.no_grad():

        z = obs_model.encoder(
            x
        )

    return (
        z.numpy()[0]
    )


# ------------------------------------------------------------
# Decode latent 4-D -> observation 6-D
#
# STATE order:
# NDVI, EVI, NDWI, VV, VH, VV_VH
#
# S2 head = NDVI, EVI, NDWI
# S1 head = VV, VH, VV_VH
# ------------------------------------------------------------

def latent_to_obs(z):

    z = torch.tensor(
        z,
        dtype=torch.float32
    ).unsqueeze(0)

    with torch.no_grad():

        s2 = obs_model.s2_head(
            z
        )

        s1 = obs_model.s1_head(
            z
        )

    out = torch.cat(
        [
            s2,
            s1
        ],
        dim=1
    )

    return (
        out.numpy()[0]
    )


# ------------------------------------------------------------
# Predict latent next state
# ------------------------------------------------------------

def predict_latent(
    z,
    weather,
    crop_id
):

    z_tensor = torch.tensor(
        z,
        dtype=torch.float32
    ).unsqueeze(0)

    w_tensor = torch.tensor(
        weather,
        dtype=torch.float32
    ).unsqueeze(0)

    c_tensor = torch.tensor(
        [crop_id],
        dtype=torch.long
    )

    with torch.no_grad():

        pred = transition(
            z_tensor,
            w_tensor,
            c_tensor
        )

    return (
        pred.numpy()[0]
    )


# ------------------------------------------------------------
# Build fixed evaluation set and ablation predictions
# ------------------------------------------------------------

eval_rows = []

all_data = data.copy()

for site, g in all_data.groupby(
    "site_id"
):

    g = (
        g.sort_values("date")
        .reset_index(drop=True)
    )

    for i in range(len(g)):

        cur = g.iloc[i]

        if cur["date"].year != 2024:
            continue

        date = cur["date"]

        prev_date = (
            date
            - pd.DateOffset(
                months=1
            )
        )

        prev = g[
            g["date"].eq(
                prev_date
            )
        ]

        if len(prev) != 1:
            continue

        prev = prev.iloc[0]

        if prev[STATE].isna().any():
            continue

        if cur[STATE].isna().any():
            continue

        wh_raw = get_weather_history(
            g,
            date
        )

        if wh_raw is None:
            continue

        # Normalize previous observed state.
        prev_norm = norm_state(
            prev[STATE]
            .values
            .astype(float)
        )

        # Encode 6-D observed state -> 4-D latent.
        z0 = encode_state(
            prev_norm
        )

        true_crop = cur["crop"]

        # ----------------------------------------------------
        # Weather shuffle.
        # Same crop and same calendar-month distributions.
        # ----------------------------------------------------

        donors = all_data[
            (all_data["crop"] == true_crop)
            &
            (
                all_data["date"]
                .dt.month
                ==
                date.month
            )
            &
            (
                all_data["date"]
                .dt.year
                .between(
                    2019,
                    2023
                )
            )
        ]

        donors = donors.dropna(
            subset=WEATHER
        )

        if len(donors) == 0:
            continue

        shuffled_parts = []

        for off in [2, 1, 0]:

            d = (
                date
                - pd.DateOffset(
                    months=off
                )
            )

            same = donors[
                donors["date"]
                .dt.month
                ==
                d.month
            ]

            if len(same) == 0:
                same = donors

            donor = same.iloc[
                rng.integers(
                    len(same)
                )
            ]

            shuffled_parts.extend(
                norm_weather(
                    donor[
                        WEATHER
                    ]
                    .values
                    .astype(float)
                )
            )

        shuffled_wh = np.asarray(
            shuffled_parts,
            dtype=np.float32
        )

        # ----------------------------------------------------
        # Deterministic cyclic crop swap.
        # ----------------------------------------------------

        crop_names = list(
            crop_to_id.keys()
        )

        crop_names = sorted(
            crop_names,
            key=lambda x:
            crop_to_id[x]
        )

        crop_idx = crop_names.index(
            true_crop
        )

        swapped_crop = crop_names[
            (
                crop_idx + 1
            )
            %
            len(crop_names)
        ]

        # ----------------------------------------------------
        # Four conditions.
        # ----------------------------------------------------

        conditions = {

            "latent": (
                norm_weather_history(
                    wh_raw
                ),
                true_crop
            ),

            "weather_shuffle": (
                shuffled_wh,
                true_crop
            ),

            "crop_swap": (
                norm_weather_history(
                    wh_raw
                ),
                swapped_crop
            ),

            "weather_shuffle_crop_swap": (
                shuffled_wh,
                swapped_crop
            ),
        }

        truth = norm_state(
            cur[STATE]
            .values
            .astype(float)
        )

        for condition, (
            weather_input,
            crop_input
        ) in conditions.items():

            pred_z = predict_latent(
                z0,
                weather_input,
                crop_to_id[
                    crop_input
                ]
            )

            pred = latent_to_obs(
                pred_z
            )

            for j, feature in enumerate(
                STATE
            ):

                error = (
                    pred[j]
                    - truth[j]
                )

                eval_rows.append({

                    "site_id": site,

                    "date": date,

                    "crop": true_crop,

                    "feature": feature,

                    "condition": condition,

                    "error": error,

                    "squared_error":
                        error ** 2,
                })


abl = pd.DataFrame(
    eval_rows
)

# ------------------------------------------------------------
# Keep only site-months where all four conditions exist.
# ------------------------------------------------------------

complete_keys = (
    abl.groupby(
        [
            "site_id",
            "date"
        ]
    )["condition"]
    .nunique()
)

complete_keys = complete_keys[
    complete_keys == 4
].index

abl = (
    abl.set_index(
        [
            "site_id",
            "date"
        ]
    )
    .loc[
        complete_keys
    ]
    .reset_index()
)

print()
print("=" * 60)
print("TEST-TIME ABLATION RMSE")
print("=" * 60)

ab_summary = (
    abl.groupby(
        "condition"
    )
    .agg(
        RMSE=(
            "squared_error",
            lambda x:
            np.sqrt(
                np.mean(x)
            )
        ),
        N=(
            "squared_error",
            "size"
        )
    )
    .reset_index()
)

latent_rmse = (
    ab_summary.loc[
        ab_summary["condition"]
        == "latent",
        "RMSE"
    ]
    .iloc[0]
)

ab_summary[
    "RMSE_change_vs_latent"
] = (
    ab_summary["RMSE"]
    - latent_rmse
)

ab_summary[
    "percent_change_vs_latent"
] = (
    ab_summary[
        "RMSE_change_vs_latent"
    ]
    / latent_rmse
    * 100
)

print(
    ab_summary.to_string(
        index=False
    )
)

print(
    "Ablation site-months:",
    abl[
        ["site_id", "date"]
    ]
    .drop_duplicates()
    .shape[0]
)

# ------------------------------------------------------------
# Save ablation results
# ------------------------------------------------------------

ab_summary.to_csv(
    "FINAL_TEST_TIME_ABLATIONS.csv",
    index=False
)

# ------------------------------------------------------------
# Anomaly persistence baseline
# ------------------------------------------------------------

# Persistence here means carrying forward the previous
# normalized observation state.

pers_rows = []

for site, g in all_data.groupby(
    "site_id"
):

    g = (
        g.sort_values("date")
        .reset_index(drop=True)
    )

    for i in range(len(g)):

        cur = g.iloc[i]

        if cur["date"].year != 2024:
            continue

        date = cur["date"]

        prev_date = (
            date
            - pd.DateOffset(
                months=1
            )
        )

        prev = g[
            g["date"].eq(
                prev_date
            )
        ]

        if len(prev) != 1:
            continue

        prev = prev.iloc[0]

        if prev[STATE].isna().any():
            continue

        if cur[STATE].isna().any():
            continue

        truth = norm_state(
            cur[STATE]
            .values
            .astype(float)
        )

        previous = norm_state(
            prev[STATE]
            .values
            .astype(float)
        )

        for j, feature in enumerate(
            STATE
        ):

            error = (
                previous[j]
                - truth[j]
            )

            pers_rows.append({

                "site_id": site,

                "date": date,

                "crop": cur["crop"],

                "feature": feature,

                "error": error,

                "squared_error":
                    error ** 2,
            })


pers = pd.DataFrame(
    pers_rows
)

pers_summary = (
    pers.groupby(
        "site_id"
    )
    .agg(
        RMSE=(
            "squared_error",
            lambda x:
            np.sqrt(
                np.mean(x)
            )
        )
    )
    .reset_index()
)

latent_site = (
    abl[
        abl["condition"]
        == "latent"
    ]
    .groupby(
        "site_id"
    )
    .agg(
        RMSE=(
            "squared_error",
            lambda x:
            np.sqrt(
                np.mean(x)
            )
        )
    )
    .reset_index()
)

site_compare = latent_site.merge(
    pers_summary,
    on="site_id",
    suffixes=(
        "_latent",
        "_persistence"
    )
)

diff = (
    site_compare["RMSE_latent"]
    -
    site_compare["RMSE_persistence"]
)

print()
print("=" * 60)
print("ANOMALY PERSISTENCE")
print("=" * 60)

print(
    "Latent RMSE:",
    site_compare[
        "RMSE_latent"
    ].mean()
)

print(
    "Anomaly-persistence RMSE:",
    site_compare[
        "RMSE_persistence"
    ].mean()
)

print(
    "Mean paired site RMSE difference:",
    diff.mean()
)

# ------------------------------------------------------------
# Site bootstrap
# ------------------------------------------------------------

boot = []

sites = (
    site_compare[
        "site_id"
    ]
    .unique()
)

for _ in range(N_BOOT):

    sampled = rng.choice(
        sites,
        size=len(sites),
        replace=True
    )

    boot.append(
        site_compare[
            site_compare[
                "site_id"
            ].isin(sampled)
        ]
        .assign(
            weight=lambda x:
            x["site_id"].map(
                pd.Series(
                    sampled
                ).value_counts()
            )
        )
        .pipe(
            lambda x:
            np.average(
                x[
                    "RMSE_latent"
                ]
                -
                x[
                    "RMSE_persistence"
                ],
                weights=x[
                    "weight"
                ]
            )
        )
    )

ci = np.percentile(
    boot,
    [2.5, 97.5]
)

print(
    "95% site-bootstrap CI:",
    ci
)

print(
    "Sites:",
    len(sites)
)

site_compare.to_csv(
    "FINAL_ANOMALY_PERSISTENCE_SITE_RESULTS.csv",
    index=False
)

print()
print("=" * 60)
print("DONE")
print("=" * 60)
