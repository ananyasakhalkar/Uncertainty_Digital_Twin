# ============================================================
# STEP3_BEYOND_SEASONALITY.py
#
# NO RETRAINING
#
# Tests whether the frozen latent digital twin adds information
# beyond:
#   1. seasonal climatology
#   2. anomaly persistence
#   3. climatological weather
#   4. shuffled weather histories
#   5. shuffled crop embeddings
#
# Builds the common evaluation table internally.
#
# INPUT FILES ARE IN PROJECT ROOT:
#   GLOBAL_ALL25_MODEL_READY.csv
#
# FROZEN MODELS:
#   models/GLOBAL_LATENT_OBSERVATION_V1.pt
#   models/GLOBAL_LATENT_TRANSITION_V1.pt
#
# ============================================================

import os
import json
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn


# ============================================================
# PATHS
# ============================================================

ROOT = os.path.expanduser(
    "~/Downloads/Crop_Digital_Twin"
)

DATA_FILE = os.path.join(
    ROOT,
    "GLOBAL_ALL25_MODEL_READY.csv"
)

OBS_MODEL_FILE = os.path.join(
    ROOT,
    "models",
    "GLOBAL_LATENT_OBSERVATION_V1.pt"
)

TRANSITION_MODEL_FILE = os.path.join(
    ROOT,
    "models",
    "GLOBAL_LATENT_TRANSITION_V1.pt"
)

OUT_TABLE = os.path.join(
    ROOT,
    "STEP3_EVAL_TABLE_194.csv"
)

OUT_RESULTS = os.path.join(
    ROOT,
    "STEP3_BEYOND_SEASONALITY_RESULTS.json"
)


# ============================================================
# CONFIG
# ============================================================

STATE_FEATURES = [
    "NDVI",
    "EVI",
    "NDWI",
    "VV",
    "VH",
    "VV_VH",
]

S1_FEATURES = [
    "VV",
    "VH",
    "VV_VH",
]

S2_FEATURES = [
    "NDVI",
    "EVI",
    "NDWI",
]

WEATHER_FEATURES = [
    "temperature_C",
    "precipitation_mm",
    "solar_MJ_m2",
]

N_BOOT = 10000
N_SHUFFLES = 200

SEED = 42

np.random.seed(SEED)
random.seed(SEED)
torch.manual_seed(SEED)


# ============================================================
# DEVICE
# ============================================================

DEVICE = torch.device(
    "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)

print("=" * 70)
print("STEP 3: BEYOND SEASONALITY")
print("=" * 70)
print("DEVICE:", DEVICE)
print()


# ============================================================
# MODEL DEFINITIONS
# ============================================================

class Encoder(nn.Module):

    def __init__(
        self,
        input_dim=6,
        hidden=32,
        latent_dim=4
    ):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(
                input_dim,
                hidden
            ),
            nn.Tanh(),
            nn.Linear(
                hidden,
                latent_dim
            )
        )

    def forward(self, x):
        return self.net(x)


class ObservationModel(nn.Module):

    def __init__(
        self,
        latent_dim=4,
        hidden=16
    ):
        super().__init__()

        # IMPORTANT:
        # The frozen checkpoint stores:
        # encoder.0.weight
        # encoder.0.bias
        # encoder.2.weight
        # encoder.2.bias
        #
        # Therefore encoder itself must be Sequential.

        self.encoder = nn.Sequential(
            nn.Linear(
                6,
                32
            ),
            nn.Tanh(),

            nn.Linear(
                32,
                latent_dim
            )
        )

        self.s1_head = nn.Sequential(
            nn.Linear(
                latent_dim,
                hidden
            ),
            nn.Tanh(),

            nn.Linear(
                hidden,
                3
            )
        )

        self.s2_head = nn.Sequential(
            nn.Linear(
                latent_dim,
                hidden
            ),
            nn.Tanh(),

            nn.Linear(
                hidden,
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

class LatentTransition(nn.Module):

    def __init__(
        self,
        state_dim=4,
        weather_dim=9,
        n_crops=5,
        crop_emb_dim=4,
        hidden=32
    ):
        super().__init__()

        self.crop_emb = nn.Embedding(
            n_crops,
            crop_emb_dim
        )

        self.net = nn.Sequential(
            nn.Linear(
                state_dim
                + weather_dim
                + crop_emb_dim,
                hidden
            ),
            nn.Tanh(),

            nn.Linear(
                hidden,
                hidden
            ),
            nn.Tanh(),

            nn.Linear(
                hidden,
                state_dim
            )
        )

    def forward(
        self,
        z,
        weather,
        crop_idx
    ):

        c = self.crop_emb(
            crop_idx
        )

        x = torch.cat(
            [
                z,
                weather,
                c
            ],
            dim=1
        )

        return self.net(x)


# ============================================================
# CHECKPOINT HELPERS
# ============================================================

def find_state_dict(
    checkpoint,
    possible_keys
):

    for key in possible_keys:

        if key in checkpoint:
            return checkpoint[key]

    return None


def load_checkpoint(
    path
):

    if not os.path.exists(path):

        raise FileNotFoundError(
            f"\nCheckpoint not found:\n{path}"
        )

    return torch.load(
        path,
        map_location="cpu",
        weights_only=False
    )


# ============================================================
# LOAD FROZEN MODELS
# ============================================================

def load_models():

    print("=" * 70)
    print("LOADING FROZEN MODELS")
    print("=" * 70)

    obs_ckpt = load_checkpoint(
        OBS_MODEL_FILE
    )

    trans_ckpt = load_checkpoint(
        TRANSITION_MODEL_FILE
    )

    print(
        "Observation checkpoint:",
        OBS_MODEL_FILE
    )

    print(
        "Transition checkpoint:",
        TRANSITION_MODEL_FILE
    )

    # ========================================================
    # OBSERVATION MODEL
    # ========================================================

    obs_model = ObservationModel()

    obs_state = obs_ckpt["model_state_dict"]

    print(
        "Observation checkpoint keys:"
    )

    for k in obs_state.keys():
        print("  ", k)

    obs_model.load_state_dict(
        obs_state,
        strict=True
    )

    obs_model = obs_model.to(
        DEVICE
    )

    obs_model.eval()

    # ========================================================
    # TRANSITION MODEL
    # ========================================================

    transition = LatentTransition()

    trans_state = trans_ckpt["model_state_dict"]

    print(
        "Transition checkpoint keys:"
    )

    for k in trans_state.keys():
        print("  ", k)

    transition.load_state_dict(
        trans_state,
        strict=True
    )

    transition = transition.to(
        DEVICE
    )

    transition.eval()

    # ========================================================
    # NORMALIZATION
    # ========================================================

    def vectorize(
        value,
        names
    ):

        if isinstance(
            value,
            dict
        ):

            return np.asarray(
                [
                    float(value[name])
                    for name in names
                ],
                dtype=np.float32
            )

        if torch.is_tensor(value):

            value = (
                value
                .detach()
                .cpu()
                .numpy()
            )

        return np.asarray(
            value,
            dtype=np.float32
        )

    # Observation checkpoint contains state normalization
    state_mean = vectorize(
        obs_ckpt["state_mean"],
        STATE_FEATURES
    )

    state_std = vectorize(
        obs_ckpt["state_std"],
        STATE_FEATURES
    )

    # Transition checkpoint should contain weather normalization.
    # Use observation checkpoint as fallback if necessary.

    if "weather_mean" in trans_ckpt:

        weather_mean = vectorize(
            trans_ckpt["weather_mean"],
            WEATHER_FEATURES
        )

    elif "weather_mean" in obs_ckpt:

        weather_mean = vectorize(
            obs_ckpt["weather_mean"],
            WEATHER_FEATURES
        )

    else:

        raise RuntimeError(
            "weather_mean not found in either checkpoint."
        )

    if "weather_std" in trans_ckpt:

        weather_std = vectorize(
            trans_ckpt["weather_std"],
            WEATHER_FEATURES
        )

    elif "weather_std" in obs_ckpt:

        weather_std = vectorize(
            obs_ckpt["weather_std"],
            WEATHER_FEATURES
        )

    else:

        raise RuntimeError(
            "weather_std not found in either checkpoint."
        )

    # ========================================================
    # CROP ENCODING
    # ========================================================

    if "crop_to_id" in obs_ckpt:

        crop_to_idx = obs_ckpt[
            "crop_to_id"
        ]

    elif "crop_to_id" in trans_ckpt:

        crop_to_idx = trans_ckpt[
            "crop_to_id"
        ]

    elif "crop_to_idx" in trans_ckpt:

        crop_to_idx = trans_ckpt[
            "crop_to_idx"
        ]

    else:

        crop_to_idx = {
            "Cotton": 0,
            "Maize": 1,
            "Rice": 2,
            "Soybean": 3,
            "Wheat": 4
        }

    print()
    print(
        "Observation weights:",
        len(obs_state),
        "tensors"
    )

    print(
        "Transition weights:",
        len(trans_state),
        "tensors"
    )

    print(
        "State features:",
        obs_ckpt.get(
            "state_features",
            STATE_FEATURES
        )
    )

    print(
        "S1 features:",
        obs_ckpt.get(
            "s1_features",
            S1_FEATURES
        )
    )

    print(
        "S2 features:",
        obs_ckpt.get(
            "s2_features",
            S2_FEATURES
        )
    )

    print(
        "Latent dimension:",
        obs_ckpt.get(
            "latent_dim",
            4
        )
    )

    print(
        "Crop encoding:",
        crop_to_idx
    )

    print(
        "State mean:",
        state_mean
    )

    print(
        "State std:",
        state_std
    )

    print(
        "Weather mean:",
        weather_mean
    )

    print(
        "Weather std:",
        weather_std
    )

    print()
    print(
        "Frozen models loaded successfully."
    )

    return (
        obs_model,
        transition,
        state_mean,
        state_std,
        weather_mean,
        weather_std,
        crop_to_idx
    )
# ============================================================
# NORMALIZATION
# ============================================================

def normalize_state(
    x,
    mean,
    std
):

    return (
        np.asarray(x, dtype=float)
        - mean
    ) / std


def normalize_weather(
    x,
    mean,
    std
):

    return (
        np.asarray(x, dtype=float)
        - mean
    ) / std


# ============================================================
# OBSERVATION DECODER
# ============================================================

def decode_latent(
    z
):

    z_tensor = torch.tensor(
        z,
        dtype=torch.float32,
        device=DEVICE
    ).reshape(1, -1)

    with torch.no_grad():

        s1 = obs_model_global.s1_head(
            z_tensor
        )

        s2 = obs_model_global.s2_head(
            z_tensor
        )

    s1 = (
        s1.detach()
        .cpu()
        .numpy()[0]
    )

    s2 = (
        s2.detach()
        .cpu()
        .numpy()[0]
    )

    out = np.zeros(
        6,
        dtype=float
    )

    # S2 order:
    # NDVI EVI NDWI
    out[0:3] = s2

    # S1 order:
    # VV VH VV_VH
    out[3:6] = s1

    return out


# ============================================================
# LATENT PREDICTION
# ============================================================

def predict_latent_observation(
    previous_state_std,
    weather_history_std,
    crop_idx
):

    x = torch.tensor(
        previous_state_std,
        dtype=torch.float32,
        device=DEVICE
    ).reshape(1, -1)

    weather = torch.tensor(
        weather_history_std,
        dtype=torch.float32,
        device=DEVICE
    ).reshape(1, -1)

    crop = torch.tensor(
        [crop_idx],
        dtype=torch.long,
        device=DEVICE
    )

    with torch.no_grad():

        z = obs_model_global.encoder(
            x
        )

        dz = transition_global(
            z,
            weather,
            crop
        )

        z_next = z + dz

        s1 = obs_model_global.s1_head(
            z_next
        )

        s2 = obs_model_global.s2_head(
            z_next
        )

    pred = np.zeros(
        6,
        dtype=float
    )

    pred[0:3] = (
        s2.detach()
        .cpu()
        .numpy()[0]
    )

    pred[3:6] = (
        s1.detach()
        .cpu()
        .numpy()[0]
    )

    return pred


# ============================================================
# RMSE
# ============================================================

def rmse(
    pred,
    obs
):

    pred = np.asarray(
        pred,
        dtype=float
    )

    obs = np.asarray(
        obs,
        dtype=float
    )

    mask = (
        np.isfinite(pred)
        &
        np.isfinite(obs)
    )

    if mask.sum() == 0:
        return np.nan

    return float(
        np.sqrt(
            np.mean(
                (
                    pred[mask]
                    -
                    obs[mask]
                ) ** 2
            )
        )
    )


def mean_rmse(
    tab,
    prefix
):

    values = []

    for f in STATE_FEATURES:

        values.append(
            rmse(
                tab[
                    f"{prefix}_{f}"
                ],
                tab[
                    f"obs_{f}"
                ]
            )
        )

    return float(
        np.nanmean(values)
    )


# ============================================================
# BUILD 194-ROW COMMON TABLE
# ============================================================

def build_common_table(
    df,
    state_mean,
    state_std,
    weather_mean,
    weather_std,
    crop_to_idx
):

    print()
    print("=" * 70)
    print("BUILDING COMMON 2024 EVALUATION TABLE")
    print("=" * 70)

    df = df.copy()

    df["date"] = pd.to_datetime(
        df["date"]
    )

    df = df.sort_values(
        [
            "site_id",
            "date"
        ]
    )

    # --------------------------------------------------------
    # 2019-2022 training climatology
    # --------------------------------------------------------

    train = df[
        df["date"].dt.year.between(
            2019,
            2022
        )
    ].copy()

    train["month_num"] = (
        train["date"].dt.month
    )

    # --------------------------------------------------------
    # Crop-month state climatology
    # --------------------------------------------------------

    state_clim = (
        train
        .groupby(
            [
                "crop",
                "month_num"
            ]
        )[STATE_FEATURES]
        .mean()
        .reset_index()
    )

    state_clim_lookup = {}

    for _, r in state_clim.iterrows():

        key = (
            r["crop"],
            int(r["month_num"])
        )

        state_clim_lookup[key] = np.asarray(
            [
                r[f]
                for f in STATE_FEATURES
            ],
            dtype=float
        )

    # --------------------------------------------------------
    # Crop-month weather climatology
    # --------------------------------------------------------

    weather_clim = (
        train
        .groupby(
            [
                "crop",
                "month_num"
            ]
        )[WEATHER_FEATURES]
        .mean()
        .reset_index()
    )

    weather_clim_lookup = {}

    for _, r in weather_clim.iterrows():

        key = (
            r["crop"],
            int(r["month_num"])
        )

        weather_clim_lookup[key] = np.asarray(
            [
                r[f]
                for f in WEATHER_FEATURES
            ],
            dtype=float
        )

    # --------------------------------------------------------
    # Build rows
    # --------------------------------------------------------

    rows = []

    test = df[
        df["date"].dt.year == 2024
    ].copy()

    for site_id, g in test.groupby(
        "site_id"
    ):

        g = g.sort_values(
            "date"
        )

        site_all = df[
            df["site_id"] == site_id
        ].sort_values(
            "date"
        )

        for _, current in g.iterrows():

            date = current["date"]

            previous_date = (
                date
                -
                pd.DateOffset(
                    months=1
                )
            )

            prev_rows = site_all[
                site_all["date"]
                ==
                previous_date
            ]

            if len(prev_rows) != 1:
                continue

            previous = (
                prev_rows.iloc[0]
            )

            current_state = np.asarray(
                [
                    current[f]
                    for f in STATE_FEATURES
                ],
                dtype=float
            )

            previous_state = np.asarray(
                [
                    previous[f]
                    for f in STATE_FEATURES
                ],
                dtype=float
            )

            # Need complete state pair
            if not (
                np.all(
                    np.isfinite(
                        current_state
                    )
                )
                and
                np.all(
                    np.isfinite(
                        previous_state
                    )
                )
            ):
                continue

            crop = current["crop"]

            if crop not in crop_to_idx:
                continue

            crop_idx = int(
                crop_to_idx[crop]
            )

            # ------------------------------------------------
            # Three-month weather history:
            #
            # t-2, t-1, t
            # ------------------------------------------------

            weather_history = []

            weather_ok = True

            for lag in [2, 1, 0]:

                d = (
                    date
                    -
                    pd.DateOffset(
                        months=lag
                    )
                )

                wr = site_all[
                    site_all["date"] == d
                ]

                if len(wr) != 1:

                    weather_ok = False
                    break

                wr = wr.iloc[0]

                vals = [
                    wr[f]
                    for f in WEATHER_FEATURES
                ]

                if not np.all(
                    np.isfinite(vals)
                ):

                    weather_ok = False
                    break

                weather_history.extend(
                    vals
                )

            if not weather_ok:
                continue

            weather_history = np.asarray(
                weather_history,
                dtype=float
            )

            # ------------------------------------------------
            # Climatological weather history
            # ------------------------------------------------

            weather_clim_history = []

            weather_clim_ok = True

            for lag in [2, 1, 0]:

                d = (
                    date
                    -
                    pd.DateOffset(
                        months=lag
                    )
                )

                key = (
                    crop,
                    int(d.month)
                )

                if key not in weather_clim_lookup:

                    weather_clim_ok = False
                    break

                weather_clim_history.extend(
                    weather_clim_lookup[key]
                )

            if not weather_clim_ok:
                continue

            weather_clim_history = np.asarray(
                weather_clim_history,
                dtype=float
            )

            # ------------------------------------------------
            # Normalize
            # ------------------------------------------------

            prev_std = normalize_state(
                previous_state,
                state_mean,
                state_std
            )

            obs_std = normalize_state(
                current_state,
                state_mean,
                state_std
            )

            wm = np.tile(
                weather_mean,
                3
            )

            ws = np.tile(
                weather_std,
                3
            )

            wx_std = (
                weather_history - wm
            ) / ws

            wxc_std = (
                weather_clim_history - wm
            ) / ws

            # ------------------------------------------------
            # Twin prediction
            # ------------------------------------------------

            pred = predict_latent_observation(
                prev_std,
                wx_std,
                crop_idx
            )

            # ------------------------------------------------
            # Twin with climatological weather
            # ------------------------------------------------

            pred_clim_weather = (
                predict_latent_observation(
                    prev_std,
                    wxc_std,
                    crop_idx
                )
            )

            # ------------------------------------------------
            # Seasonal climatology
            # ------------------------------------------------

            key = (
                crop,
                int(date.month)
            )

            prev_key = (
                crop,
                int(
                    previous_date.month
                )
            )

            if (
                key not in state_clim_lookup
                or
                prev_key not in state_clim_lookup
            ):
                continue

            clim_raw = (
                state_clim_lookup[key]
            )

            clim_prev_raw = (
                state_clim_lookup[
                    prev_key
                ]
            )

            clim_std = normalize_state(
                clim_raw,
                state_mean,
                state_std
            )

            clim_prev_std = normalize_state(
                clim_prev_raw,
                state_mean,
                state_std
            )

            # ------------------------------------------------
            # Anomaly persistence
            #
            # phi = 1
            # ------------------------------------------------

            persistence = (
                clim_std
                +
                (
                    prev_std
                    -
                    clim_prev_std
                )
            )

            # ------------------------------------------------
            # phi = 0.5
            # ------------------------------------------------

            persistence05 = (
                clim_std
                +
                0.5
                *
                (
                    prev_std
                    -
                    clim_prev_std
                )
            )

            # ------------------------------------------------
            # Weather anomalies
            # ------------------------------------------------

            actual_current_weather = (
                weather_history[6:9]
            )

            clim_current_weather = (
                weather_clim_history[6:9]
            )

            temp_anom = (
                actual_current_weather[0]
                -
                clim_current_weather[0]
            )

            precip_anom = (
                actual_current_weather[1]
                -
                clim_current_weather[1]
            )

            rec = {
                "site": site_id,
                "crop": crop,
                "crop_idx": crop_idx,
                "date": date,
                "temp_anom": temp_anom,
                "precip_anom": precip_anom,
            }

            for j, f in enumerate(
                STATE_FEATURES
            ):

                rec[
                    f"obs_{f}"
                ] = obs_std[j]

                rec[
                    f"prev_{f}"
                ] = prev_std[j]

                rec[
                    f"clim_{f}"
                ] = clim_std[j]

                rec[
                    f"clim_prev_{f}"
                ] = clim_prev_std[j]

                rec[
                    f"twin_{f}"
                ] = pred[j]

                rec[
                    f"twin_climwx_{f}"
                ] = pred_clim_weather[j]

                rec[
                    f"pers_{f}"
                ] = persistence[j]

                rec[
                    f"pers05_{f}"
                ] = persistence05[j]

            for j in range(9):

                rec[
                    f"wx_{j}"
                ] = wx_std[j]

                rec[
                    f"wxc_{j}"
                ] = wxc_std[j]

            rows.append(rec)

    tab = pd.DataFrame(
        rows
    )

    tab = tab.sort_values(
        [
            "site",
            "date"
        ]
    ).reset_index(
        drop=True
    )

    print(
        "Constructed valid site-months:",
        len(tab)
    )

    print(
        "Sites:",
        tab["site"].nunique()
    )

    # --------------------------------------------------------
    # We require the established 194-row common set.
    # If more are available, retain the same deterministic
    # common-set construction.
    # --------------------------------------------------------

    if len(tab) < 194:

        raise RuntimeError(
            f"Only {len(tab)} valid site-months "
            f"were constructed, but 194 are required."
        )

    tab = tab.iloc[
        :194
    ].copy()

    tab = tab.reset_index(
        drop=True
    )

    print()
    print(
        "FINAL COMMON SET:"
    )

    print(
        "Site-months:",
        len(tab)
    )

    print(
        "Sites:",
        tab["site"].nunique()
    )

    print(
        "Feature evaluations:",
        len(tab) * 6
    )

    print(
        "Date range:",
        tab["date"].min(),
        "to",
        tab["date"].max()
    )

    return tab


# ============================================================
# SITE BOOTSTRAP
# ============================================================

def paired_site_bootstrap(
    tab,
    pred_a,
    pred_b,
    feature=None,
    n_boot=10000
):

    sites = sorted(
        tab["site"].unique()
    )

    site_diffs = []

    for site in sites:

        sub = tab[
            tab["site"] == site
        ]

        if feature is None:

            a = mean_rmse(
                sub,
                pred_a
            )

            b = mean_rmse(
                sub,
                pred_b
            )

        else:

            a = rmse(
                sub[
                    f"{pred_a}_{feature}"
                ],
                sub[
                    f"obs_{feature}"
                ]
            )

            b = rmse(
                sub[
                    f"{pred_b}_{feature}"
                ],
                sub[
                    f"obs_{feature}"
                ]
            )

        site_diffs.append(
            a - b
        )

    site_diffs = np.asarray(
        site_diffs,
        dtype=float
    )

    rng = np.random.default_rng(
        SEED
    )

    boot = np.zeros(
        n_boot
    )

    for i in range(
        n_boot
    ):

        sample = rng.choice(
            site_diffs,
            size=len(site_diffs),
            replace=True
        )

        boot[i] = np.mean(
            sample
        )

    return {
        "difference": float(
            np.mean(site_diffs)
        ),
        "ci_low": float(
            np.percentile(
                boot,
                2.5
            )
        ),
        "ci_high": float(
            np.percentile(
                boot,
                97.5
            )
        ),
        "n_sites": int(
            len(sites)
        )
    }


# ============================================================
# TEST-TIME WEATHER SHUFFLE
# ============================================================

def weather_shuffle_predictions(
    tab,
    n_repeats=200
):

    rng = np.random.default_rng(
        SEED
    )

    results = []

    for repeat in range(
        n_repeats
    ):

        shuffled = tab.copy()

        # Shuffle weather histories within crop
        # and calendar month.

        for (
            crop,
            month
        ), idx in shuffled.groupby(
            [
                "crop",
                shuffled["date"].dt.month
            ]
        ).groups.items():

            idx = list(idx)

            if len(idx) < 2:
                continue

            source = shuffled.loc[
                idx,
                [
                    f"wx_{i}"
                    for i in range(9)
                ]
            ].values.copy()

            rng.shuffle(
                source
            )

            shuffled.loc[
                idx,
                [
                    f"wx_{i}"
                    for i in range(9)
                ]
            ] = source

        # Recalculate twin using shuffled weather
        # while retaining actual previous state.

        for row_idx, row in shuffled.iterrows():

            prev = np.asarray(
                [
                    row[
                        f"prev_{f}"
                    ]
                    for f in STATE_FEATURES
                ],
                dtype=float
            )

            wx = np.asarray(
                [
                    row[
                        f"wx_{i}"
                    ]
                    for i in range(9)
                ],
                dtype=float
            )

            pred = predict_latent_observation(
                prev,
                wx,
                int(
                    row["crop_idx"]
                )
            )

            for j, f in enumerate(
                STATE_FEATURES
            ):

                shuffled.loc[
                    row_idx,
                    f"shuffle_{f}"
                ] = pred[j]

        score = mean_rmse(
            shuffled,
            "shuffle"
        )

        results.append(
            score
        )

    return np.asarray(
        results,
        dtype=float
    )


# ============================================================
# TEST-TIME CROP SWAP
# ============================================================

def crop_swap_predictions(
    tab,
    n_repeats=200
):

    rng = np.random.default_rng(
        SEED
    )

    results = []

    crops = sorted(
        tab["crop"].unique()
    )

    crop_to_idx = {
        crop: i
        for i, crop in enumerate(crops)
    }

    for repeat in range(
        n_repeats
    ):

        swapped = tab.copy()

        # Within each calendar month, randomly
        # assign another crop embedding.

        for month, idx in swapped.groupby(
            swapped["date"].dt.month
        ).groups.items():

            idx = list(idx)

            crop_indices = (
                swapped.loc[
                    idx,
                    "crop_idx"
                ].values.copy()
            )

            rng.shuffle(
                crop_indices
            )

            swapped.loc[
                idx,
                "swapped_crop_idx"
            ] = crop_indices

        for row_idx, row in swapped.iterrows():

            prev = np.asarray(
                [
                    row[
                        f"prev_{f}"
                    ]
                    for f in STATE_FEATURES
                ],
                dtype=float
            )

            wx = np.asarray(
                [
                    row[
                        f"wx_{i}"
                    ]
                    for i in range(9)
                ],
                dtype=float
            )

            crop_idx = int(
                row[
                    "swapped_crop_idx"
                ]
            )

            pred = predict_latent_observation(
                prev,
                wx,
                crop_idx
            )

            for j, f in enumerate(
                STATE_FEATURES
            ):

                swapped.loc[
                    row_idx,
                    f"swap_{f}"
                ] = pred[j]

        score = mean_rmse(
            swapped,
            "swap"
        )

        results.append(
            score
        )

    return np.asarray(
        results,
        dtype=float
    )


# ============================================================
# BOTH ABLATIONS
# ============================================================

def weather_and_crop_swap(
    tab,
    n_repeats=200
):

    rng = np.random.default_rng(
        SEED
    )

    results = []

    for repeat in range(
        n_repeats
    ):

        modified = tab.copy()

        # Weather shuffle within crop/month
        for (
            crop,
            month
        ), idx in modified.groupby(
            [
                "crop",
                modified["date"].dt.month
            ]
        ).groups.items():

            idx = list(idx)

            if len(idx) >= 2:

                source = modified.loc[
                    idx,
                    [
                        f"wx_{i}"
                        for i in range(9)
                    ]
                ].values.copy()

                rng.shuffle(
                    source
                )

                modified.loc[
                    idx,
                    [
                        f"wx_{i}"
                        for i in range(9)
                    ]
                ] = source

        # Crop shuffle within calendar month
        for month, idx in modified.groupby(
            modified["date"].dt.month
        ).groups.items():

            idx = list(idx)

            crop_idx = modified.loc[
                idx,
                "crop_idx"
            ].values.copy()

            rng.shuffle(
                crop_idx
            )

            modified.loc[
                idx,
                "swap_crop_idx"
            ] = crop_idx

        for row_idx, row in modified.iterrows():

            prev = np.asarray(
                [
                    row[
                        f"prev_{f}"
                    ]
                    for f in STATE_FEATURES
                ],
                dtype=float
            )

            wx = np.asarray(
                [
                    row[
                        f"wx_{i}"
                    ]
                    for i in range(9)
                ],
                dtype=float
            )

            crop_idx = int(
                row[
                    "swap_crop_idx"
                ]
            )

            pred = predict_latent_observation(
                prev,
                wx,
                crop_idx
            )

            for j, f in enumerate(
                STATE_FEATURES
            ):

                modified.loc[
                    row_idx,
                    f"both_{f}"
                ] = pred[j]

        results.append(
            mean_rmse(
                modified,
                "both"
            )
        )

    return np.asarray(
        results,
        dtype=float
    )


# ============================================================
# WEATHER ANOMALY STRATA
# ============================================================

def anomaly_strata(
    tab
):

    results = {}

    for name in [
        "temp_anom",
        "precip_anom"
    ]:

        values = np.abs(
            tab[name].values
        )

        q1, q2 = np.quantile(
            values,
            [
                1 / 3,
                2 / 3
            ]
        )

        labels = np.where(
            values <= q1,
            "low",
            np.where(
                values <= q2,
                "medium",
                "high"
            )
        )

        results[name] = {}

        for level in [
            "low",
            "medium",
            "high"
        ]:

            mask = (
                labels == level
            )

            sub = tab.loc[
                mask
            ]

            if len(sub) == 0:
                continue

            twin = mean_rmse(
                sub,
                "twin"
            )

            clim = mean_rmse(
                sub,
                "clim"
            )

            results[name][level] = {
                "n": int(
                    len(sub)
                ),
                "twin_rmse": twin,
                "climatology_rmse": clim,
                "twin_minus_climatology": (
                    twin - clim
                )
            }

    return results


# ============================================================
# MAIN
# ============================================================

def main():

    global obs_model_global
    global transition_global

    # --------------------------------------------------------
    # Verify files
    # --------------------------------------------------------

    for path in [
        DATA_FILE,
        OBS_MODEL_FILE,
        TRANSITION_MODEL_FILE
    ]:

        if not os.path.exists(path):

            raise FileNotFoundError(
                f"\nRequired file not found:\n{path}"
            )

    print(
        "Data:",
        DATA_FILE
    )

    print(
        "Observation model:",
        OBS_MODEL_FILE
    )

    print(
        "Transition model:",
        TRANSITION_MODEL_FILE
    )

    print()

    # --------------------------------------------------------
    # Load data
    # --------------------------------------------------------

    df = pd.read_csv(
        DATA_FILE
    )

    print(
        "Input rows:",
        len(df)
    )

    print(
        "Input sites:",
        df["site_id"].nunique()
    )

    print(
        "Input crops:",
        df["crop"].nunique()
    )

    # --------------------------------------------------------
    # Load frozen models
    # --------------------------------------------------------

    (
        obs_model_global,
        transition_global,
        state_mean,
        state_std,
        weather_mean,
        weather_std,
        crop_to_idx
    ) = load_models()

    # --------------------------------------------------------
    # Build common table
    # --------------------------------------------------------

    tab = build_common_table(
        df,
        state_mean,
        state_std,
        weather_mean,
        weather_std,
        crop_to_idx
    )

    # --------------------------------------------------------
    # Save table
    # --------------------------------------------------------

    tab.to_csv(
        OUT_TABLE,
        index=False
    )

    print()
    print(
        "Saved:",
        OUT_TABLE
    )

    # --------------------------------------------------------
    # Sanity check
    #
    # Re-run frozen latent model on each row and compare
    # against stored twin prediction.
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("LATENT PREDICTION SANITY CHECK")
    print("=" * 70)

    max_diff = 0.0

    for _, row in tab.iterrows():

        prev = np.asarray(
            [
                row[
                    f"prev_{f}"
                ]
                for f in STATE_FEATURES
            ],
            dtype=float
        )

        wx = np.asarray(
            [
                row[
                    f"wx_{i}"
                ]
                for i in range(9)
            ],
            dtype=float
        )

        pred = predict_latent_observation(
            prev,
            wx,
            int(
                row["crop_idx"]
            )
        )

        stored = np.asarray(
            [
                row[
                    f"twin_{f}"
                ]
                for f in STATE_FEATURES
            ],
            dtype=float
        )

        diff = np.max(
            np.abs(
                pred - stored
            )
        )

        max_diff = max(
            max_diff,
            diff
        )

    print(
        "Maximum prediction difference:",
        max_diff
    )

    if max_diff > 1e-5:

        raise RuntimeError(
            "SANITY CHECK FAILED. "
            "Stored twin predictions do not reproduce "
            "from the frozen models."
        )

    print(
        "SANITY CHECK PASSED."
    )

    # --------------------------------------------------------
    # Reference RMSEs
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("REFERENCE RMSE")
    print("=" * 70)

    twin_rmse = mean_rmse(
        tab,
        "twin"
    )

    clim_rmse = mean_rmse(
        tab,
        "clim"
    )

    pers_rmse = mean_rmse(
        tab,
        "pers"
    )

    print(
        f"Twin          : {twin_rmse:.6f}"
    )

    print(
        f"Climatology   : {clim_rmse:.6f}"
    )

    print(
        f"Persistence   : {pers_rmse:.6f}"
    )

    print()
    print("Feature RMSE:")

    feature_results = {}

    for f in STATE_FEATURES:

        t = rmse(
            tab[
                f"twin_{f}"
            ],
            tab[
                f"obs_{f}"
            ]
        )

        c = rmse(
            tab[
                f"clim_{f}"
            ],
            tab[
                f"obs_{f}"
            ]
        )

        p = rmse(
            tab[
                f"pers_{f}"
            ],
            tab[
                f"obs_{f}"
            ]
        )

        feature_results[f] = {
            "twin": t,
            "climatology": c,
            "persistence": p
        }

        print(
            f"{f:8s} "
            f"twin={t:.6f} "
            f"clim={c:.6f} "
            f"pers={p:.6f}"
        )

    # --------------------------------------------------------
    # Bootstrap
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("PAIRED SITE BOOTSTRAP")
    print("=" * 70)

    bootstrap_results = {}

    for baseline_name, baseline_prefix in [
        (
            "climatology",
            "clim"
        ),
        (
            "persistence",
            "pers"
        )
    ]:

        result = paired_site_bootstrap(
            tab,
            "twin",
            baseline_prefix,
            feature=None,
            n_boot=N_BOOT
        )

        bootstrap_results[
            f"twin_minus_{baseline_name}"
        ] = result

        print(
            f"Twin - {baseline_name}: "
            f"{result['difference']:.6f} "
            f"[{result['ci_low']:.6f}, "
            f"{result['ci_high']:.6f}]"
        )

        for f in STATE_FEATURES:

            r = paired_site_bootstrap(
                tab,
                "twin",
                baseline_prefix,
                feature=f,
                n_boot=N_BOOT
            )

            bootstrap_results[
                f"{f}_twin_minus_{baseline_name}"
            ] = r

            print(
                f"  {f:8s}: "
                f"{r['difference']:.6f} "
                f"[{r['ci_low']:.6f}, "
                f"{r['ci_high']:.6f}]"
            )

    # --------------------------------------------------------
    # Weather shuffle
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("TEST-TIME WEATHER SHUFFLE")
    print("=" * 70)

    shuffled_scores = (
        weather_shuffle_predictions(
            tab,
            N_SHUFFLES
        )
    )

    print(
        "Original twin RMSE:",
        twin_rmse
    )

    print(
        "Weather-shuffled RMSE:",
        np.mean(
            shuffled_scores
        )
    )

    print(
        "Weather-shuffled 95% range:",
        np.percentile(
            shuffled_scores,
            2.5
        ),
        np.percentile(
            shuffled_scores,
            97.5
        )
    )

    # --------------------------------------------------------
    # Crop swap
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("TEST-TIME CROP-EMBEDDING SWAP")
    print("=" * 70)

    crop_swap_scores = (
        crop_swap_predictions(
            tab,
            N_SHUFFLES
        )
    )

    print(
        "Original twin RMSE:",
        twin_rmse
    )

    print(
        "Crop-swapped RMSE:",
        np.mean(
            crop_swap_scores
        )
    )

    print(
        "Crop-swapped 95% range:",
        np.percentile(
            crop_swap_scores,
            2.5
        ),
        np.percentile(
            crop_swap_scores,
            97.5
        )
    )

    # --------------------------------------------------------
    # Both
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("WEATHER + CROP SWAP")
    print("=" * 70)

    both_scores = (
        weather_and_crop_swap(
            tab,
            N_SHUFFLES
        )
    )

    print(
        "Original twin RMSE:",
        twin_rmse
    )

    print(
        "Both shuffled RMSE:",
        np.mean(
            both_scores
        )
    )

    print(
        "Both shuffled 95% range:",
        np.percentile(
            both_scores,
            2.5
        ),
        np.percentile(
            both_scores,
            97.5
        )
    )

    # --------------------------------------------------------
    # Climatological weather
    # --------------------------------------------------------

    climwx_rmse = mean_rmse(
        tab,
        "twin_climwx"
    )

    print()
    print("=" * 70)
    print("CLIMATOLOGICAL WEATHER ABLATION")
    print("=" * 70)

    print(
        "Actual weather RMSE:",
        twin_rmse
    )

    print(
        "Climatological weather RMSE:",
        climwx_rmse
    )

    print(
        "Difference:",
        climwx_rmse - twin_rmse
    )

    # --------------------------------------------------------
    # Weather anomaly strata
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("WEATHER ANOMALY STRATA")
    print("=" * 70)

    strata_results = anomaly_strata(
        tab
    )

    for variable, levels in (
        strata_results.items()
    ):

        print()
        print(
            variable
        )

        for level, r in levels.items():

            print(
                f"  {level:8s} "
                f"N={r['n']:3d} "
                f"twin={r['twin_rmse']:.6f} "
                f"clim={r['climatology_rmse']:.6f} "
                f"delta={r['twin_minus_climatology']:.6f}"
            )

    # --------------------------------------------------------
    # Save results
    # --------------------------------------------------------

    results = {

        "evaluation": {
            "site_months": int(
                len(tab)
            ),
            "sites": int(
                tab["site"].nunique()
            ),
            "feature_evaluations": int(
                len(tab) * 6
            ),
            "date_min": str(
                tab["date"].min()
            ),
            "date_max": str(
                tab["date"].max()
            )
        },

        "rmse": {
            "twin": twin_rmse,
            "climatology": clim_rmse,
            "persistence": pers_rmse,
            "twin_climatological_weather":
                climwx_rmse
        },

        "feature_rmse":
            feature_results,

        "paired_site_bootstrap":
            bootstrap_results,

        "weather_shuffle": {
            "n_repeats": N_SHUFFLES,
            "mean_rmse": float(
                np.mean(
                    shuffled_scores
                )
            ),
            "ci_low": float(
                np.percentile(
                    shuffled_scores,
                    2.5
                )
            ),
            "ci_high": float(
                np.percentile(
                    shuffled_scores,
                    97.5
                )
            )
        },

        "crop_swap": {
            "n_repeats": N_SHUFFLES,
            "mean_rmse": float(
                np.mean(
                    crop_swap_scores
                )
            ),
            "ci_low": float(
                np.percentile(
                    crop_swap_scores,
                    2.5
                )
            ),
            "ci_high": float(
                np.percentile(
                    crop_swap_scores,
                    97.5
                )
            )
        },

        "weather_and_crop_swap": {
            "n_repeats": N_SHUFFLES,
            "mean_rmse": float(
                np.mean(
                    both_scores
                )
            ),
            "ci_low": float(
                np.percentile(
                    both_scores,
                    2.5
                )
            ),
            "ci_high": float(
                np.percentile(
                    both_scores,
                    97.5
                )
            )
        },

        "weather_anomaly_strata":
            strata_results,

        "sanity_check": {
            "max_prediction_difference":
                float(max_diff),
            "passed":
                bool(
                    max_diff <= 1e-5
                )
        }
    }

    with open(
        OUT_RESULTS,
        "w"
    ) as f:

        json.dump(
            results,
            f,
            indent=2
        )

    print()
    print("=" * 70)
    print("DONE")
    print("=" * 70)

    print(
        "Evaluation table:",
        OUT_TABLE
    )

    print(
        "Results:",
        OUT_RESULTS
    )


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    main()