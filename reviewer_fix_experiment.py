import pandas as pd
import numpy as np
import torch
import torch.nn as nn

DEVICE = torch.device(
    "mps" if torch.backends.mps.is_available() else "cpu"
)

DATA = "GLOBAL_ALL25_MODEL_READY.csv"

STATE = [
    "NDVI",
    "EVI",
    "NDWI",
    "VV",
    "VH",
    "VV_VH"
]

WEATHER = [
    "temperature_C",
    "precipitation_mm",
    "solar_MJ_m2"
]

LATENT_OBS = "models/GLOBAL_LATENT_OBSERVATION_V1.pt"
LATENT_TRANS = "models/GLOBAL_LATENT_TRANSITION_V1.pt"

SPATIAL_TRAIN = "GLOBAL_SPATIAL_TRAIN.csv"
SPATIAL_TEST = "GLOBAL_SPATIAL_TEST.csv"


# ============================================================
# LOAD DATA
# ============================================================

df = pd.read_csv(DATA)
df["date"] = pd.to_datetime(df["date"])

df = df.sort_values(
    ["site_id", "date"]
).reset_index(drop=True)

obs_ckpt = torch.load(
    LATENT_OBS,
    map_location="cpu",
    weights_only=False
)

trans_ckpt = torch.load(
    LATENT_TRANS,
    map_location="cpu",
    weights_only=False
)


# ============================================================
# MODEL
# ============================================================

class Encoder(nn.Module):

    def __init__(self):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Linear(6, 32),
            nn.Tanh(),
            nn.Linear(32, 4)
        )

    def forward(self, x):
        return self.encoder(x)


class ObservationModel(nn.Module):

    def __init__(self):
        super().__init__()

        self.encoder = Encoder().encoder

        self.s1_head = nn.Sequential(
            nn.Linear(4, 16),
            nn.Tanh(),
            nn.Linear(16, 3)
        )

        self.s2_head = nn.Sequential(
            nn.Linear(4, 16),
            nn.Tanh(),
            nn.Linear(16, 3)
        )


class LatentTransition(nn.Module):

    def __init__(self):
        super().__init__()

        self.crop_embedding = nn.Embedding(5, 4)

        self.net = nn.Sequential(
            nn.Linear(17, 32),
            nn.Tanh(),
            nn.Linear(32, 32),
            nn.Tanh(),
            nn.Linear(32, 4)
        )

    def forward(self, z, weather, crop):

        emb = self.crop_embedding(crop)

        x = torch.cat(
            [z, weather, emb],
            dim=1
        )

        return self.net(x)


obs_model = ObservationModel().to(DEVICE)

# Load encoder/head checkpoint flexibly
obs_state = obs_ckpt.get(
    "model_state_dict",
    obs_ckpt
)

obs_model.load_state_dict(
    obs_state,
    strict=True
)

obs_model.eval()


transition = LatentTransition().to(DEVICE)

transition.load_state_dict(
    trans_ckpt["model_state_dict"]
)

transition.eval()


crop_to_id = trans_ckpt["crop_to_id"]


# ============================================================
# STANDARDIZATION
# ============================================================

state_mean = np.array([
    obs_ckpt["state_mean"][f]
    for f in STATE
], dtype=np.float32)

state_std = np.array([
    obs_ckpt["state_std"][f]
    for f in STATE
], dtype=np.float32)

weather_mean = np.array([
    trans_ckpt["weather_mean"][f]
    for f in WEATHER
], dtype=np.float32)

weather_std = np.array([
    trans_ckpt["weather_std"][f]
    for f in WEATHER
], dtype=np.float32)


def norm_state(values):

    return (
        values.astype(np.float32)
        - state_mean
    ) / state_std


def norm_weather(values):

    return (
        values.astype(np.float32)
        - weather_mean
    ) / weather_std


# ============================================================
# LATENT ONE-STEP PREDICTION
# ============================================================

def predict_one(cur, prev, prev2):

    previous = norm_state(
        prev[STATE].values
    )

    with torch.no_grad():

        x = torch.tensor(
            previous,
            dtype=torch.float32,
            device=DEVICE
        ).unsqueeze(0)

        z = obs_model.encoder(x)

    weather = np.concatenate([
        norm_weather(
            cur[WEATHER].values
        ),
        norm_weather(
            prev[WEATHER].values
        ),
        norm_weather(
            prev2[WEATHER].values
        )
    ])

    w = torch.tensor(
        weather,
        dtype=torch.float32,
        device=DEVICE
    ).unsqueeze(0)

    crop = torch.tensor(
        [crop_to_id[cur["crop"]]],
        dtype=torch.long,
        device=DEVICE
    )

    with torch.no_grad():

        delta = transition(
            z,
            w,
            crop
        )

        z_next = z + delta

        s2 = obs_model.s2_head(z_next)
        s1 = obs_model.s1_head(z_next)

    # Observation order:
    # S2 = NDVI EVI NDWI
    # S1 = VV VH VV_VH

    pred = torch.cat(
        [s2, s1],
        dim=1
    ).cpu().numpy()[0]

    # Reorder to STATE:
    # NDVI EVI NDWI VV VH VV_VH
    return pred


# ============================================================
# HELPER
# ============================================================

def valid_complete(row):

    return (
        not row[STATE].isna().any()
        and not row[WEATHER].isna().any()
    )


def consecutive_month(a, b):

    return (
        b["date"]
        == a["date"] + pd.DateOffset(months=1)
    )


# ============================================================
# PART 1
# TEMPORAL BASELINES
# ============================================================

print("\n==========================================")
print("TEMPORAL BASELINES")
print("==========================================")

train_temporal = df[
    df["date"] <= pd.Timestamp("2022-12-01")
].copy()

test_temporal = df[
    df["date"] >= pd.Timestamp("2024-01-01")
].copy()

# Training climatology:
# site × month-of-year
clim_site = (
    train_temporal
    .assign(month=train_temporal["date"].dt.month)
    .groupby(
        ["site_id", "month"]
    )[STATE]
    .mean()
)

# Crop × month fallback
clim_crop = (
    train_temporal
    .assign(month=train_temporal["date"].dt.month)
    .groupby(
        ["crop", "month"]
    )[STATE]
    .mean()
)

# Global month fallback
clim_global = (
    train_temporal
    .assign(month=train_temporal["date"].dt.month)
    .groupby("month")[STATE]
    .mean()
)


temporal_rows = []

for site, g in df.groupby("site_id"):

    g = g.sort_values("date").reset_index(drop=True)

    for i in range(len(g)):

        cur = g.iloc[i]

        if not (
            pd.Timestamp("2024-01-01")
            <= cur["date"]
            <= pd.Timestamp("2024-10-01")
        ):
            continue

        if cur[STATE].isna().any():
            continue

        # Target
        obs = norm_state(
            cur[STATE].values
        )

        month = cur["date"].month

        # -----------------------------
        # Site climatology
        # -----------------------------

        key = (site, month)

        if key in clim_site.index:

            clim_values = clim_site.loc[
                key
            ][STATE].values.astype(np.float32)

        else:

            clim_values = clim_crop.loc[
                (cur["crop"], month)
            ][STATE].values.astype(np.float32)

        clim_pred = norm_state(
            clim_values
        )

        # -----------------------------
        # Persistence
        # -----------------------------

        persistence_ok = False

        if i > 0:

            prev = g.iloc[i-1]

            if (
                prev["date"]
                == cur["date"]
                - pd.DateOffset(months=1)
                and
                not prev[STATE].isna().any()
            ):

                persistence_pred = norm_state(
                    prev[STATE].values
                )

                persistence_ok = True

        # -----------------------------
        # Learned latent model
        # -----------------------------

        latent_ok = False

        if i >= 2:

            prev = g.iloc[i-1]
            prev2 = g.iloc[i-2]

            if (
                cur["date"]
                == prev["date"]
                + pd.DateOffset(months=1)
                and
                prev["date"]
                == prev2["date"]
                + pd.DateOffset(months=1)
                and
                valid_complete(prev)
                and
                valid_complete(prev2)
                and
                valid_complete(cur)
            ):

                latent_pred = predict_one(
                    cur,
                    prev,
                    prev2
                )

                latent_ok = True

        for j, f in enumerate(STATE):

            row = {
                "site_id": site,
                "crop": cur["crop"],
                "country": cur["country"],
                "date": cur["date"],
                "feature": f,
                "observation": obs[j],
                "climatology_error": (
                    clim_pred[j] - obs[j]
                ),
                "persistence_error": np.nan,
                "latent_error": np.nan
            }

            if persistence_ok:

                row["persistence_error"] = (
                    persistence_pred[j]
                    - obs[j]
                )

            if latent_ok:

                row["latent_error"] = (
                    latent_pred[j]
                    - obs[j]
                )

            temporal_rows.append(row)


temporal = pd.DataFrame(
    temporal_rows
)


def rmse(series):

    x = series.dropna().values

    return np.sqrt(
        np.mean(x**2)
    )


temporal_summary = []

for f in STATE:

    d = temporal[
        temporal["feature"] == f
    ]

    temporal_summary.append({
        "feature": f,
        "climatology_RMSE": rmse(
            d["climatology_error"]
        ),
        "persistence_RMSE": rmse(
            d["persistence_error"]
        ),
        "latent_RMSE": rmse(
            d["latent_error"]
        ),
        "N_climatology": d[
            "climatology_error"
        ].notna().sum(),
        "N_persistence": d[
            "persistence_error"
        ].notna().sum(),
        "N_latent": d[
            "latent_error"
        ].notna().sum()
    })


temporal_summary = pd.DataFrame(
    temporal_summary
)

print(
    temporal_summary.to_string(
        index=False
    )
)

print("\nMean RMSE:")

print(
    temporal_summary[
        [
            "climatology_RMSE",
            "persistence_RMSE",
            "latent_RMSE"
        ]
    ].mean()
)

temporal.to_csv(
    "REVIEWER_TEMPORAL_BASELINES.csv",
    index=False
)

temporal_summary.to_csv(
    "REVIEWER_TEMPORAL_BASELINE_SUMMARY.csv",
    index=False
)


# ============================================================
# PART 2
# SPATIAL BASELINES
# ============================================================

print("\n==========================================")
print("SPATIAL BASELINES")
print("==========================================")

sp_train = pd.read_csv(
    SPATIAL_TRAIN
)

sp_test = pd.read_csv(
    SPATIAL_TEST
)

sp_train["date"] = pd.to_datetime(
    sp_train["date"]
)

sp_test["date"] = pd.to_datetime(
    sp_test["date"]
)


# IMPORTANT:
# For spatial holdout, held-out-site historical
# observations are NOT used for climatology.
#
# We therefore construct crop × month climatology
# from the 20 training sites only.

sp_train["month"] = (
    sp_train["date"].dt.month
)

sp_test["month"] = (
    sp_test["date"].dt.month
)

sp_clim = (
    sp_train
    .groupby(
        ["crop", "month"]
    )[STATE]
    .mean()
)


spatial_rows = []

for site, g in sp_test.groupby("site_id"):

    g = g.sort_values(
        "date"
    ).reset_index(drop=True)

    for i in range(len(g)):

        cur = g.iloc[i]

        if cur[STATE].isna().any():
            continue

        obs = norm_state(
            cur[STATE].values
        )

        month = cur["date"].month

        key = (
            cur["crop"],
            month
        )

        if key not in sp_clim.index:
            continue

        clim_pred = norm_state(
            sp_clim.loc[key][STATE]
            .values.astype(np.float32)
        )

        persistence_pred = None

        if i > 0:

            prev = g.iloc[i-1]

            if (
                prev["date"]
                ==
                cur["date"]
                - pd.DateOffset(months=1)
                and
                not prev[STATE].isna().any()
            ):

                persistence_pred = norm_state(
                    prev[STATE].values
                )

        for j, f in enumerate(STATE):

            row = {
                "site_id": site,
                "crop": cur["crop"],
                "country": cur["country"],
                "date": cur["date"],
                "feature": f,
                "climatology_error":
                    clim_pred[j] - obs[j],
                "persistence_error":
                    np.nan
            }

            if persistence_pred is not None:

                row["persistence_error"] = (
                    persistence_pred[j]
                    - obs[j]
                )

            spatial_rows.append(row)


spatial = pd.DataFrame(
    spatial_rows
)

spatial_summary = []

for f in STATE:

    d = spatial[
        spatial["feature"] == f
    ]

    spatial_summary.append({
        "feature": f,
        "climatology_RMSE": rmse(
            d["climatology_error"]
        ),
        "persistence_RMSE": rmse(
            d["persistence_error"]
        ),
        "N_climatology": d[
            "climatology_error"
        ].notna().sum(),
        "N_persistence": d[
            "persistence_error"
        ].notna().sum()
    })


spatial_summary = pd.DataFrame(
    spatial_summary
)

print(
    spatial_summary.to_string(
        index=False
    )
)

print("\nMean RMSE:")

print(
    spatial_summary[
        [
            "climatology_RMSE",
            "persistence_RMSE"
        ]
    ].mean()
)

spatial.to_csv(
    "REVIEWER_SPATIAL_BASELINES.csv",
    index=False
)

spatial_summary.to_csv(
    "REVIEWER_SPATIAL_BASELINE_SUMMARY.csv",
    index=False
)


# ============================================================
# PART 3
# OUT-OF-SAMPLE 2023 CALIBRATION
# ============================================================

print("\n==========================================")
print("2023 OUT-OF-SAMPLE ERROR CALIBRATION")
print("==========================================")

calibration_rows = []

for site, g in df.groupby("site_id"):

    g = g.sort_values(
        "date"
    ).reset_index(drop=True)

    for i in range(2, len(g)):

        cur = g.iloc[i]
        prev = g.iloc[i-1]
        prev2 = g.iloc[i-2]

        if not (
            pd.Timestamp("2023-01-01")
            <= cur["date"]
            <= pd.Timestamp("2023-12-01")
        ):
            continue

        if not (
            cur["date"]
            == prev["date"]
            + pd.DateOffset(months=1)
            and
            prev["date"]
            == prev2["date"]
            + pd.DateOffset(months=1)
        ):
            continue

        if not (
            valid_complete(cur)
            and
            valid_complete(prev)
            and
            valid_complete(prev2)
        ):
            continue

        pred = predict_one(
            cur,
            prev,
            prev2
        )

        obs = norm_state(
            cur[STATE].values
        )

        error = np.abs(
            pred - obs
        )

        for j, f in enumerate(STATE):

            calibration_rows.append({
                "site_id": site,
                "crop": cur["crop"],
                "country": cur["country"],
                "date": cur["date"],
                "feature": f,
                "absolute_error": error[j]
            })


calibration = pd.DataFrame(
    calibration_rows
)

calibration_summary = (
    calibration
    .groupby("feature")["absolute_error"]
    .agg(
        median="median",
        MAD=lambda x: np.median(
            np.abs(
                x - np.median(x)
            )
        ),
        p90=lambda x: np.quantile(
            x, 0.90
        ),
        p95=lambda x: np.quantile(
            x, 0.95
        )
    )
    .reset_index()
)

calibration_summary["robust_scale"] = (
    1.4826 *
    calibration_summary["MAD"]
)

print(
    calibration_summary.to_string(
        index=False
    )
)

print(
    "\n2023 calibration rows:",
    len(calibration)
)

print(
    "2023 site-months:",
    calibration[
        ["site_id","date"]
    ].drop_duplicates().shape[0]
)


# ============================================================
# PART 4
# APPLY 2023 CALIBRATION TO 2023 AND 2024
# ============================================================

# Convert calibration values to lookup
scale_lookup = dict(
    zip(
        calibration_summary["feature"],
        calibration_summary["robust_scale"]
    )
)

p95_lookup = dict(
    zip(
        calibration_summary["feature"],
        calibration_summary["p95"]
    )
)


def evaluate_deviation_period(
    start_date,
    end_date,
    period_name
):

    rows = []

    for site, g in df.groupby("site_id"):

        g = g.sort_values(
            "date"
        ).reset_index(drop=True)

        for i in range(2, len(g)):

            cur = g.iloc[i]
            prev = g.iloc[i-1]
            prev2 = g.iloc[i-2]

            if not (
                start_date
                <= cur["date"]
                <= end_date
            ):
                continue

            if not (
                cur["date"]
                == prev["date"]
                + pd.DateOffset(months=1)
                and
                prev["date"]
                == prev2["date"]
                + pd.DateOffset(months=1)
            ):
                continue

            if not (
                valid_complete(cur)
                and
                valid_complete(prev)
                and
                valid_complete(prev2)
            ):
                continue

            pred = predict_one(
                cur,
                prev,
                prev2
            )

            obs = norm_state(
                cur[STATE].values
            )

            error = np.abs(
                pred - obs
            )

            for j, f in enumerate(STATE):

                scale = scale_lookup[f]
                p95 = p95_lookup[f]

                rows.append({
                    "period": period_name,
                    "site_id": site,
                    "crop": cur["crop"],
                    "country": cur["country"],
                    "date": cur["date"],
                    "feature": f,
                    "absolute_deviation": error[j],
                    "robust_score":
                        error[j] /
                        (scale + 1e-6),
                    "above_p95":
                        bool(error[j] > p95)
                })

    return pd.DataFrame(rows)


cal_2023 = evaluate_deviation_period(
    pd.Timestamp("2023-01-01"),
    pd.Timestamp("2023-12-01"),
    "2023_calibration"
)

test_2024 = evaluate_deviation_period(
    pd.Timestamp("2024-01-01"),
    pd.Timestamp("2024-10-01"),
    "2024_test"
)

all_dev = pd.concat(
    [cal_2023, test_2024],
    ignore_index=True
)

print("\n==========================================")
print("CALIBRATION VS TEST EVENT RATES")
print("==========================================")


def event_table(d):

    # Site-month level:
    sm = (
        d.groupby(
            [
                "site_id",
                "crop",
                "country",
                "date"
            ]
        )
        .agg(
            high_feature_count=(
                "above_p95",
                "sum"
            ),
            mean_score=(
                "robust_score",
                "mean"
            ),
            max_score=(
                "robust_score",
                "max"
            )
        )
        .reset_index()
    )

    sm["multi_sensor_event"] = (
        sm["high_feature_count"] >= 2
    )

    # Consecutive multi-sensor event
    sm = sm.sort_values(
        ["site_id","date"]
    )

    previous_event = (
        sm.groupby("site_id")
        ["multi_sensor_event"]
        .shift(1)
        .fillna(False)
    )

    previous_date = (
        sm.groupby("site_id")
        ["date"]
        .shift(1)
    )

    sm["persistent_event"] = (
        sm["multi_sensor_event"]
        &
        previous_event
        &
        (
            sm["date"]
            ==
            previous_date
            + pd.DateOffset(months=1)
        )
    )

    return sm


ev2023 = event_table(cal_2023)
ev2024 = event_table(test_2024)


def print_event_stats(
    ev,
    name
):

    total = len(ev)

    multi = int(
        ev["multi_sensor_event"].sum()
    )

    persistent = int(
        ev["persistent_event"].sum()
    )

    sites = ev[
        ev["multi_sensor_event"]
    ]["site_id"].nunique()

    print(
        f"\n{name}"
    )

    print(
        "Site-months:",
        total
    )

    print(
        "Multi-sensor events:",
        multi
    )

    print(
        "Multi-sensor event rate:",
        round(
            multi / total,
            4
        )
        if total else np.nan
    )

    print(
        "Sites affected:",
        sites
    )

    print(
        "Persistent consecutive months:",
        persistent
    )


print_event_stats(
    ev2023,
    "2023 OUT-OF-SAMPLE CALIBRATION PERIOD"
)

print_event_stats(
    ev2024,
    "2024 FINAL TEST PERIOD"
)


# Feature-level exceedance rates
print(
    "\nFeature-level 95th-error exceedance rates:"
)

feature_rates = (
    all_dev
    .groupby(
        ["period","feature"]
    )["above_p95"]
    .mean()
    .reset_index()
)

print(
    feature_rates.to_string(
        index=False
    )
)


# ============================================================
# SAVE EVERYTHING
# ============================================================

calibration.to_csv(
    "REVIEWER_2023_OUT_OF_SAMPLE_ERRORS.csv",
    index=False
)

calibration_summary.to_csv(
    "REVIEWER_2023_CALIBRATION_SCALES.csv",
    index=False
)

all_dev.to_csv(
    "REVIEWER_OUT_OF_SAMPLE_CALIBRATED_DEVIATIONS.csv",
    index=False
)

ev2023.to_csv(
    "REVIEWER_2023_EVENT_RESULTS.csv",
    index=False
)

ev2024.to_csv(
    "REVIEWER_2024_EVENT_RESULTS.csv",
    index=False
)

# Final compact comparison
comparison = pd.DataFrame([
    {
        "period": "2023_calibration",
        "site_months": len(ev2023),
        "multi_sensor_events":
            int(ev2023["multi_sensor_event"].sum()),
        "multi_sensor_event_rate":
            ev2023["multi_sensor_event"].mean(),
        "persistent_months":
            int(ev2023["persistent_event"].sum())
    },
    {
        "period": "2024_test",
        "site_months": len(ev2024),
        "multi_sensor_events":
            int(ev2024["multi_sensor_event"].sum()),
        "multi_sensor_event_rate":
            ev2024["multi_sensor_event"].mean(),
        "persistent_months":
            int(ev2024["persistent_event"].sum())
    }
])

comparison.to_csv(
    "REVIEWER_EVENT_RATE_COMPARISON.csv",
    index=False
)

print(
    "\n=========================================="
)

print(
    "FILES SAVED"
)

print(
    "=========================================="
)

print(
    "REVIEWER_TEMPORAL_BASELINE_SUMMARY.csv"
)

print(
    "REVIEWER_SPATIAL_BASELINE_SUMMARY.csv"
)

print(
    "REVIEWER_2023_CALIBRATION_SCALES.csv"
)

print(
    "REVIEWER_2023_EVENT_RESULTS.csv"
)

print(
    "REVIEWER_2024_EVENT_RESULTS.csv"
)

print(
    "REVIEWER_EVENT_RATE_COMPARISON.csv"
)

print(
    "\nExperiment complete."
)
