import pandas as pd
import numpy as np
import torch
from pathlib import Path

# ============================================================
# FINAL TEST: DOES THE TWIN ADD ANYTHING BEYOND SEASONALITY?
# NO RETRAINING
# ============================================================

DATA = "GLOBAL_ALL25_MODEL_READY.csv"
LATENT_RESULTS = "GLOBAL_LATENT_TRANSITION_V1_2024_RESULTS.csv"
BASELINES = "REVIEWER_TEMPORAL_BASELINES.csv"
MODEL = "models/GLOBAL_LATENT_TRANSITION_V1.pt"

N_BOOT = 10000
SEED = 42

STATE = ["NDVI", "EVI", "NDWI", "VV", "VH", "VV_VH"]
WEATHER = ["temperature_C", "precipitation_mm", "solar_MJ_m2"]

rng = np.random.default_rng(SEED)

print("\n============================================================")
print("FINAL TEST: DOES THE TWIN ADD ANYTHING BEYOND SEASONALITY?")
print("NO RETRAINING")
print("============================================================")

data = pd.read_csv(DATA)
latent = pd.read_csv(LATENT_RESULTS)
base = pd.read_csv(BASELINES)

for df in [data, latent, base]:
    df["date"] = pd.to_datetime(df["date"])

# ------------------------------------------------------------
# FIXED 194 SITE-MONTH EVALUATION SET
# ------------------------------------------------------------

keys = ["site_id", "date", "feature"]

common = (
    base[keys]
    .drop_duplicates()
    .merge(
        latent[keys].drop_duplicates(),
        on=keys,
        how="inner"
    )
)

# Require exactly the six state variables.
common = (
    common.groupby(["site_id", "date"])
    .filter(lambda x: set(x["feature"]) == set(STATE))
)

site_months = (
    common[["site_id", "date"]]
    .drop_duplicates()
    .sort_values(["site_id", "date"])
)

print("\nFixed evaluation site-months:", len(site_months))
print("Sites:", site_months["site_id"].nunique())

# ------------------------------------------------------------
# OBSERVATIONS + 2019-2022 MONTHLY CLIMATOLOGY
# ------------------------------------------------------------

train = data[
    data["date"].dt.year.between(2019, 2022)
].copy()

train["month"] = train["date"].dt.month

clim = (
    train.groupby(
        ["site_id", "month"]
    )[STATE + WEATHER]
    .mean()
    .reset_index()
)

# Attach observations
obs = data.copy()
obs["month"] = obs["date"].dt.month

eval_df = (
    site_months
    .merge(
        obs[
            ["site_id", "date", "month", "crop",
             "country"] + STATE + WEATHER
        ],
        on=["site_id", "date"],
        how="left"
    )
)

eval_df = eval_df.merge(
    clim[
        ["site_id", "month"] + STATE + WEATHER
    ].rename(
        columns={
            c: f"{c}_clim"
            for c in STATE + WEATHER
        }
    ),
    on=["site_id", "month"],
    how="left"
)

# ------------------------------------------------------------
# LATENT MODEL ERROR
# ------------------------------------------------------------

lat = latent.copy()

lat["sq"] = lat["error"] ** 2

latent_pivot = (
    lat.pivot_table(
        index=["site_id", "date"],
        columns="feature",
        values="error"
    )
    .reset_index()
)

# ------------------------------------------------------------
# 1. ANOMALY PERSISTENCE
#
# prediction =
# current month climatology
# + previous month's observed anomaly
#
# ------------------------------------------------------------

anom_rows = []

for site, g in obs.groupby("site_id"):

    g = g.sort_values("date").copy()
    g["month"] = g["date"].dt.month

    cg = clim[
        clim["site_id"].eq(site)
    ].set_index("month")

    g = g.set_index("date")

    for date, row in g.iterrows():

        if date.year != 2024:
            continue

        prev_date = date - pd.DateOffset(months=1)

        if prev_date not in g.index:
            continue

        prev = g.loc[prev_date]
        month = date.month
        prev_month = prev_date.month

        if month not in cg.index or prev_month not in cg.index:
            continue

        for f in STATE:

            if pd.isna(row[f]) or pd.isna(prev[f]):
                continue

            if pd.isna(cg.loc[month, f]) or pd.isna(
                cg.loc[prev_month, f]
            ):
                continue

            prediction = (
                cg.loc[month, f]
                + (
                    prev[f]
                    - cg.loc[prev_month, f]
                )
            )

            error = prediction - row[f]

            anom_rows.append({
                "site_id": site,
                "date": date,
                "feature": f,
                "error": error,
                "squared_error": error ** 2
            })

anom = pd.DataFrame(anom_rows)

# Restrict anomaly persistence to exact common latent keys
anom = anom.merge(
    common,
    on=keys,
    how="inner"
)

# ------------------------------------------------------------
# PAIRED SITE BOOTSTRAP
# latent vs anomaly persistence
# ------------------------------------------------------------

lat_err = latent[
    keys + ["error"]
].rename(
    columns={"error": "latent_error"}
)

anom_err = anom[
    keys + ["error"]
].rename(
    columns={"error": "anomaly_persistence_error"}
)

paired = lat_err.merge(
    anom_err,
    on=keys,
    how="inner"
)

paired["latent_sq"] = paired["latent_error"] ** 2
paired["anom_sq"] = (
    paired["anomaly_persistence_error"] ** 2
)

site_diff = []

for site, g in paired.groupby("site_id"):

    latent_rmse = np.sqrt(
        g["latent_sq"].mean()
    )

    anomaly_rmse = np.sqrt(
        g["anom_sq"].mean()
    )

    site_diff.append({
        "site_id": site,
        "latent_rmse": latent_rmse,
        "anomaly_persistence_rmse": anomaly_rmse,
        "difference": (
            latent_rmse - anomaly_rmse
        )
    })

site_diff = pd.DataFrame(site_diff)

boot = []

sites = site_diff["site_id"].values

for _ in range(N_BOOT):

    sampled = rng.choice(
        sites,
        size=len(sites),
        replace=True
    )

    b = site_diff.set_index(
        "site_id"
    ).loc[sampled]

    boot.append(
        b["difference"].mean()
    )

boot = np.asarray(boot)

point_difference = site_diff["difference"].mean()

print("\n============================================================")
print("1. ANOMALY PERSISTENCE")
print("============================================================")

print(
    "Latent RMSE:",
    np.sqrt(paired["latent_sq"].mean())
)

print(
    "Anomaly-persistence RMSE:",
    np.sqrt(paired["anom_sq"].mean())
)

print(
    "Mean paired site RMSE difference:",
    point_difference
)

print(
    "95% site-bootstrap CI:",
    np.percentile(boot, [2.5, 97.5])
)

print(
    "Sites:",
    len(site_diff)
)

# ------------------------------------------------------------
# TEST-TIME ABLATIONS
#
# Reconstruct frozen latent transition model.
# Weather-history and crop embeddings are taken from checkpoint.
# ------------------------------------------------------------

checkpoint = torch.load(
    MODEL,
    map_location="cpu",
    weights_only=False
)

print("\n============================================================")
print("2. TEST-TIME ABLATIONS")
print("============================================================")

print(
    "Checkpoint keys:",
    list(checkpoint.keys())
)

print(
    "\nIMPORTANT: the ablation section uses the frozen checkpoint "
    "and changes only test-time weather histories / crop IDs."
)

# We use the saved model through the existing evaluation code
# architecture. Import the model class from the frozen training
# implementation if available.

import importlib.util

spec = importlib.util.spec_from_file_location(
    "latent_transition",
    "train_latent_transition_v1.py"
)

mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

TransitionModel = getattr(
    mod,
    "LatentTransition"
)

# ------------------------------------------------------------
# Recover checkpoint architecture parameters
# ------------------------------------------------------------

state_dict = checkpoint.get(
    "model_state_dict",
    checkpoint.get("state_dict")
)

if state_dict is None:
    state_dict = checkpoint.get("transition")

if state_dict is None:
    raise RuntimeError(
        "Could not locate frozen transition model weights."
    )

# Use the existing module constructor.
try:
    model = TransitionModel(
        latent_dim=4,
        weather_dim=9,
        crop_dim=4,
        hidden_dim=32
    )
except Exception:
    model = TransitionModel()

model.load_state_dict(
    state_dict,
    strict=False
)

model.eval()

# ------------------------------------------------------------
# Build frozen normalization from checkpoint
# ------------------------------------------------------------

state_mean = np.asarray(
    checkpoint["state_mean"]
)

state_std = np.asarray(
    checkpoint["state_std"]
)

weather_mean = np.asarray(
    checkpoint["weather_mean"]
)

weather_std = np.asarray(
    checkpoint["weather_std"]
)

crop_to_id = checkpoint[
    "crop_to_id"
]

# ------------------------------------------------------------
# Helper functions
# ------------------------------------------------------------

def norm_state(x):
    return (
        (x - state_mean)
        / np.where(
            state_std == 0,
            1,
            state_std
        )
    )

def norm_weather(x):
    return (
        (x - weather_mean)
        / np.where(
            weather_std == 0,
            1,
            weather_std
        )
    )

def get_weather_history(
    site_df,
    current_date,
    weather_override=None
):
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

        w = row[WEATHER].values[0].astype(float)

        if weather_override is not None:
            w = weather_override(
                site_df,
                d,
                w
            )

        vals.extend(
            norm_weather(w)
        )

    return np.asarray(
        vals,
        dtype=np.float32
    )

def crop_vector(crop):
    vec = np.zeros(4, dtype=np.float32)

    cid = int(crop_to_id[crop])

    if cid >= len(vec):
        raise RuntimeError(
            "Crop embedding dimension mismatch."
        )

    # For ablation we replace the crop ID itself.
    # The model converts this ID internally if required.
    return cid

# ------------------------------------------------------------
# Determine model input interface.
# ------------------------------------------------------------

# The frozen latent transition implementation uses a crop ID.
# We therefore call its forward method directly.

def predict_latent(
    latent_state,
    weather_hist,
    crop_id
):
    x = torch.tensor(
        latent_state,
        dtype=torch.float32
    ).unsqueeze(0)

    w = torch.tensor(
        weather_hist,
        dtype=torch.float32
    ).unsqueeze(0)

    c = torch.tensor(
        [crop_id],
        dtype=torch.long
    )

    with torch.no_grad():
        out = model(
            x,
            w,
            c
        )

    if isinstance(out, tuple):
        out = out[0]

    out = out.detach().cpu().numpy()[0]

    return out

# ------------------------------------------------------------
# Because the saved transition model operates in latent space,
# initialize latent state from the previous observed 6-D state.
# ------------------------------------------------------------

# Load frozen observation encoder.
OBS_MODEL = "models/GLOBAL_LATENT_OBSERVATION_V1.pt"

obs_ckpt = torch.load(
    OBS_MODEL,
    map_location="cpu",
    weights_only=False
)

# Import observation model class.
spec2 = importlib.util.spec_from_file_location(
    "latent_obs",
    "train_latent_observation_v1.py"
)

obsmod = importlib.util.module_from_spec(spec2)
spec2.loader.exec_module(obsmod)

ObsModel = getattr(
    obsmod,
    "LatentObservationModel"
)

try:
    obs_model = ObsModel()
except Exception:
    obs_model = ObsModel(
        state_dim=6,
        latent_dim=4
    )

obs_state = obs_ckpt.get(
    "model_state_dict",
    obs_ckpt.get("state_dict")
)

obs_model.load_state_dict(
    obs_state,
    strict=False
)

obs_model.eval()

# ------------------------------------------------------------
# Map latent prediction back to six observations.
# ------------------------------------------------------------

def latent_to_obs(z):
    zt = torch.tensor(
        z,
        dtype=torch.float32
    ).unsqueeze(0)

    with torch.no_grad():
        out = obs_model(
            zt
        )

    if isinstance(out, tuple):
        out = out[0]

    return out.detach().cpu().numpy()[0]

# ------------------------------------------------------------
# Test-time weather/crop ablation.
# ------------------------------------------------------------

eval_rows = []

all_data = data.copy()

for site, g in all_data.groupby("site_id"):

    g = g.sort_values("date").reset_index(drop=True)

    # Only 2024 evaluation months
    for i in range(len(g)):

        cur = g.iloc[i]

        if cur["date"].year != 2024:
            continue

        date = cur["date"]

        prev_date = date - pd.DateOffset(
            months=1
        )

        prev = g[
            g["date"].eq(prev_date)
        ]

        if len(prev) != 1:
            continue

        prev = prev.iloc[0]

        if prev[STATE].isna().any():
            continue

        if cur[STATE].isna().any():
            continue

        # Need complete 3-month weather history.
        wh = get_weather_history(
            g,
            date
        )

        if wh is None:
            continue

        # Encode previous observed state.
        prev_norm = norm_state(
            prev[STATE].values.astype(float)
        )

        prev_tensor = torch.tensor(
            prev_norm,
            dtype=torch.float32
        ).unsqueeze(0)

        with torch.no_grad():
            z0 = (
                obs_model.encoder(
                    prev_tensor
                )
                .numpy()[0]
            )

        true_crop = cur["crop"]

        # ----------------------------------------------------
        # Weather shuffle:
        # donor selected within same crop and calendar month.
        # ----------------------------------------------------

        donors = all_data[
            (all_data["crop"] == true_crop)
            & (
                all_data["date"].dt.month
                == date.month
            )
            & (
                all_data["date"].dt.year.between(
                    2019, 2023
                )
            )
        ]

        donors = donors.dropna(
            subset=WEATHER
        )

        if len(donors) == 0:
            continue

        donor = donors.iloc[
            rng.integers(len(donors))
        ]

        shuffled_wh = get_weather_history(
            g,
            date
        )

        # Replace the 3 weather months with donor-month
        # values from the same crop/month distribution.
        donor_month = donor["date"].month

        shuffled_parts = []

        for off in [2, 1, 0]:

            d = date - pd.DateOffset(
                months=off
            )

            same = donors[
                donors["date"].dt.month
                == d.month
            ]

            if len(same) == 0:
                same = donors

            dr = same.iloc[
                rng.integers(len(same))
            ]

            shuffled_parts.extend(
                norm_weather(
                    dr[WEATHER]
                    .values
                    .astype(float)
                )
            )

        shuffled_wh = np.asarray(
            shuffled_parts,
            dtype=np.float32
        )

        # Crop swaps: deterministic cyclic swap
        crop_names = list(
            crop_to_id.keys()
        )

        idx = crop_names.index(
            true_crop
        )

        swapped_crop = crop_names[
            (idx + 1) % len(crop_names)
        ]

        conditions = {
            "latent": (
                wh,
                true_crop
            ),
            "weather_shuffle": (
                shuffled_wh,
                true_crop
            ),
            "crop_swap": (
                wh,
                swapped_crop
            ),
            "weather_shuffle_crop_swap": (
                shuffled_wh,
                swapped_crop
            )
        }

        for condition, (
            weather_input,
            crop_input
        ) in conditions.items():

            pred_z = predict_latent(
                z0,
                weather_input,
                crop_to_id[crop_input]
            )

            pred = latent_to_obs(
                pred_z
            )

            truth = norm_state(
                cur[STATE]
                .values
                .astype(float)
            )

            for j, f in enumerate(STATE):

                error = (
                    pred[j] - truth[j]
                )

                eval_rows.append({
                    "site_id": site,
                    "date": date,
                    "crop": true_crop,
                    "feature": f,
                    "condition": condition,
                    "error": error,
                    "squared_error": error ** 2
                })

abl = pd.DataFrame(eval_rows)

# Keep only site-months where all four conditions exist.
complete_keys = (
    abl.groupby(
        ["site_id", "date"]
    )["condition"]
    .nunique()
)

complete_keys = complete_keys[
    complete_keys == 4
].index

abl = (
    abl.set_index(
        ["site_id", "date"]
    )
    .loc[complete_keys]
    .reset_index()
)

print("\n============================================================")
print("TEST-TIME ABLATION RMSE")
print("============================================================")

ab_summary = (
    abl.groupby("condition")
       .agg(
           RMSE=(
               "squared_error",
               lambda x: np.sqrt(np.mean(x))
           ),
           N=("squared_error", "size")
       )
       .reset_index()
)

latent_rmse = float(
    ab_summary.loc[
        ab_summary["condition"] == "latent",
        "RMSE"
    ].iloc[0]
)

ab_summary["RMSE_change_vs_latent"] = (
    ab_summary["RMSE"] - latent_rmse
)

ab_summary["percent_change_vs_latent"] = (
    ab_summary["RMSE_change_vs_latent"]
    / latent_rmse
    * 100
)

print(
    ab_summary.to_string(
        index=False,
        float_format=lambda x: f"{x:.6f}"
    )
)

print("\nAblation site-months:",
      abl[["site_id", "date"]]
      .drop_duplicates()
      .shape[0])

# ------------------------------------------------------------
# 3. WEATHER-ANOMALY STRATA
#
# Absolute temperature anomaly relative to 2019-2022
# site-month climatology.
# ------------------------------------------------------------

strata = eval_df.copy()

strata["temp_anomaly"] = (
    strata["temperature_C"]
    - strata["temperature_C_clim"]
)

strata["abs_temp_anomaly"] = (
    strata["temp_anomaly"].abs()
)

# Restrict to 194 common site-months.
strata = strata[
    ["site_id", "date", "abs_temp_anomaly"]
].drop_duplicates()

# Terciles based on the fixed evaluation population.
q1, q2 = np.quantile(
    strata["abs_temp_anomaly"],
    [1/3, 2/3]
)

strata["temp_stratum"] = pd.cut(
    strata["abs_temp_anomaly"],
    bins=[
        -np.inf,
        q1,
        q2,
        np.inf
    ],
    labels=[
        "low_abs_temp_anomaly",
        "medium_abs_temp_anomaly",
        "high_abs_temp_anomaly"
    ]
)

# Latent errors
lat_strata = latent.merge(
    strata[
        ["site_id", "date", "temp_stratum"]
    ],
    on=["site_id", "date"],
    how="inner"
)

# Climatology errors
for f in STATE:
    pass

clim_long = []

for _, r in eval_df.iterrows():

    for f in STATE:

        if pd.isna(r[f]):
            continue

        pred = r[f"{f}_clim"]

        if pd.isna(pred):
            continue

        clim_long.append({
            "site_id": r["site_id"],
            "date": r["date"],
            "feature": f,
            "clim_error": pred - r[f]
        })

clim_long = pd.DataFrame(clim_long)

clim_long = clim_long.merge(
    strata[
        ["site_id", "date", "temp_stratum"]
    ],
    on=["site_id", "date"],
    how="inner"
)

lat_strata["latent_sq"] = (
    lat_strata["error"] ** 2
)

clim_long["clim_sq"] = (
    clim_long["clim_error"] ** 2
)

latent_s = (
    lat_strata.groupby("temp_stratum")
    ["latent_sq"]
    .mean()
    .pow(0.5)
)

clim_s = (
    clim_long.groupby("temp_stratum")
    ["clim_sq"]
    .mean()
    .pow(0.5)
)

weather_results = pd.DataFrame({
    "latent_RMSE": latent_s,
    "climatology_RMSE": clim_s
}).reset_index()

weather_results["latent_minus_climatology"] = (
    weather_results["latent_RMSE"]
    - weather_results["climatology_RMSE"]
)

weather_results["latent_minus_climatology_percent"] = (
    weather_results["latent_minus_climatology"]
    / weather_results["climatology_RMSE"]
    * 100
)

print("\n============================================================")
print("3. WEATHER-ANOMALY STRATA")
print("============================================================")

print(
    "Temperature-anomaly tercile cutoffs:",
    q1,
    q2
)

print(
    weather_results.to_string(
        index=False,
        float_format=lambda x: f"{x:.6f}"
    )
)

# ------------------------------------------------------------
# SAVE
# ------------------------------------------------------------

site_diff.to_csv(
    "FINAL_ANOMALY_PERSISTENCE_SITE_RESULTS.csv",
    index=False
)

ab_summary.to_csv(
    "FINAL_TEST_TIME_ABLATIONS.csv",
    index=False
)

weather_results.to_csv(
    "FINAL_WEATHER_ANOMALY_STRATA.csv",
    index=False
)

print("\n============================================================")
print("FILES SAVED")
print("============================================================")

print(
    "FINAL_ANOMALY_PERSISTENCE_SITE_RESULTS.csv"
)

print(
    "FINAL_TEST_TIME_ABLATIONS.csv"
)

print(
    "FINAL_WEATHER_ANOMALY_STRATA.csv"
)

print("\nDONE.")
