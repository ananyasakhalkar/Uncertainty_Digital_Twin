import os
import glob
import numpy as np
import pandas as pd

SEED = 20261001
N_BOOT = 10000
rng = np.random.default_rng(SEED)

STATE = ["NDVI", "EVI", "NDWI", "VV", "VH", "VV_VH"]
S2 = {"NDVI", "EVI", "NDWI"}
S1 = {"VV", "VH", "VV_VH"}

# ============================================================
# FILES
# ============================================================

BASELINE_FILE = "REVIEWER_TEMPORAL_BASELINES.csv"
LATENT_FILE = "GLOBAL_LATENT_TRANSITION_V1_2024_RESULTS.csv"
ENKF_FILE = "GLOBAL_LATENT_V1_ASSIMILATION_2024_RESULTS.csv"
CAL_FILE = "REVIEWER_2023_CALIBRATION_SCALES.csv"
CAL_ERRORS = "REVIEWER_2023_OUT_OF_SAMPLE_ERRORS.csv"

print("\n============================================================")
print("FINAL REVIEWER STATISTICS")
print("NO RETRAINING")
print("============================================================")

for f in [
    BASELINE_FILE,
    LATENT_FILE,
    ENKF_FILE,
    CAL_FILE,
    CAL_ERRORS
]:
    print(f"{f}: {'FOUND' if os.path.exists(f) else 'MISSING'}")

# ============================================================
# HELPERS
# ============================================================

def read_required(path):
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    return pd.read_csv(path)

def rmse(x):
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    return np.sqrt(np.mean(x ** 2))

def ci_bootstrap_site_difference(
    df,
    error_a,
    error_b,
    n_boot=N_BOOT
):
    """
    Paired bootstrap by site.
    Difference = RMSE(A) - RMSE(B).
    Each bootstrap resamples complete sites.
    """

    sites = np.array(
        sorted(df["site_id"].dropna().unique())
    )

    # Pre-compute site-level sums and counts.
    site_stats = []

    for site in sites:

        g = df[df["site_id"] == site]

        a = g[error_a].to_numpy(float)
        b = g[error_b].to_numpy(float)

        mask = np.isfinite(a) & np.isfinite(b)

        a = a[mask]
        b = b[mask]

        if len(a) == 0:
            continue

        site_stats.append(
            (
                site,
                np.sum(a ** 2),
                np.sum(b ** 2),
                len(a)
            )
        )

    sites = np.array(
        [x[0] for x in site_stats]
    )

    a_ss = np.array(
        [x[1] for x in site_stats]
    )

    b_ss = np.array(
        [x[2] for x in site_stats]
    )

    n = np.array(
        [x[3] for x in site_stats]
    )

    point = (
        np.sqrt(a_ss.sum() / n.sum())
        -
        np.sqrt(b_ss.sum() / n.sum())
    )

    boot = np.empty(n_boot)

    for i in range(n_boot):

        idx = rng.integers(
            0,
            len(sites),
            size=len(sites)
        )

        boot[i] = (
            np.sqrt(
                a_ss[idx].sum()
                /
                n[idx].sum()
            )
            -
            np.sqrt(
                b_ss[idx].sum()
                /
                n[idx].sum()
            )
        )

    lo, hi = np.percentile(
        boot,
        [2.5, 97.5]
    )

    return point, lo, hi, len(sites)

# ============================================================
# 1. COMMON 2024 TEMPORAL EVALUATION SET
# ============================================================

baseline = read_required(BASELINE_FILE)
latent = read_required(LATENT_FILE)

print("\n============================================================")
print("INPUT SCHEMAS")
print("============================================================")

print("\nBaseline columns:")
print(baseline.columns.tolist())

print("\nLatent columns:")
print(latent.columns.tolist())

# ------------------------------------------------------------
# Normalize dates
# ------------------------------------------------------------

baseline["date"] = pd.to_datetime(
    baseline["date"]
)

latent["date"] = pd.to_datetime(
    latent["date"]
)

# ------------------------------------------------------------
# Identify baseline error columns
# ------------------------------------------------------------

def find_col(df, candidates):
    for c in candidates:
        if c in df.columns:
            return c
    return None

clim_col = find_col(
    baseline,
    [
        "climatology_error",
        "climatology_residual"
    ]
)

pers_col = find_col(
    baseline,
    [
        "persistence_error",
        "persistence_residual"
    ]
)

if clim_col is None or pers_col is None:
    raise RuntimeError(
        "Could not identify climatology/persistence error columns."
    )

# Latent frozen evaluator stores prediction-observation error.
latent_error_col = "error"

if latent_error_col not in latent.columns:
    raise RuntimeError(
        "Expected latent 'error' column not found."
    )

# ------------------------------------------------------------
# Rename
# ------------------------------------------------------------

baseline2 = baseline[
    [
        "site_id",
        "date",
        "feature",
        clim_col,
        pers_col
    ]
].copy()

baseline2 = baseline2.rename(
    columns={
        clim_col: "climatology_error",
        pers_col: "persistence_error"
    }
)

latent2 = latent[
    [
        "site_id",
        "date",
        "feature",
        latent_error_col
    ]
].copy()

latent2 = latent2.rename(
    columns={
        latent_error_col: "latent_error"
    }
)

# ------------------------------------------------------------
# Common observations
# ------------------------------------------------------------

common = baseline2.merge(
    latent2,
    on=[
        "site_id",
        "date",
        "feature"
    ],
    how="inner"
)

common = common.dropna(
    subset=[
        "climatology_error",
        "persistence_error",
        "latent_error"
    ]
).copy()

print("\n============================================================")
print("FIXED 2024 TEMPORAL EVALUATION SET")
print("============================================================")

print(
    "Feature rows:",
    len(common)
)

print(
    "Site-months:",
    common[
        ["site_id", "date"]
    ].drop_duplicates().shape[0]
)

print(
    "Sites:",
    common["site_id"].nunique()
)

print(
    "Date range:",
    common["date"].min(),
    "to",
    common["date"].max()
)

# ============================================================
# 2. PAIRED SITE BOOTSTRAP
# ============================================================

print("\n============================================================")
print("PAIRED SITE BOOTSTRAP")
print("10,000 RESAMPLES")
print("95% PERCENTILE INTERVALS")
print("Difference = latent RMSE - baseline RMSE")
print("============================================================")

bootstrap_rows = []

# Overall
for baseline_name, baseline_col in [
    (
        "climatology",
        "climatology_error"
    ),
    (
        "persistence",
        "persistence_error"
    )
]:

    point, lo, hi, nsites = (
        ci_bootstrap_site_difference(
            common,
            "latent_error",
            baseline_col
        )
    )

    bootstrap_rows.append({
        "level": "overall",
        "feature": "ALL",
        "comparison":
            f"latent_minus_{baseline_name}",
        "RMSE_difference": point,
        "CI_low": lo,
        "CI_high": hi,
        "sites": nsites
    })

# Per feature
for feature in STATE:

    g = common[
        common["feature"] == feature
    ].copy()

    for baseline_name, baseline_col in [
        (
            "climatology",
            "climatology_error"
        ),
        (
            "persistence",
            "persistence_error"
        )
    ]:

        point, lo, hi, nsites = (
            ci_bootstrap_site_difference(
                g,
                "latent_error",
                baseline_col
            )
        )

        bootstrap_rows.append({
            "level": "feature",
            "feature": feature,
            "comparison":
                f"latent_minus_{baseline_name}",
            "RMSE_difference": point,
            "CI_low": lo,
            "CI_high": hi,
            "sites": nsites
        })

bootstrap = pd.DataFrame(
    bootstrap_rows
)

print(
    bootstrap.to_string(
        index=False,
        float_format=lambda x: f"{x:.6f}"
    )
)

bootstrap.to_csv(
    "FINAL_PAIRED_SITE_BOOTSTRAP.csv",
    index=False
)

# ============================================================
# 3. FIXED 2024 SET INCLUDING FOUR ENKF MODES
# ============================================================

enkf = read_required(ENKF_FILE)

print("\n============================================================")
print("ENKF INPUT")
print("============================================================")

print(
    "EnKF shape:",
    enkf.shape
)

print(
    "EnKF columns:"
)

print(
    enkf.columns.tolist()
)

print(
    "\nFirst rows:"
)

print(
    enkf.head().to_string(index=False)
)

# ------------------------------------------------------------
# Identify mode / feature / squared-error columns
# ------------------------------------------------------------

mode_col = find_col(
    enkf,
    [
        "mode",
        "assimilation_mode"
    ]
)

feature_col = find_col(
    enkf,
    [
        "feature",
        "variable"
    ]
)

se_col = find_col(
    enkf,
    [
        "squared_error",
        "sq_error"
    ]
)

if mode_col is None or feature_col is None or se_col is None:

    raise RuntimeError(
        "\nCould not identify EnKF mode/feature/squared-error columns.\n"
        "The columns printed above are needed to finish the fixed-set table."
    )

enkf["date"] = pd.to_datetime(
    enkf["date"]
)

# ------------------------------------------------------------
# Restrict to the SAME 2024 site-month-feature set
# used by latent + baselines.
# ------------------------------------------------------------

keys = common[
    [
        "site_id",
        "date",
        "feature"
    ]
].drop_duplicates()

enkf_common = enkf.merge(
    keys,
    on=[
        "site_id",
        "date",
        "feature"
    ],
    how="inner"
)

enkf_common["squared_error"] = (
    enkf_common[se_col].astype(float)
)

# ------------------------------------------------------------
# Add latent free-running as another mode
# ------------------------------------------------------------

latent_mode = common[
    [
        "site_id",
        "date",
        "feature",
        "latent_error"
    ]
].copy()

latent_mode["mode"] = "free_running"
latent_mode["squared_error"] = (
    latent_mode["latent_error"] ** 2
)

latent_mode = latent_mode[
    [
        "site_id",
        "date",
        "feature",
        "mode",
        "squared_error"
    ]
]

enkf_mode = enkf_common[
    [
        "site_id",
        "date",
        "feature",
        mode_col,
        "squared_error"
    ]
].copy()

enkf_mode = enkf_mode.rename(
    columns={
        mode_col: "mode"
    }
)

all_modes = pd.concat(
    [
        latent_mode,
        enkf_mode
    ],
    ignore_index=True
)

print("\n============================================================")
print("FIXED 2024 ENKF EVALUATION SET")
print("============================================================")

print(
    "Site-months:",
    common[
        ["site_id", "date"]
    ].drop_duplicates().shape[0]
)

print(
    "Sites:",
    common["site_id"].nunique()
)

print(
    "\nRows by mode:"
)

print(
    all_modes.groupby("mode").size()
)

# ------------------------------------------------------------
# RMSE by mode and feature
# ------------------------------------------------------------

mode_feature = (
    all_modes
    .groupby(
        [
            "mode",
            "feature"
        ]
    )
    .agg(
        RMSE=(
            "squared_error",
            lambda x:
                np.sqrt(
                    np.mean(
                        np.asarray(x)
                    )
                )
        ),
        N=("squared_error", "count")
    )
    .reset_index()
)

mode_overall = (
    all_modes
    .groupby("mode")
    .agg(
        RMSE=(
            "squared_error",
            lambda x:
                np.sqrt(
                    np.mean(
                        np.asarray(x)
                    )
                )
        ),
        N=("squared_error", "count")
    )
    .reset_index()
)

print("\nOverall:")
print(
    mode_overall.to_string(
        index=False,
        float_format=lambda x: f"{x:.6f}"
    )
)

print("\nFeature-level:")
print(
    mode_feature.to_string(
        index=False,
        float_format=lambda x: f"{x:.6f}"
    )
)

mode_overall.to_csv(
    "FINAL_FIXED_2024_ENKF_MODES.csv",
    index=False
)

mode_feature.to_csv(
    "FINAL_FIXED_2024_ENKF_FEATURES.csv",
    index=False
)

# ============================================================
# 4. 2023 NULL EVENT RATES
# ============================================================

print("\n============================================================")
print("2023 NULL EVENT RATES")
print("USING 2023 OUT-OF-SAMPLE THRESHOLDS")
print("============================================================")

cal = read_required(CAL_FILE)

print("\nCalibration scale columns:")
print(cal.columns.tolist())

# Find 2023 residual file
if os.path.exists(CAL_ERRORS):

    err23 = read_required(CAL_ERRORS)

else:

    # Try to use any likely calibration-error file
    candidates = [
        x for x in glob.glob("*2023*ERROR*.csv")
        if os.path.basename(x) != CAL_FILE
    ]

    if not candidates:
        raise RuntimeError(
            "Could not find 2023 out-of-sample error file."
        )

    print(
        "\nUsing discovered calibration file:",
        candidates[0]
    )

    err23 = pd.read_csv(
        candidates[0]
    )

print("\n2023 error columns:")
print(err23.columns.tolist())

err_col = find_col(
    err23,
    [
        "absolute_error",
        "error",
        "residual",
        "prediction_error"
    ]
)

if err_col is None:
    raise RuntimeError(
        "Could not identify the 2023 residual column."
    )

err23["date"] = pd.to_datetime(
    err23["date"]
)

# This file already stores absolute error.
if err_col == "absolute_error":
    err23["abs_error"] = (
        err23[err_col].astype(float)
    )
else:
    err23["abs_error"] = (
        err23[err_col].astype(float).abs()
    )

# 2023 only
err23 = err23[
    err23["date"].dt.year == 2023
].copy()

# p95 threshold map
p95_col = find_col(
    cal,
    [
        "p95",
        "historical_p95_error"
    ]
)

if p95_col is None:
    raise RuntimeError(
        "Could not identify p95 threshold column."
    )

p95_map = dict(
    zip(
        cal["feature"],
        cal[p95_col]
    )
)

err23["above_p95"] = (
    err23["abs_error"]
    >=
    err23["feature"].map(p95_map)
)

# ------------------------------------------------------------
# Sensor flags
# ------------------------------------------------------------

err23["sensor"] = np.where(
    err23["feature"].isin(S2),
    "S2",
    "S1"
)

# Feature-level threshold flags
wide = (
    err23
    .pivot_table(
        index=[
            "site_id",
            "date"
        ],
        columns="feature",
        values="above_p95",
        aggfunc="max",
        fill_value=False
    )
    .reset_index()
)

for f in STATE:
    if f not in wide.columns:
        wide[f] = False

wide["n_features"] = (
    wide[STATE]
    .sum(axis=1)
)

wide["n_s2"] = (
    wide[
        ["NDVI", "EVI", "NDWI"]
    ]
    .sum(axis=1)
)

wide["n_s1"] = (
    wide[
        ["VV", "VH", "VV_VH"]
    ]
    .sum(axis=1)
)

wide["has_s1"] = (
    wide["n_s1"] >= 1
)

wide["has_s2"] = (
    wide["n_s2"] >= 1
)

# ------------------------------------------------------------
# Event definitions
#
# D1 = >=1 feature
# D2 = >=2 features
# D3 = >=3 features
#
# Each also reported with/without cross-sensor requirement.
# ------------------------------------------------------------

wide["event_1_any"] = (
    wide["n_features"] >= 1
)

wide["event_2_any"] = (
    wide["n_features"] >= 2
)

wide["event_3_any"] = (
    wide["n_features"] >= 3
)

wide["event_1_cross"] = (
    wide["event_1_any"] &
    wide["has_s1"] &
    wide["has_s2"]
)

wide["event_2_cross"] = (
    wide["event_2_any"] &
    wide["has_s1"] &
    wide["has_s2"]
)

wide["event_3_cross"] = (
    wide["event_3_any"] &
    wide["has_s1"] &
    wide["has_s2"]
)

# ------------------------------------------------------------
# Denominator
# ------------------------------------------------------------

n_site_months = len(wide)

definitions = [
    (
        ">=1 feature, any sensor",
        "event_1_any"
    ),
    (
        ">=1 feature, cross-sensor",
        "event_1_cross"
    ),
    (
        ">=2 features, any sensor",
        "event_2_any"
    ),
    (
        ">=2 features, cross-sensor",
        "event_2_cross"
    ),
    (
        ">=3 features, any sensor",
        "event_3_any"
    ),
    (
        ">=3 features, cross-sensor",
        "event_3_cross"
    )
]

event_rows = []

for name, col in definitions:

    count = int(
        wide[col].sum()
    )

    event_rows.append({
        "definition": name,
        "site_months": n_site_months,
        "events": count,
        "event_rate": count / n_site_months
    })

event_summary = pd.DataFrame(
    event_rows
)

print("\n2023 site-months:", n_site_months)

print(
    event_summary.to_string(
        index=False,
        float_format=lambda x: f"{x:.6f}"
    )
)

event_summary.to_csv(
    "FINAL_2023_EVENT_DEFINITIONS.csv",
    index=False
)

# ============================================================
# 5. FINAL SUMMARY
# ============================================================

print("\n============================================================")
print("FILES SAVED")
print("============================================================")

print("FINAL_PAIRED_SITE_BOOTSTRAP.csv")
print("FINAL_FIXED_2024_ENKF_MODES.csv")
print("FINAL_FIXED_2024_ENKF_FEATURES.csv")
print("FINAL_2023_EVENT_DEFINITIONS.csv")

print("\nDONE.")
