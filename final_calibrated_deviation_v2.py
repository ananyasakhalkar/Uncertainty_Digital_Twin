import pandas as pd
import numpy as np

PRED = "GLOBAL_LATENT_TRANSITION_V1_2024_RESULTS.csv"
CAL  = "REVIEWER_2023_CALIBRATION_SCALES.csv"

pred = pd.read_csv(PRED)
cal = pd.read_csv(CAL)

print("\n==========================================")
print("FINAL 2023-CALIBRATED 2024 DEVIATION")
print("==========================================")

print("Prediction columns:")
print(pred.columns.tolist())

print("\nRows:", len(pred))

# ---------------------------------------------------------
# The frozen evaluator stores:
#   error = prediction - observation
#   squared_error = error^2
# Therefore absolute deviation is simply |error|.
# ---------------------------------------------------------

required = [
    "site_id",
    "crop",
    "country",
    "date",
    "feature",
    "error"
]

missing = [
    x for x in required
    if x not in pred.columns
]

if missing:
    raise SystemExit(
        "Missing columns: " + str(missing)
    )

pred["date"] = pd.to_datetime(pred["date"])

pred["absolute_deviation"] = (
    pred["error"].abs()
)

# ---------------------------------------------------------
# 2023 calibration scales
# ---------------------------------------------------------

scale_map = dict(
    zip(
        cal["feature"],
        cal["robust_scale"]
    )
)

p95_map = dict(
    zip(
        cal["feature"],
        cal["p95"]
    )
)

pred["historical_robust_scale"] = (
    pred["feature"].map(scale_map)
)

pred["historical_p95_error"] = (
    pred["feature"].map(p95_map)
)

# ---------------------------------------------------------
# Robust calibrated deviation
# ---------------------------------------------------------

pred["robust_deviation_score"] = (
    pred["absolute_deviation"] /
    pred["historical_robust_scale"]
)

pred["above_historical_p95"] = (
    pred["absolute_deviation"] >=
    pred["historical_p95_error"]
)

# ---------------------------------------------------------
# Feature summary
# ---------------------------------------------------------

feature_summary = (
    pred
    .groupby("feature")
    .agg(
        N=("absolute_deviation", "count"),
        mean_absolute_deviation=(
            "absolute_deviation",
            "mean"
        ),
        median_robust_score=(
            "robust_deviation_score",
            "median"
        ),
        mean_robust_score=(
            "robust_deviation_score",
            "mean"
        ),
        p95_exceedance_rate=(
            "above_historical_p95",
            "mean"
        )
    )
    .reset_index()
)

print("\n==========================================")
print("FEATURE DEVIATION SUMMARY")
print("==========================================")

print(
    feature_summary.to_string(index=False)
)

# ---------------------------------------------------------
# Site-month aggregation
# ---------------------------------------------------------

monthly = (
    pred
    .groupby(
        [
            "site_id",
            "crop",
            "country",
            "date"
        ]
    )
    .agg(
        n_features=(
            "feature",
            "count"
        ),
        high_feature_count=(
            "above_historical_p95",
            "sum"
        ),
        mean_score=(
            "robust_deviation_score",
            "mean"
        ),
        max_score=(
            "robust_deviation_score",
            "max"
        ),
        mean_deviation=(
            "absolute_deviation",
            "mean"
        )
    )
    .reset_index()
)

# ---------------------------------------------------------
# Multi-feature event
#
# At least two features exceed their historical
# 95th-error threshold.
# ---------------------------------------------------------

monthly["multi_feature_event"] = (
    monthly["high_feature_count"] >= 2
)

# ---------------------------------------------------------
# Require both Sentinel sensors for the stronger
# multi-sensor event definition.
# ---------------------------------------------------------

pred["sensor"] = np.where(
    pred["feature"].isin(
        ["NDVI", "EVI", "NDWI"]
    ),
    "S2",
    "S1"
)

sensor_flags = (
    pred
    .groupby(
        [
            "site_id",
            "date",
            "sensor"
        ]
    )["above_historical_p95"]
    .any()
    .unstack(fill_value=False)
    .reset_index()
)

if "S1" not in sensor_flags.columns:
    sensor_flags["S1"] = False

if "S2" not in sensor_flags.columns:
    sensor_flags["S2"] = False

monthly = monthly.merge(
    sensor_flags[
        [
            "site_id",
            "date",
            "S1",
            "S2"
        ]
    ],
    on=[
        "site_id",
        "date"
    ],
    how="left"
)

monthly["multi_sensor_event"] = (
    monthly["S1"].fillna(False) &
    monthly["S2"].fillna(False) &
    (monthly["high_feature_count"] >= 2)
)

# ---------------------------------------------------------
# Persistence
# ---------------------------------------------------------

monthly = monthly.sort_values(
    [
        "site_id",
        "date"
    ]
)

monthly["previous_event"] = (
    monthly
    .groupby("site_id")[
        "multi_sensor_event"
    ]
    .shift(1)
    .fillna(False)
)

monthly["consecutive_month_event"] = (
    monthly["multi_sensor_event"] &
    monthly["previous_event"]
)

# ---------------------------------------------------------
# Event tables
# ---------------------------------------------------------

events = monthly[
    monthly["multi_sensor_event"]
].copy()

persistent = monthly[
    monthly["consecutive_month_event"]
].copy()

print("\n==========================================")
print("FINAL 2024 EVENT RESULTS")
print("==========================================")

print(
    "Site-months:",
    len(monthly)
)

print(
    "Multi-sensor events:",
    len(events)
)

print(
    "Multi-sensor event rate:",
    round(
        len(events) / len(monthly),
        4
    )
)

print(
    "Sites affected:",
    events["site_id"].nunique()
)

print(
    "Persistent consecutive months:",
    len(persistent)
)

print("\nMulti-sensor events:")

if len(events):
    print(
        events[
            [
                "site_id",
                "crop",
                "country",
                "date",
                "high_feature_count",
                "mean_score",
                "max_score",
                "S1",
                "S2",
                "consecutive_month_event"
            ]
        ]
        .sort_values(
            "max_score",
            ascending=False
        )
        .to_string(index=False)
    )
else:
    print("None")

# ---------------------------------------------------------
# Site summary
# ---------------------------------------------------------

site_summary = (
    monthly
    .groupby(
        [
            "site_id",
            "crop",
            "country"
        ]
    )
    .agg(
        site_months=(
            "date",
            "count"
        ),
        multi_sensor_months=(
            "multi_sensor_event",
            "sum"
        ),
        persistent_months=(
            "consecutive_month_event",
            "sum"
        ),
        maximum_score=(
            "max_score",
            "max"
        ),
        mean_score=(
            "mean_score",
            "mean"
        )
    )
    .reset_index()
    .sort_values(
        [
            "multi_sensor_months",
            "maximum_score"
        ],
        ascending=False
    )
)

print("\n==========================================")
print("SITE SUMMARY")
print("==========================================")

print(
    site_summary.to_string(index=False)
)

# ---------------------------------------------------------
# Save
# ---------------------------------------------------------

pred.to_csv(
    "FINAL_2024_CALIBRATED_DEVIATIONS_V2.csv",
    index=False
)

monthly.to_csv(
    "FINAL_2024_SITE_MONTH_DEVIATIONS_V2.csv",
    index=False
)

events.to_csv(
    "FINAL_2024_MULTISENSOR_EVENTS_V2.csv",
    index=False
)

persistent.to_csv(
    "FINAL_2024_PERSISTENT_EVENTS_V2.csv",
    index=False
)

site_summary.to_csv(
    "FINAL_2024_DEVIATION_SITE_SUMMARY_V2.csv",
    index=False
)

feature_summary.to_csv(
    "FINAL_2024_DEVIATION_FEATURE_SUMMARY_V2.csv",
    index=False
)

print("\nSaved final V2 calibrated deviation files.")
