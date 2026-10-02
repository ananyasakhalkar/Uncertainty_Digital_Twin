import pandas as pd
import numpy as np

INPUT = "GLOBAL_LATENT_V1_2024_CALIBRATED_DEVIATIONS.csv"

df = pd.read_csv(INPUT)
df["date"] = pd.to_datetime(df["date"])

# A feature is considered unusually large when it exceeds
# its historical 95th-percentile error.
df["high"] = df["above_historical_p95"].astype(bool)

# Count high-deviation features at each site-month.
site_month = (
    df.groupby(
        ["site_id","crop","country","date"]
    )
    .agg(
        high_feature_count=("high","sum"),
        mean_score=("robust_deviation_score","mean"),
        max_score=("robust_deviation_score","max"),
        mean_deviation=("absolute_deviation","mean")
    )
    .reset_index()
)

# Multi-sensor event:
# at least two features exceed historical 95th error.
site_month["multi_feature_event"] = (
    site_month["high_feature_count"] >= 2
)

# Look for consecutive monthly events.
site_month = site_month.sort_values(
    ["site_id","date"]
)

site_month["previous_event"] = (
    site_month
    .groupby("site_id")["multi_feature_event"]
    .shift(1)
    .fillna(False)
)

site_month["consecutive_month_event"] = (
    site_month["multi_feature_event"]
    & site_month["previous_event"]
)

print("\n==========================================")
print("MULTI-SENSOR DEVIATION EVENTS")
print("==========================================")

events = site_month[
    site_month["multi_feature_event"]
].copy()

print(
    "Multi-feature site-months:",
    len(events)
)

print(
    "Sites affected:",
    events["site_id"].nunique()
)

print("\nEvents:")

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
            "mean_deviation",
            "consecutive_month_event"
        ]
    ]
    .sort_values(
        ["max_score"],
        ascending=False
    )
    .head(30)
    .to_string(index=False)
)

# Persistent events
persistent = events[
    events["consecutive_month_event"]
].copy()

print("\n==========================================")
print("PERSISTENT MULTI-SENSOR EVENTS")
print("==========================================")

print(
    "Persistent site-months:",
    len(persistent)
)

print(
    "Sites:",
    persistent["site_id"].nunique()
)

print(
    persistent[
        [
            "site_id",
            "crop",
            "country",
            "date",
            "high_feature_count",
            "mean_score",
            "max_score"
        ]
    ]
    .sort_values(
        ["site_id","date"]
    )
    .to_string(index=False)
)

# Site-level summary
summary = (
    site_month
    .groupby(
        ["site_id","crop","country"]
    )
    .agg(
        multi_sensor_months=(
            "multi_feature_event",
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
)

summary = summary.sort_values(
    [
        "persistent_months",
        "multi_sensor_months",
        "maximum_score"
    ],
    ascending=False
)

print("\n==========================================")
print("SITE-LEVEL DEVIATION SUMMARY")
print("==========================================")

print(
    summary.to_string(index=False)
)

site_month.to_csv(
    "GLOBAL_LATENT_V1_SITE_MONTH_DEVIATIONS.csv",
    index=False
)

events.to_csv(
    "GLOBAL_LATENT_V1_MULTISENSOR_EVENTS.csv",
    index=False
)

persistent.to_csv(
    "GLOBAL_LATENT_V1_PERSISTENT_EVENTS.csv",
    index=False
)

summary.to_csv(
    "GLOBAL_LATENT_V1_DEVIATION_SITE_SUMMARY.csv",
    index=False
)

print("\nSaved event-analysis files.")
