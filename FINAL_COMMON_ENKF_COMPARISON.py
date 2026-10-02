import pandas as pd
import numpy as np

BASE = "REVIEWER_TEMPORAL_BASELINES.csv"
ENKF = "GLOBAL_LATENT_V1_ASSIMILATION_2024_RESULTS.csv"

base = pd.read_csv(BASE)
enkf = pd.read_csv(ENKF)

for df in [base, enkf]:
    df["date"] = pd.to_datetime(df["date"])

# Exact common evaluation keys from the reviewer temporal set.
keys = ["site_id", "date", "feature"]

common = (
    base[keys]
    .drop_duplicates()
    .merge(
        enkf[keys].drop_duplicates(),
        on=keys,
        how="inner"
    )
)

print("\n============================================================")
print("FINAL COMMON 2024 ENKF COMPARISON")
print("============================================================")

print("Common feature rows:", len(common))
print("Common site-months:",
      common[keys[:2]].drop_duplicates().shape[0])
print("Sites:",
      common["site_id"].nunique())

# ------------------------------------------------------------
# Restrict every mode to exactly the same keys.
# ------------------------------------------------------------

modes = [
    "free_running",
    "S1_only",
    "S2_only",
    "S1_plus_S2"
]

rows = []

for mode in modes:

    x = enkf[
        enkf["mode"].eq(mode)
    ].merge(
        common,
        on=keys,
        how="inner"
    )

    # One observation per exact key is required.
    # If duplicates exist, retain the first identical evaluation
    # record rather than silently weighting duplicated keys.
    before = len(x)

    x = (
        x.sort_values(keys)
         .drop_duplicates(keys, keep="first")
    )

    after = len(x)

    rmse = np.sqrt(
        np.mean(
            x["squared_error"].values
        )
    )

    rows.append({
        "mode": mode,
        "RMSE": rmse,
        "N": len(x),
        "duplicates_removed": before - after
    })

    print(
        f"{mode:15s} "
        f"RMSE={rmse:.6f} "
        f"N={len(x)} "
        f"duplicates_removed={before-after}"
    )

summary = pd.DataFrame(rows)

print("\n------------------------------------------------------------")
print("COMMON-SET OVERALL RESULTS")
print("------------------------------------------------------------")

print(
    summary.to_string(
        index=False,
        float_format=lambda x: f"{x:.6f}"
    )
)

# ------------------------------------------------------------
# Feature-level common-set results
# ------------------------------------------------------------

feature_rows = []

for mode in modes:

    x = enkf[
        enkf["mode"].eq(mode)
    ].merge(
        common,
        on=keys,
        how="inner"
    )

    x = (
        x.sort_values(keys)
         .drop_duplicates(keys, keep="first")
    )

    g = (
        x.groupby("feature")
         .agg(
             RMSE=(
                 "squared_error",
                 lambda z: np.sqrt(np.mean(z))
             ),
             N=("squared_error", "size")
         )
         .reset_index()
    )

    g["mode"] = mode
    feature_rows.append(g)

feature_summary = pd.concat(
    feature_rows,
    ignore_index=True
)

print("\n------------------------------------------------------------")
print("COMMON-SET FEATURE RESULTS")
print("------------------------------------------------------------")

print(
    feature_summary[
        ["mode", "feature", "RMSE", "N"]
    ].to_string(
        index=False,
        float_format=lambda x: f"{x:.6f}"
    )
)

# ------------------------------------------------------------
# Gains relative to free-running
# ------------------------------------------------------------

free_rmse = float(
    summary.loc[
        summary["mode"].eq("free_running"),
        "RMSE"
    ].iloc[0]
)

print("\n------------------------------------------------------------")
print("GAIN VS FREE-RUNNING")
print("------------------------------------------------------------")

for mode in [
    "S1_only",
    "S2_only",
    "S1_plus_S2"
]:
    rmse = float(
        summary.loc[
            summary["mode"].eq(mode),
            "RMSE"
        ].iloc[0]
    )

    gain = (
        (free_rmse - rmse)
        / free_rmse
        * 100
    )

    print(
        f"{mode}: {gain:.3f}%"
    )

summary.to_csv(
    "FINAL_COMMON_2024_ENKF_RESULTS.csv",
    index=False
)

feature_summary.to_csv(
    "FINAL_COMMON_2024_ENKF_FEATURE_RESULTS.csv",
    index=False
)

print("\nSaved:")
print("FINAL_COMMON_2024_ENKF_RESULTS.csv")
print("FINAL_COMMON_2024_ENKF_FEATURE_RESULTS.csv")
