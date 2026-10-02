import pandas as pd
import numpy as np

DATA = "GLOBAL_ALL25_MODEL_READY.csv"

df = pd.read_csv(DATA)
df["date"] = pd.to_datetime(df["date"])

STATE = [
    "NDVI",
    "EVI",
    "NDWI",
    "VV",
    "VH",
    "VV_VH"
]

# The exact five spatially held-out sites
TEST_SITES = [
    "COTTON_AUS_01",
    "MAIZE_04",
    "RICE_02",
    "SOY_03",
    "WHEAT_03"
]

test = df[df["site_id"].isin(TEST_SITES)].copy()

print("\n==========================================")
print("SPATIAL HOLDOUT DATA CHECK")
print("==========================================")

print("Rows:", len(test))
print("Sites:", test["site_id"].nunique())
print("Site list:")
print(sorted(test["site_id"].unique()))

print("\nRows per site:")
print(test.groupby("site_id").size())

print("\nDate range:")
print(test["date"].min(), "to", test["date"].max())

# ---------------------------------------------------------
# Build genuine consecutive monthly transitions
# ---------------------------------------------------------

rows = []

for site, g in test.groupby("site_id"):

    g = g.sort_values("date").reset_index(drop=True)

    for i in range(1, len(g)):

        prev = g.iloc[i-1]
        curr = g.iloc[i]

        if curr["date"] != prev["date"] + pd.DateOffset(months=1):
            continue

        # Complete target and previous state
        if (
            prev[STATE].isna().any()
            or curr[STATE].isna().any()
        ):
            continue

        for f in STATE:

            rows.append({
                "site_id": site,
                "date": curr["date"],
                "feature": f,
                "actual": curr[f],
                "previous": prev[f]
            })

common = pd.DataFrame(rows)

print("\n==========================================")
print("COMMON SPATIAL EVALUATION")
print("==========================================")

print("Feature rows:", len(common))
print(
    "Site-months:",
    common[["site_id", "date"]].drop_duplicates().shape[0]
)
print("Sites:", common["site_id"].nunique())

# ---------------------------------------------------------
# Persistence
# ---------------------------------------------------------

common["persistence_error"] = (
    common["previous"] - common["actual"]
)

# ---------------------------------------------------------
# Training-only spatial climatology
#
# Use all NON-held-out sites.
# Calculate crop x month climatology.
# ---------------------------------------------------------

train = df[
    ~df["site_id"].isin(TEST_SITES)
].copy()

train["month_num"] = train["date"].dt.month

clim_tables = {}

for f in STATE:

    tmp = (
        train
        .dropna(subset=[f])
        .assign(month_num=train["date"].dt.month)
        .groupby(["crop", "month_num"])[f]
        .mean()
        .reset_index()
        .rename(columns={f: "climatology"})
    )

    clim_tables[f] = tmp

common["month_num"] = common["date"].dt.month

# Need crop attached to each row
site_crop = (
    test[["site_id", "crop"]]
    .drop_duplicates()
)

common = common.merge(
    site_crop,
    on="site_id",
    how="left"
)

common["climatology"] = np.nan

for f in STATE:

    mask = common["feature"] == f

    lookup = clim_tables[f]

    merged = common.loc[mask, ["crop", "month_num"]].merge(
        lookup,
        on=["crop", "month_num"],
        how="left"
    )

    common.loc[mask, "climatology"] = (
        merged["climatology"].to_numpy()
    )

common["climatology_error"] = (
    common["climatology"] - common["actual"]
)

# ---------------------------------------------------------
# Load existing frozen spatial model predictions
#
# The previous spatial result file should contain the
# feature-level predictions/errors. We do NOT retrain.
# ---------------------------------------------------------

candidate_files = [
    "GLOBAL_SPATIAL_TEST_RESULTS.csv",
    "GLOBAL_SPATIAL_TEST_PREDICTIONS.csv",
    "GLOBAL_SPATIAL_RESULTS.csv"
]

model_file = None

for f in candidate_files:

    try:
        tmp = pd.read_csv(f)

        if "site_id" in tmp.columns:
            model_file = f
            model = tmp
            break

    except FileNotFoundError:
        pass

if model_file is None:

    print("\nERROR: Could not find the frozen spatial prediction file.")
    print("Available CSV files containing SPATIAL:")
    import glob
    print(glob.glob("*SPATIAL*.csv"))
    raise SystemExit(1)

print("\nFrozen spatial model file:")
print(model_file)

print("\nModel columns:")
print(model.columns.tolist())

# ---------------------------------------------------------
# Try to identify prediction/error format
# ---------------------------------------------------------

# If the result file already contains feature-level errors,
# use them directly.

error_col = None

for c in [
    "error",
    "residual",
    "prediction_error",
    "model_error"
]:

    if c in model.columns:
        error_col = c
        break

if error_col is not None:

    model_common = model[
        ["site_id", "date", "feature", error_col]
    ].copy()

    model_common = model_common.rename(
        columns={error_col: "latent_error"}
    )

else:

    # Otherwise look for actual/predicted columns
    pred_col = "prediction"
    actual_col = "observation"

    if pred_col is None or actual_col is None:

        print(
            "\nERROR: Cannot identify prediction/error columns "
            "in the spatial result file."
        )

        print(
            "Please paste the column list above."
        )

        raise SystemExit(1)

    model_common = model[
        ["site_id", "date", "feature",
         pred_col, actual_col]
    ].copy()

    model_common["latent_error"] = (
        model_common[pred_col]
        - model_common[actual_col]
    )

    model_common = model_common[
        ["site_id", "date", "feature", "latent_error"]
    ]

model_common["date"] = pd.to_datetime(
    model_common["date"]
)

# ---------------------------------------------------------
# Merge only exact common observations
# ---------------------------------------------------------

common = common.merge(
    model_common,
    on=["site_id", "date", "feature"],
    how="inner"
)

print("\nExact common model/baseline rows:", len(common))

print(
    "Exact common site-months:",
    common[
        ["site_id", "date"]
    ].drop_duplicates().shape[0]
)

# ---------------------------------------------------------
# RMSE
# ---------------------------------------------------------

def rmse(x):
    x = np.asarray(x, dtype=float)
    return np.sqrt(np.mean(x ** 2))

results = []

for feature, g in common.groupby("feature"):

    results.append({
        "feature": feature,
        "climatology_RMSE":
            rmse(g["climatology_error"]),
        "persistence_RMSE":
            rmse(g["persistence_error"]),
        "spatial_model_RMSE":
            rmse(g["latent_error"]),
        "N": len(g)
    })

summary = pd.DataFrame(results)

print("\n==========================================")
print("FINAL COMMON SPATIAL COMPARISON")
print("==========================================")

print(
    summary.to_string(index=False)
)

means = summary[
    [
        "climatology_RMSE",
        "persistence_RMSE",
        "spatial_model_RMSE"
    ]
].mean()

print("\nMean RMSE:")
print(means)

model_rmse = means["spatial_model_RMSE"]
clim_rmse = means["climatology_RMSE"]
pers_rmse = means["persistence_RMSE"]

print(
    "\nModel vs climatology:",
    round(
        100 * (clim_rmse - model_rmse) / clim_rmse,
        2
    ),
    "%"
)

print(
    "Model vs persistence:",
    round(
        100 * (pers_rmse - model_rmse) / pers_rmse,
        2
    ),
    "%"
)

summary.to_csv(
    "FINAL_COMMON_SPATIAL_BASELINE_SUMMARY.csv",
    index=False
)

print(
    "\nSaved FINAL_COMMON_SPATIAL_BASELINE_SUMMARY.csv"
)
