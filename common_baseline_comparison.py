import pandas as pd
import numpy as np

x = pd.read_csv(
    "REVIEWER_TEMPORAL_BASELINES.csv"
)

print("\n==========================================")
print("COMMON-SUBSET TEMPORAL BASELINE COMPARISON")
print("==========================================")

# Keep only site-month-feature observations where
# all three prediction methods exist.
common = x.dropna(
    subset=[
        "climatology_error",
        "persistence_error",
        "latent_error"
    ]
).copy()

print(
    "\nCommon feature-month rows:",
    len(common)
)

print(
    "Common site-months:",
    common[
        ["site_id", "date"]
    ].drop_duplicates().shape[0]
)

print(
    "Common sites:",
    common["site_id"].nunique()
)

def rmse(x):
    return np.sqrt(
        np.mean(
            np.asarray(x) ** 2
        )
    )

rows = []

for feature, g in common.groupby("feature"):

    rows.append({
        "feature": feature,
        "climatology_RMSE":
            rmse(g["climatology_error"]),
        "persistence_RMSE":
            rmse(g["persistence_error"]),
        "latent_RMSE":
            rmse(g["latent_error"]),
        "N": len(g)
    })

summary = pd.DataFrame(rows)

print(
    "\nFeature-level results:"
)

print(
    summary.to_string(
        index=False
    )
)

means = summary[
    [
        "climatology_RMSE",
        "persistence_RMSE",
        "latent_RMSE"
    ]
].mean()

print(
    "\nMean RMSE on identical observations:"
)

print(means)

clim = means["climatology_RMSE"]
pers = means["persistence_RMSE"]
latent = means["latent_RMSE"]

print(
    "\nLatent vs climatology improvement:",
    round(
        100 * (clim - latent) / clim,
        2
    ),
    "%"
)

print(
    "Latent vs persistence improvement:",
    round(
        100 * (pers - latent) / pers,
        2
    ),
    "%"
)

# Site-level summary
site_rows = []

for site, g in common.groupby("site_id"):

    site_rows.append({
        "site_id": site,
        "climatology_RMSE":
            rmse(g["climatology_error"]),
        "persistence_RMSE":
            rmse(g["persistence_error"]),
        "latent_RMSE":
            rmse(g["latent_error"]),
        "N": len(g)
    })

site_summary = pd.DataFrame(
    site_rows
).sort_values(
    "latent_RMSE"
)

print(
    "\nSite-level results:"
)

print(
    site_summary.to_string(
        index=False
    )
)

summary.to_csv(
    "FINAL_COMMON_TEMPORAL_BASELINE_SUMMARY.csv",
    index=False
)

site_summary.to_csv(
    "FINAL_COMMON_TEMPORAL_BASELINE_SITE_RESULTS.csv",
    index=False
)

print(
    "\nSaved:"
)

print(
    "FINAL_COMMON_TEMPORAL_BASELINE_SUMMARY.csv"
)

print(
    "FINAL_COMMON_TEMPORAL_BASELINE_SITE_RESULTS.csv"
)
