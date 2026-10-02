import os
import numpy as np
import pandas as pd

# ============================================================
# CONFIG
# ============================================================

BASE = os.path.expanduser(
    "~/Downloads/Crop_Digital_Twin"
)

INPUT = os.path.join(
    BASE,
    "data/global/US_uncertainty_calibration_2024.csv"
)

OUTPUT = os.path.join(
    BASE,
    "data/global/US_uncertainty_calibrated_2024.csv"
)

SUMMARY = os.path.join(
    BASE,
    "data/global/US_uncertainty_calibrated_summary.csv"
)

CALIBRATION = os.path.join(
    BASE,
    "data/global/US_uncertainty_calibration_factors.csv"
)


# ============================================================
# LOAD ORIGINAL FORECAST RESULTS
# ============================================================

print("Loading uncertainty results...")

df = pd.read_csv(INPUT)

df["date"] = pd.to_datetime(df["date"])

numeric_cols = [
    "prior_mean",
    "prior_std",
    "observation",
    "error",
    "abs_error",
    "squared_error"
]

for col in numeric_cols:
    df[col] = pd.to_numeric(
        df[col],
        errors="coerce"
    )

df = df.dropna(
    subset=numeric_cols
).copy()

print("Rows:", len(df))


# ============================================================
# CALIBRATION DATA
#
# IMPORTANT:
#
# The current file contains 2024 only.
#
# Therefore we cannot legitimately learn a calibration factor
# from 2019-2023 using this file alone.
#
# Instead, estimate a robust inflation factor from the
# relationship between actual forecast error and raw ensemble
# spread in the available independent 2024 evaluation.
#
# This is diagnostic calibration, NOT a final train/test
# calibration claim.
# ============================================================

print("\nCalculating uncertainty inflation factors...")


calibration_rows = []

for feature, g in df.groupby("feature"):

    # Robust scale ratio:
    #
    # median absolute error
    # ---------------------
    # median ensemble spread
    #
    # This avoids letting a few extreme errors dominate.

    median_abs_error = g[
        "abs_error"
    ].median()

    median_std = g[
        "prior_std"
    ].median()

    if median_std > 0:

        factor = (
            median_abs_error /
            median_std
        )

    else:

        factor = np.nan

    # Alternative RMSE/spread diagnostic
    rmse = np.sqrt(
        np.mean(
            g["squared_error"]
        )
    )

    rms_std = np.sqrt(
        np.mean(
            g["prior_std"] ** 2
        )
    )

    if rms_std > 0:

        rmse_factor = (
            rmse /
            rms_std
        )

    else:

        rmse_factor = np.nan

    calibration_rows.append({

        "feature": feature,

        "n": len(g),

        "median_abs_error":
            median_abs_error,

        "median_prior_std":
            median_std,

        "median_inflation_factor":
            factor,

        "RMSE":
            rmse,

        "RMS_prior_std":
            rms_std,

        "RMSE_inflation_factor":
            rmse_factor

    })


calibration = pd.DataFrame(
    calibration_rows
)

calibration.to_csv(
    CALIBRATION,
    index=False
)


# ============================================================
# APPLY CALIBRATION
# ============================================================

factor_map = dict(
    zip(
        calibration["feature"],
        calibration["median_inflation_factor"]
    )
)

df["calibration_factor"] = (
    df["feature"]
    .map(factor_map)
)

df["calibrated_std"] = (
    df["prior_std"] *
    df["calibration_factor"]
)


# ============================================================
# CALIBRATED COVERAGE
# ============================================================

summary_rows = []

for feature, g in df.groupby("feature"):

    raw_1std = (
        g["abs_error"] <=
        g["prior_std"]
    ).mean()

    raw_2std = (
        g["abs_error"] <=
        2 * g["prior_std"]
    ).mean()

    calibrated_1std = (
        g["abs_error"] <=
        g["calibrated_std"]
    ).mean()

    calibrated_2std = (
        g["abs_error"] <=
        2 * g["calibrated_std"]
    ).mean()

    raw_corr = g[
        "prior_std"
    ].corr(
        g["abs_error"],
        method="spearman"
    )

    calibrated_corr = g[
        "calibrated_std"
    ].corr(
        g["abs_error"],
        method="spearman"
    )

    summary_rows.append({

        "feature": feature,

        "n": len(g),

        "calibration_factor":
            g["calibration_factor"].iloc[0],

        "raw_mean_std":
            g["prior_std"].mean(),

        "calibrated_mean_std":
            g["calibrated_std"].mean(),

        "raw_coverage_1std":
            raw_1std,

        "calibrated_coverage_1std":
            calibrated_1std,

        "raw_coverage_2std":
            raw_2std,

        "calibrated_coverage_2std":
            calibrated_2std,

        "raw_spearman":
            raw_corr,

        "calibrated_spearman":
            calibrated_corr

    })


summary = pd.DataFrame(
    summary_rows
)


# ============================================================
# SAVE
# ============================================================

df.to_csv(
    OUTPUT,
    index=False
)

summary.to_csv(
    SUMMARY,
    index=False
)


# ============================================================
# PRINT RESULTS
# ============================================================

pd.set_option(
    "display.max_columns",
    None
)

print("\n========================================")
print("UNCERTAINTY CALIBRATION FACTORS")
print("========================================")

print(
    calibration.round(4)
)


print("\n========================================")
print("RAW VS CALIBRATED UNCERTAINTY")
print("========================================")

print(
    summary.round(4)
)


print("\nSaved:")
print(OUTPUT)

print(SUMMARY)

print(CALIBRATION)
