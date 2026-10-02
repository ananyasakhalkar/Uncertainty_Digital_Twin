import pandas as pd
import numpy as np

FILE = "GLOBAL_LATENT_V1_ASSIMILATION_2024_RESULTS.csv"

df = pd.read_csv(FILE)

print("\n============================================================")
print("FINAL ENKF INNOVATION DIAGNOSTICS")
print("NO RETRAINING / NO TUNING")
print("============================================================")

print("Columns:")
print(df.columns.tolist())

print("\nShape:", df.shape)

# ------------------------------------------------------------
# The existing file contains forecast/assimilation errors,
# but innovation itself may not be stored.
#
# First inspect whether innovation columns already exist.
# ------------------------------------------------------------

innovation_candidates = [
    c for c in df.columns
    if any(
        x in c.lower()
        for x in [
            "innovation",
            "innov",
            "prior",
            "forecast",
            "observation"
        ]
    )
]

print("\nPotential diagnostic columns:")
print(innovation_candidates)

# ------------------------------------------------------------
# If innovation was not saved, we cannot reconstruct the
# actual EnKF innovation from final errors alone.
# ------------------------------------------------------------

if not any(
    "innovation" in c.lower()
    or "innov" in c.lower()
    for c in df.columns
):

    print("\n============================================================")
    print("IMPORTANT")
    print("============================================================")

    print(
        "The frozen EnKF result file does not contain the "
        "pre-assimilation forecast or innovation."
    )

    print(
        "\nAvailable columns are:"
    )

    print(
        df.columns.tolist()
    )

    print(
        "\nTherefore an actual innovation/NIS analysis "
        "cannot be reconstructed from this result file."
    )

    print(
        "\nWe need to modify the existing EnKF evaluation script "
        "to save, for each observation:"
    )

    print(
        "1. prior forecast"
    )
    print(
        "2. observation"
    )
    print(
        "3. innovation = observation - prior forecast"
    )
    print(
        "4. innovation covariance / ensemble variance"
    )
    print(
        "5. normalized innovation squared (NIS)"
    )

    print(
        "\nSTOPPING HERE TO AVOID INVENTING INNOVATION STATISTICS."
    )

    raise SystemExit(0)

# ------------------------------------------------------------
# If innovation exists, continue.
# ------------------------------------------------------------

innov_col = next(
    c for c in df.columns
    if "innovation" in c.lower()
    or "innov" in c.lower()
)

df["innovation_value"] = pd.to_numeric(
    df[innov_col],
    errors="coerce"
)

df["date"] = pd.to_datetime(
    df["date"]
)

# ------------------------------------------------------------
# Basic innovation statistics
# ------------------------------------------------------------

summary = (
    df
    .groupby(
        [
            "mode",
            "feature"
        ]
    )
    .agg(
        N=("innovation_value", "count"),
        innovation_mean=(
            "innovation_value",
            "mean"
        ),
        innovation_std=(
            "innovation_value",
            "std"
        ),
        innovation_RMSE=(
            "innovation_value",
            lambda x:
                np.sqrt(
                    np.mean(
                        np.asarray(x) ** 2
                    )
                )
        ),
        innovation_MAE=(
            "innovation_value",
            lambda x:
                np.mean(
                    np.abs(
                        np.asarray(x)
                    )
                )
        )
    )
    .reset_index()
)

print("\n============================================================")
print("INNOVATION STATISTICS")
print("============================================================")

print(
    summary.to_string(
        index=False,
        float_format=lambda x: f"{x:.6f}"
    )
)

summary.to_csv(
    "FINAL_ENKF_INNOVATION_SUMMARY.csv",
    index=False
)

print(
    "\nSaved FINAL_ENKF_INNOVATION_SUMMARY.csv"
)
