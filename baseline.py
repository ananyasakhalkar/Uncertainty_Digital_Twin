import pandas as pd
import numpy as np

df = pd.read_csv(
    "data/model/model_daily.csv",
    parse_dates=["date"]
)

df = df.sort_values("date").reset_index(drop=True)

# 2019-2023 = reference period
# 2024 = test period

train = df[df["date"].dt.year <= 2023]
test = df[df["date"].dt.year == 2024]

s2_cols = [
    "NDVI_z",
    "EVI_z",
    "FPAR_z",
    "NDWI_z",
    "SVI_z"
]

s1_cols = [
    "VV_z",
    "VH_z",
    "VV_VH_z"
]

# --------------------------------------------------
# Seasonal baseline
# --------------------------------------------------

def seasonal_prediction(train, date, columns):

    doy = date.timetuple().tm_yday

    history = train[
        train["date"].dt.dayofyear
        .between(doy - 7, doy + 7)
    ]

    return history[columns].mean()


s2_errors = []
s1_errors = []

for _, row in test.iterrows():

    # -----------------------------
    # S2
    # -----------------------------

    if row["S2_available"] == 1:

        target = row[s2_cols].values.astype(float)

        prediction = seasonal_prediction(
            train,
            row["date"],
            s2_cols
        ).values

        if (
            np.isfinite(target).all()
            and np.isfinite(prediction).all()
        ):

            s2_errors.append(
                np.mean(
                    (prediction - target) ** 2
                )
            )

    # -----------------------------
    # S1
    # -----------------------------

    if row["S1_available"] == 1:

        target = row[s1_cols].values.astype(float)

        prediction = seasonal_prediction(
            train,
            row["date"],
            s1_cols
        ).values

        if (
            np.isfinite(target).all()
            and np.isfinite(prediction).all()
        ):

            s1_errors.append(
                np.mean(
                    (prediction - target) ** 2
                )
            )


print("2024 seasonal baseline")

print(
    "S2 observations:",
    len(s2_errors)
)

print(
    "S1 observations:",
    len(s1_errors)
)

print(
    "S2 RMSE:",
    np.sqrt(np.mean(s2_errors))
)

print(
    "S1 RMSE:",
    np.sqrt(np.mean(s1_errors))
)
