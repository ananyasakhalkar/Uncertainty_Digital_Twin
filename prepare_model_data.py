import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import StandardScaler

INPUT = Path("data/master/features_daily.csv")
OUTPUT = Path("data/model/model_daily.csv")

df = pd.read_csv(INPUT, parse_dates=["date"])

# --------------------------------------------------
# Weather preprocessing
# --------------------------------------------------

# Temperature already converted to Celsius.
df["temperature_C"] = df["temperature_C"].astype(float)

# ERA5-Land precipitation is in metres.
# Convert to millimetres.
df["precipitation_mm"] = df["precipitation"] * 1000.0

# ERA5-Land accumulated solar radiation is J/m2.
# Convert to MJ/m2.
df["solar_radiation_MJ"] = (
    df["solar_radiation"] / 1_000_000.0
)

# --------------------------------------------------
# Observation variables
# --------------------------------------------------

observation_cols = [
    "NDVI",
    "EVI",
    "FPAR",
    "NDWI",
    "SVI",
    "VV",
    "VH",
    "VV_VH",
]

forcing_cols = [
    "temperature_C",
    "precipitation_mm",
    "solar_radiation_MJ",
]

# --------------------------------------------------
# Standardization parameters
# --------------------------------------------------

scalers = {}

for col in observation_cols + forcing_cols:
    values = df[[col]].copy()

    scaler = StandardScaler()

    # Fit only using available values
    valid = values[col].notna()

    scaler.fit(values.loc[valid, [col]])

    transformed = np.full(len(df), np.nan)

    transformed[valid] = scaler.transform(
        values.loc[valid, [col]]
    ).ravel()

    df[col + "_z"] = transformed

    scalers[col] = scaler

# --------------------------------------------------
# Save model dataset
# --------------------------------------------------

OUTPUT.parent.mkdir(exist_ok=True)
df.to_csv(OUTPUT, index=False)

print("Model dataset created.")
print("Shape:", df.shape)

print("\nWeather ranges after unit conversion:")
print(
    "Temperature C:",
    df["temperature_C"].min(),
    "to",
    df["temperature_C"].max()
)

print(
    "Precipitation mm:",
    df["precipitation_mm"].min(),
    "to",
    df["precipitation_mm"].max()
)

print(
    "Solar radiation MJ/m2:",
    df["solar_radiation_MJ"].min(),
    "to",
    df["solar_radiation_MJ"].max()
)

print("\nSaved:", OUTPUT)
