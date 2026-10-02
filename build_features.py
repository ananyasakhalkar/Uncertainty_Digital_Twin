import pandas as pd
import numpy as np
from pathlib import Path

INPUT = Path("data/master/master_daily.csv")
OUTPUT = Path("data/master/features_daily.csv")

df = pd.read_csv(INPUT, parse_dates=["date"])

# --------------------------------------------------
# Weather forcing
# --------------------------------------------------

# ERA5-Land temperature is in Kelvin.
# Convert to Celsius for the model.
df["temperature_C"] = df["temperature_2m"] - 273.15

# Keep precipitation and solar radiation as supplied.
df["precipitation"] = df["precipitation"].clip(lower=0)
df["solar_radiation"] = df["solar_radiation"].clip(lower=0)

# --------------------------------------------------
# Sentinel-2 vegetation signals
# --------------------------------------------------

# Optical vegetation state
df["optical_state"] = (
    0.50 * df["NDVI"]
    + 0.25 * df["EVI"]
    + 0.15 * df["FPAR"]
    + 0.10 * df["NDWI"]
)

# --------------------------------------------------
# Sentinel-1 structural signal
# --------------------------------------------------

# VV/VH can be unstable around zero, so use the
# provided ratio only when finite.
df["sar_state"] = df["VV_VH"].replace(
    [np.inf, -np.inf], np.nan
)

# --------------------------------------------------
# Observation vector
# --------------------------------------------------

df["optical_available"] = (
    df["NDVI"].notna().astype(int)
)

df["sar_available"] = (
    df["VV"].notna().astype(int)
)

# --------------------------------------------------
# Environmental stress proxy
# --------------------------------------------------

# Rolling 14-day precipitation
df["precip_14d"] = (
    df["precipitation"]
    .rolling(14, min_periods=1)
    .sum()
)

# Rolling 14-day mean temperature
df["temp_14d"] = (
    df["temperature_C"]
    .rolling(14, min_periods=1)
    .mean()
)

# --------------------------------------------------
# Save
# --------------------------------------------------

OUTPUT.parent.mkdir(exist_ok=True)
df.to_csv(OUTPUT, index=False)

print("Feature dataset created.")
print("Shape:", df.shape)

print("\nNew feature columns:")
for c in [
    "temperature_C",
    "optical_state",
    "sar_state",
    "optical_available",
    "sar_available",
    "precip_14d",
    "temp_14d",
]:
    print(c)

print("\nOptical observations:", df["optical_available"].sum())
print("SAR observations:", df["sar_available"].sum())
