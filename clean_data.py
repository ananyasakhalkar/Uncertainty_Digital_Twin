import pandas as pd
from pathlib import Path

DATA = Path("data")

# -----------------------------
# Sentinel-1
# -----------------------------
s1 = pd.read_csv(DATA / "S1.csv")
s1["date"] = pd.to_datetime(s1["date"])
s1 = s1.sort_values("date").drop_duplicates("date", keep="first")

s1 = s1[
    ["date", "VV", "VH", "VV_VH"]
].copy()

# -----------------------------
# Sentinel-2
# -----------------------------
s2 = pd.read_csv(DATA / "S2.csv")
s2["date"] = pd.to_datetime(s2["date"])

# Remove duplicate observation date
s2 = s2.sort_values("date").drop_duplicates("date", keep="first")

s2 = s2[
    ["date", "B2", "B3", "B4", "B8",
     "EVI", "FPAR", "NDVI", "NDWI", "SVI"]
].copy()

# Remove the single incomplete observation
s2 = s2.dropna()

# -----------------------------
# Weather
# -----------------------------
weather = pd.read_csv(DATA / "weather.csv")
weather["date"] = pd.to_datetime(weather["date"])

weather = weather.sort_values("date").drop_duplicates("date", keep="first")

weather = weather[
    ["date", "precipitation",
     "solar_radiation", "temperature_2m"]
].copy()

# -----------------------------
# Save cleaned data
# -----------------------------
out = DATA / "clean"
out.mkdir(exist_ok=True)

s1.to_csv(out / "S1_clean.csv", index=False)
s2.to_csv(out / "S2_clean.csv", index=False)
weather.to_csv(out / "weather_clean.csv", index=False)

print("Cleaning complete.")
print("S1:", s1.shape)
print("S2:", s2.shape)
print("Weather:", weather.shape)

print("\nS1 dates:", s1.date.min().date(), "to", s1.date.max().date())
print("S2 dates:", s2.date.min().date(), "to", s2.date.max().date())
print("Weather dates:", weather.date.min().date(), "to", weather.date.max().date())
