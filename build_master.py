import pandas as pd
from pathlib import Path

DATA = Path("data/clean")

s1 = pd.read_csv(DATA / "S1_clean.csv", parse_dates=["date"])
s2 = pd.read_csv(DATA / "S2_clean.csv", parse_dates=["date"])
weather = pd.read_csv(DATA / "weather_clean.csv", parse_dates=["date"])

# Weather provides the daily timeline
master = weather.copy()

# Exact-date merge only.
# Satellite values remain missing on days without an acquisition.
master = master.merge(
    s1,
    on="date",
    how="left"
)

master = master.merge(
    s2,
    on="date",
    how="left"
)

master = master.sort_values("date").reset_index(drop=True)

# Observation availability flags
master["S1_available"] = master["VV"].notna().astype(int)
master["S2_available"] = master["NDVI"].notna().astype(int)

# Save
out = Path("data/master")
out.mkdir(exist_ok=True)

master.to_csv(out / "master_daily.csv", index=False)

print("Master dataset created.")
print("Shape:", master.shape)

print("\nS1 actual observations:")
print(master["S1_available"].sum())

print("S2 actual observations:")
print(master["S2_available"].sum())

print("\nDate range:")
print(
    master["date"].min().date(),
    "to",
    master["date"].max().date()
)

print("\nMissing satellite days:")
print("S1 missing:", master["VV"].isna().sum())
print("S2 missing:", master["NDVI"].isna().sum())
