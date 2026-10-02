import pandas as pd
from pathlib import Path

BASE = Path.home() / "Downloads" / "Crop_Digital_Twin"
DATA = BASE / "data" / "global"

S1_FILE = DATA / "US_S1_monthly_2019_2024.csv"
S2_FILE = DATA / "US_S2_monthly_2019_2024.csv"
WX_FILE = DATA / "US_weather_monthly_2019_2024.csv"

OUT_FILE = DATA / "US_global_panel_2019_2024.csv"


def clean_columns(df):
    df.columns = [
        str(c).strip()
        .replace(" ", "_")
        .replace("-", "_")
        for c in df.columns
    ]
    return df


# ---------------------------------------------------------
# LOAD
# ---------------------------------------------------------

s1 = clean_columns(pd.read_csv(S1_FILE))
s2 = clean_columns(pd.read_csv(S2_FILE))
wx = clean_columns(pd.read_csv(WX_FILE))

print("\nS1 columns:")
print(s1.columns.tolist())

print("\nS2 columns:")
print(s2.columns.tolist())

print("\nWeather columns:")
print(wx.columns.tolist())


# ---------------------------------------------------------
# STANDARDIZE DATE
# ---------------------------------------------------------

for df in [s1, s2, wx]:
    df["date"] = pd.to_datetime(df["date"])


# ---------------------------------------------------------
# KEEP ONLY NEEDED COLUMNS
# ---------------------------------------------------------

s1 = s1[
    [
        "site_id",
        "date",
        "crop",
        "country",
        "VV",
        "VH",
        "VV_VH"
    ]
]

s2 = s2[
    [
        "site_id",
        "date",
        "crop",
        "country",
        "NDVI",
        "EVI",
        "NDWI"
    ]
]

wx = wx[
    [
        "site_id",
        "date",
        "crop",
        "country",
        "temperature_2m",
        "precipitation",
        "solar_radiation"
    ]
]


# ---------------------------------------------------------
# MERGE
# ---------------------------------------------------------

panel = s1.merge(
    s2,
    on=["site_id", "date"],
    how="outer",
    suffixes=("_s1", "_s2")
)

panel = panel.merge(
    wx,
    on=["site_id", "date"],
    how="outer",
    suffixes=("", "_wx")
)


# ---------------------------------------------------------
# CLEAN DUPLICATED METADATA
# ---------------------------------------------------------

panel["crop"] = (
    panel["crop_s1"]
    .fillna(panel["crop_s2"])
    .fillna(panel["crop"])
)

panel["country"] = (
    panel["country_s1"]
    .fillna(panel["country_s2"])
    .fillna(panel["country"])
)


panel = panel.drop(
    columns=[
        c for c in [
            "crop_s1",
            "crop_s2",
            "country_s1",
            "country_s2"
        ]
        if c in panel.columns
    ]
)


# ---------------------------------------------------------
# SORT
# ---------------------------------------------------------

panel = panel.sort_values(
    ["site_id", "date"]
).reset_index(drop=True)


# ---------------------------------------------------------
# BASIC DATA QUALITY REPORT
# ---------------------------------------------------------

print("\n========================================")
print("GLOBAL PANEL")
print("========================================")

print("Rows:", len(panel))
print("Sites:", panel["site_id"].nunique())

print("\nSites:")
print(panel["site_id"].unique())

print("\nCrop counts:")
print(panel.groupby("crop")["site_id"].nunique())


print("\nDate range:")
print(panel["date"].min(), "to", panel["date"].max())


# ---------------------------------------------------------
# MISSINGNESS
# ---------------------------------------------------------

features = [
    "NDVI",
    "EVI",
    "NDWI",
    "VV",
    "VH",
    "VV_VH",
    "temperature_2m",
    "precipitation",
    "solar_radiation"
]

print("\nMissing values:")
print(
    panel[features]
    .isna()
    .sum()
)


print("\nMissing percentage:")
print(
    (
        panel[features]
        .isna()
        .mean() * 100
    ).round(2)
)


# ---------------------------------------------------------
# SAVE
# ---------------------------------------------------------

panel.to_csv(
    OUT_FILE,
    index=False
)

print("\nSaved:")
print(OUT_FILE)
