import pandas as pd
import os

BASE = os.path.expanduser(
    "~/Downloads/Crop_Digital_Twin"
)

OUT_DIR = os.path.join(
    BASE,
    "data/global/sites"
)

os.makedirs(OUT_DIR, exist_ok=True)

# ============================================================
# VERIFIED U.S. CROP SITES
#
# Crop identity has already been checked against USDA CDL.
# These are the sites we are currently willing to use.
# ============================================================

sites = [

    {
        "site_id": "MAIZE_US_01",
        "crop": "Maize",
        "country": "United States",
        "region": "North America",
        "latitude": 41.8800,
        "longitude": -93.1000,
        "verification_source": "USDA_CDL_2023"
    },

    {
        "site_id": "MAIZE_US_02",
        "crop": "Maize",
        "country": "United States",
        "region": "North America",
        "latitude": 42.0308,
        "longitude": -93.6319,
        "verification_source": "USDA_CDL_2023"
    },

    {
        "site_id": "MAIZE_US_03",
        "crop": "Maize",
        "country": "United States",
        "region": "North America",
        "latitude": 40.6331,
        "longitude": -89.3985,
        "verification_source": "USDA_CDL_2023"
    },

    {
        "site_id": "COTTON_US_01",
        "crop": "Cotton",
        "country": "United States",
        "region": "North America",
        "latitude": 33.53122435619296,
        "longitude": -102.02176391448941,
        "verification_source": "USDA_CDL_2023"
    },

    {
        "site_id": "COTTON_US_02",
        "crop": "Cotton",
        "country": "United States",
        "region": "North America",
        "latitude": 35.87714173277247,
        "longitude": -77.40965118754804,
        "verification_source": "USDA_CDL_2023"
    }

]

df = pd.DataFrame(sites)

path = os.path.join(
    OUT_DIR,
    "verified_us_crop_sites.csv"
)

df.to_csv(
    path,
    index=False
)

print("\nVERIFIED U.S. SITE REGISTRY")
print("==========================")

print(df.to_string(index=False))

print("\nSites:", len(df))

print(
    "\nCrop counts:"
)

print(
    df["crop"].value_counts()
)

print(
    "\nSaved:",
    path
)
