from pathlib import Path
import pandas as pd

# ============================================================
# GLOBAL CROP DIGITAL TWIN
# Initial multi-site experiment design
# ============================================================

PROJECT = Path.home() / "Downloads" / "Crop_Digital_Twin"

# Main directories
directories = [
    "data/global",
    "data/global/sites",
    "data/global/raw",
    "data/global/clean",
    "data/global/master",
    "data/global/model",
    "models/global",
    "results/global",
    "figures/global",
]

for directory in directories:
    (PROJECT / directory).mkdir(parents=True, exist_ok=True)

# ------------------------------------------------------------
# Initial globally distributed crop sites
# ------------------------------------------------------------
#
# These are candidate sampling locations, NOT observations yet.
# We will populate them with Sentinel-1, Sentinel-2 and weather
# data in the next step.
#
# Five target crops:
# Wheat, Rice, Maize, Soybean, Cotton
#
# Multiple agricultural regions and climate environments are
# represented.

sites = [
    # Wheat
    ["WHEAT_01", "Wheat", "India", "South Asia", 28.6139, 77.2090],
    ["WHEAT_02", "Wheat", "United States", "North America", 41.8780, -93.0977],
    ["WHEAT_03", "Wheat", "France", "Europe", 45.7640, 4.8357],
    ["WHEAT_04", "Wheat", "Australia", "Oceania", -34.9285, 138.6007],

    # Rice
    ["RICE_01", "Rice", "India", "South Asia", 22.5726, 88.3639],
    ["RICE_02", "Rice", "Vietnam", "Southeast Asia", 10.8231, 106.6297],
    ["RICE_03", "Rice", "Indonesia", "Southeast Asia", -6.2088, 106.8456],
    ["RICE_04", "Rice", "Philippines", "Southeast Asia", 14.5995, 120.9842],

    # Maize
    ["MAIZE_01", "Maize", "United States", "North America", 42.3314, -93.6122],
    ["MAIZE_02", "Maize", "Brazil", "South America", -15.7939, -47.8828],
    ["MAIZE_03", "Maize", "South Africa", "Africa", -26.2041, 28.0473],
    ["MAIZE_04", "Maize", "Argentina", "South America", -31.4201, -64.1888],

    # Soybean
    ["SOY_01", "Soybean", "United States", "North America", 40.6331, -89.3985],
    ["SOY_02", "Soybean", "Brazil", "South America", -16.6869, -49.2648],
    ["SOY_03", "Soybean", "Argentina", "South America", -32.8895, -60.6510],
    ["SOY_04", "Soybean", "China", "East Asia", 31.2304, 121.4737],

    # Cotton
    ["COTTON_01", "Cotton", "India", "South Asia", 17.3850, 78.4867],
    ["COTTON_02", "Cotton", "United States", "North America", 33.4484, -112.0740],
    ["COTTON_03", "Cotton", "Brazil", "South America", -16.6869, -49.2648],
    ["COTTON_04", "Cotton", "Australia", "Oceania", -29.0136, 134.7540],
]

columns = [
    "site_id",
    "crop",
    "country",
    "region",
    "latitude",
    "longitude",
]

sites_df = pd.DataFrame(sites, columns=columns)

output = PROJECT / "data/global/sites/global_sites.csv"
sites_df.to_csv(output, index=False)

print("=" * 60)
print("GLOBAL CROP DIGITAL TWIN")
print("=" * 60)
print()
print(f"Project: {PROJECT}")
print(f"Sites: {len(sites_df)}")
print(f"Crops: {sites_df['crop'].nunique()}")
print()

print("Crop distribution:")
print(sites_df["crop"].value_counts().sort_index())

print()
print("Regional distribution:")
print(sites_df["region"].value_counts())

print()
print(f"Saved: {output}")
print()
print("Next stage: retrieve Sentinel-1, Sentinel-2 and weather")
print("for these sites and construct a common multi-site time series.")
