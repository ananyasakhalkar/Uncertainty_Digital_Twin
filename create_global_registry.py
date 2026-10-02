from pathlib import Path
import pandas as pd

PROJECT = Path.home() / "Downloads" / "Crop_Digital_Twin"

sites = [
    # ---------------- WHEAT ----------------
    ["WHEAT_01", "Wheat", "India", "South Asia", 29.0588, 76.0856],
    ["WHEAT_02", "Wheat", "United States", "North America", 42.0308, -93.6319],
    ["WHEAT_03", "Wheat", "France", "Europe", 48.1173, 2.1560],
    ["WHEAT_04", "Wheat", "Australia", "Oceania", -35.0, 138.5],
    ["WHEAT_05", "Wheat", "Argentina", "South America", -35.0, -61.0],

    # ---------------- RICE ----------------
    ["RICE_01", "Rice", "India", "South Asia", 22.9868, 87.8550],
    ["RICE_02", "Rice", "Vietnam", "Southeast Asia", 10.0452, 105.7469],
    ["RICE_03", "Rice", "Indonesia", "Southeast Asia", -7.6145, 110.7122],
    ["RICE_04", "Rice", "Philippines", "Southeast Asia", 15.4828, 120.7120],
    ["RICE_05", "Rice", "Thailand", "Southeast Asia", 15.8700, 100.9925],

    # ---------------- MAIZE ----------------
    ["MAIZE_01", "Maize", "United States", "North America", 41.8780, -93.0977],
    ["MAIZE_02", "Maize", "Brazil", "South America", -15.7939, -47.8828],
    ["MAIZE_03", "Maize", "South Africa", "Africa", -26.2041, 28.0473],
    ["MAIZE_04", "Maize", "Argentina", "South America", -31.4201, -64.1888],
    ["MAIZE_05", "Maize", "Mexico", "North America", 19.4326, -99.1332],

    # ---------------- SOYBEAN ----------------
    ["SOY_01", "Soybean", "United States", "North America", 40.6331, -89.3985],
    ["SOY_02", "Soybean", "Brazil", "South America", -16.6869, -49.2648],
    ["SOY_03", "Soybean", "Argentina", "South America", -32.8895, -60.6510],
    ["SOY_04", "Soybean", "China", "East Asia", 31.2304, 121.4737],
    ["SOY_05", "Soybean", "India", "South Asia", 20.5937, 78.9629],

    # ---------------- COTTON ----------------
    ["COTTON_01", "Cotton", "India", "South Asia", 17.3850, 78.4867],
    ["COTTON_02", "Cotton", "United States", "North America", 33.4484, -112.0740],
    ["COTTON_03", "Cotton", "Brazil", "South America", -16.6869, -49.2648],
    ["COTTON_04", "Cotton", "Australia", "Oceania", -29.0, 134.75],
    ["COTTON_05", "Cotton", "Pakistan", "South Asia", 30.3753, 69.3451],
]

columns = [
    "site_id",
    "crop",
    "country",
    "region",
    "latitude",
    "longitude",
]

df = pd.DataFrame(sites, columns=columns)

output_dir = PROJECT / "data" / "global" / "sites"
output_dir.mkdir(parents=True, exist_ok=True)

output = output_dir / "global_site_registry.csv"

df.to_csv(output, index=False)

print("=" * 60)
print("GLOBAL CROP DIGITAL TWIN")
print("SITE REGISTRY")
print("=" * 60)

print(f"\nSites: {len(df)}")
print(f"Crops: {df['crop'].nunique()}")
print(f"Countries: {df['country'].nunique()}")
print(f"Regions: {df['region'].nunique()}")

print("\nCrop distribution:")
print(df["crop"].value_counts().sort_index())

print("\nCountry distribution:")
print(df["country"].value_counts().sort_index())

print(f"\nSaved:")
print(output)
