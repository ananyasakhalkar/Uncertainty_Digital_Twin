import os
import pandas as pd

BASE = os.path.expanduser("~/Downloads/Crop_Digital_Twin")

candidate_file = os.path.join(
    BASE, "data/global/sites/global_site_registry.csv"
)

verification_file = os.path.join(
    BASE, "data/global/sites/mapspam_site_verification.csv"
)

out_file = os.path.join(
    BASE, "data/global/sites/final_global_site_registry.csv"
)

df = pd.read_csv(candidate_file)
verification = pd.read_csv(verification_file)

# Remove the two weak original candidates.
remove = {"WHEAT_02", "COTTON_04"}

df = df[~df["site_id"].isin(remove)].copy()

# Replace them with MapSPAM-supported locations.
replacements = pd.DataFrame([
    {
        "site_id": "WHEAT_US_01",
        "crop": "Wheat",
        "country": "United States",
        "region": "North America",
        "latitude": 45.958333365363664,
        "longitude": -101.95833339009089,
    },
    {
        "site_id": "COTTON_AUS_01",
        "crop": "Cotton",
        "country": "Australia",
        "region": "Oceania",
        "latitude": -34.458333333305006,
        "longitude": 144.79166666659285,
    },
])

df = pd.concat([df, replacements], ignore_index=True)

# Add MapSPAM verification information.
verification_cols = [
    "site_id",
    "mapspam_center_harvested_area",
    "mapspam_max_harvested_area_neighborhood",
    "mapspam_mean_positive_area",
    "mapspam_positive_fraction",
]

df = df.merge(
    verification[verification_cols],
    on="site_id",
    how="left"
)

# Add explicit evidence for replacement sites.
replacement_evidence = {
    "WHEAT_US_01": 4115.220703125,
    "COTTON_AUS_01": 2786.89990234375,
}

for site_id, value in replacement_evidence.items():
    df.loc[
        df["site_id"] == site_id,
        "mapspam_max_harvested_area_neighborhood"
    ] = value

    df.loc[
        df["site_id"] == site_id,
        "mapspam_positive_fraction"
    ] = 1.0

df["crop_source"] = "MapSPAM_2020_V2r2"
df["verification_status"] = "crop_specific_spatial_evidence"

df = df.sort_values(
    ["crop", "country"]
).reset_index(drop=True)

df.to_csv(out_file, index=False)

print("\nFINAL GLOBAL SITE REGISTRY")
print("=" * 90)

print(
    df[
        [
            "site_id",
            "crop",
            "country",
            "latitude",
            "longitude",
            "mapspam_max_harvested_area_neighborhood"
        ]
    ].to_string(index=False)
)

print("\nCounts:")
print(df.groupby("crop")["site_id"].count())

print("\nCountries per crop:")
print(
    df.groupby("crop")["country"]
      .nunique()
)

print("\nTotal sites:", len(df))

print("\nSaved:")
print(out_file)
