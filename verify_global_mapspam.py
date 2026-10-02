import os
import numpy as np
import pandas as pd
import rasterio
from rasterio.windows import from_bounds

BASE = os.path.expanduser("~/Downloads/Crop_Digital_Twin")

sites_file = os.path.join(
    BASE, "data/global/sites/global_site_registry.csv"
)

raster_dir = os.path.join(
    BASE, "data/global/mapspam"
)

output_file = os.path.join(
    BASE, "data/global/sites/mapspam_site_verification.csv"
)

raster_map = {
    "Wheat": "spam2020_V2r2_global_H_WHEA_A.tif",
    "Rice": "spam2020_V2r2_global_H_RICE_A.tif",
    "Maize": "spam2020_V2r2_global_H_MAIZ_A.tif",
    "Soybean": "spam2020_V2r2_global_H_SOYB_A.tif",
    "Cotton": "spam2020_V2r2_global_H_COTT_A.tif",
}

sites = pd.read_csv(sites_file)

results = []

for _, site in sites.iterrows():

    crop = site["crop"]
    raster_path = os.path.join(
        raster_dir,
        raster_map[crop]
    )

    with rasterio.open(raster_path) as src:

        x = float(site["longitude"])
        y = float(site["latitude"])

        row, col = src.index(x, y)

        # Small neighborhood around candidate point.
        r0 = max(0, row - 5)
        r1 = min(src.height, row + 6)
        c0 = max(0, col - 5)
        c1 = min(src.width, col + 6)

        data = src.read(
            1,
            window=((r0, r1), (c0, c1)),
            masked=True
        )

        values = data.compressed()

        if len(values) == 0:
            center_value = np.nan
            max_value = np.nan
            mean_positive = np.nan
            positive_fraction = 0.0
        else:
            values = values.astype(float)

            center = src.read(
                1,
                window=((row, row + 1), (col, col + 1))
            )[0, 0]

            center_value = float(center)

            positive = values[values > 0]

            if len(positive) == 0:
                max_value = 0.0
                mean_positive = 0.0
            else:
                max_value = float(np.max(positive))
                mean_positive = float(np.mean(positive))

            positive_fraction = float(
                np.mean(values > 0)
            )

    results.append({
        "site_id": site["site_id"],
        "crop": crop,
        "country": site["country"],
        "region": site.get("region", ""),
        "latitude": float(site["latitude"]),
        "longitude": float(site["longitude"]),
        "mapspam_center_harvested_area": center_value,
        "mapspam_max_harvested_area_neighborhood": max_value,
        "mapspam_mean_positive_area": mean_positive,
        "mapspam_positive_fraction": positive_fraction,
    })

result = pd.DataFrame(results)

result.to_csv(output_file, index=False)

print("\nMAPSPAM VERIFICATION")
print("=" * 80)

print(
    result[
        [
            "site_id",
            "crop",
            "country",
            "mapspam_center_harvested_area",
            "mapspam_max_harvested_area_neighborhood",
            "mapspam_positive_fraction",
        ]
    ].to_string(index=False)
)

print("\nSaved:")
print(output_file)

print("\nPositive-area sites by crop:")
print(
    result.groupby("crop")["mapspam_positive_fraction"]
    .apply(lambda x: (x > 0).sum())
)
