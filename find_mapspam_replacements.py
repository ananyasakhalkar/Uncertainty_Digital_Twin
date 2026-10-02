import os
import numpy as np
import pandas as pd
import rasterio
from rasterio.transform import rowcol

BASE = os.path.expanduser("~/Downloads/Crop_Digital_Twin")
RDIR = os.path.join(BASE, "data/global/mapspam")

targets = {
    "Wheat_US": {
        "crop": "Wheat",
        "file": "spam2020_V2r2_global_H_WHEA_A.tif",
        "lon": -93.7,
        "lat": 41.9,
    },
    "Cotton_Australia": {
        "crop": "Cotton",
        "file": "spam2020_V2r2_global_H_COTT_A.tif",
        "lon": 138.6,
        "lat": -34.6,
    },
}

for name, t in targets.items():

    path = os.path.join(RDIR, t["file"])

    with rasterio.open(path) as src:

        row, col = src.index(t["lon"], t["lat"])

        # Search roughly 100 km around the original candidate.
        radius = 100

        x0 = max(0, col - radius)
        x1 = min(src.width, col + radius + 1)
        y0 = max(0, row - radius)
        y1 = min(src.height, row + radius + 1)

        data = src.read(
            1,
            window=((y0, y1), (x0, x1)),
            masked=True
        ).astype(float)

        data[data <= 0] = np.nan

        if np.all(np.isnan(data)):
            print(f"\n{name}: NO POSITIVE MAPSPAM CELLS FOUND")
            continue

        flat = np.nanargmax(data)
        rr, cc = np.unravel_index(flat, data.shape)

        global_row = y0 + rr
        global_col = x0 + cc

        lon, lat = src.xy(global_row, global_col)

        value = data[rr, cc]

        print("\n" + "=" * 70)
        print(name)
        print("=" * 70)
        print(f"latitude: {lat}")
        print(f"longitude: {lon}")
        print(f"harvested_area: {value}")
