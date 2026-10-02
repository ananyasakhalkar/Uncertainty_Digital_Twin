import os
import pandas as pd

files = [
    "GLOBAL_ALL25_MODEL_READY.csv",
    "GLOBAL_TRAIN_2019_2023.csv",
    "GLOBAL_TEST_2024.csv",

    "models/GLOBAL_5CROP_TEMPORAL_TWIN.pt",
    "models/GLOBAL_5CROP_WEATHER_V3.pt",
    "models/GLOBAL_LATENT_OBSERVATION_V1.pt",
    "models/GLOBAL_LATENT_TRANSITION_V1.pt",
    "models/GLOBAL_SPATIAL_TWIN.pt",

    "GLOBAL_LATENT_V1_HISTORICAL_CALIBRATION.csv",
    "GLOBAL_LATENT_V1_2024_CALIBRATED_DEVIATIONS.csv",
    "GLOBAL_LATENT_V1_MULTISENSOR_EVENTS.csv",
    "GLOBAL_LATENT_V1_PERSISTENT_EVENTS.csv",
    "GLOBAL_LATENT_V1_DEVIATION_SITE_SUMMARY.csv",
    "FINAL_GLOBAL_TWIN_RESULTS_SUMMARY.csv",

    "WHEAT_AU_04_PERSISTENT_DEVIATION.png",
    "GLOBAL_25SITE_DEVIATION_SUMMARY.png"
]

rows = []

for f in files:

    exists = os.path.exists(f)

    if exists:
        size_mb = os.path.getsize(f) / (1024**2)
    else:
        size_mb = None

    rows.append({
        "file": f,
        "exists": exists,
        "size_MB": size_mb
    })

manifest = pd.DataFrame(rows)

manifest.to_csv(
    "GLOBAL_TWIN_REPRODUCIBILITY_MANIFEST.csv",
    index=False
)

print(
    manifest.to_string(index=False)
)

missing = manifest[
    ~manifest["exists"]
]

print(
    "\nMissing files:",
    len(missing)
)

if len(missing):
    print(
        missing["file"].to_string(index=False)
    )

print(
    "\nSaved GLOBAL_TWIN_REPRODUCIBILITY_MANIFEST.csv"
)
