import pandas as pd
import numpy as np

Q = np.load("GLOBAL_EMPIRICAL_PROCESS_NOISE.npy")

features = ["NDVI","EVI","NDWI","VV","VH","VV_VH"]

print("\n==========================================")
print("PROCESS-NOISE SENSITIVITY")
print("==========================================")

for scale in [0.25, 0.50, 1.00]:

    variance = Q * scale
    std = np.sqrt(variance)

    print(f"\nScale: {scale:.2f}")
    for f, s, v in zip(features, std, variance):
        print(
            f"{f:7s} "
            f"std={s:.4f} "
            f"variance={v:.4f}"
        )
