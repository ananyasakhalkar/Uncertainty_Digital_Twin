import pandas as pd
import matplotlib.pyplot as plt

INPUT = "GLOBAL_LATENT_V1_2024_CALIBRATED_DEVIATIONS.csv"

df = pd.read_csv(INPUT)
df["date"] = pd.to_datetime(df["date"])

site = "WHEAT_AU_04"

x = df[
    df["site_id"] == site
].copy()

features = [
    "EVI",
    "NDVI",
    "NDWI",
    "VH",
    "VV",
    "VV_VH"
]

fig, ax = plt.subplots(
    figsize=(11,6)
)

for feature in features:

    d = x[
        x["feature"] == feature
    ].sort_values("date")

    ax.plot(
        d["date"],
        d["robust_deviation_score"],
        marker="o",
        linewidth=1.5,
        markersize=4,
        label=feature
    )

ax.axhline(
    1.0,
    linestyle="--",
    linewidth=1
)

# Highlight the consecutive May-June event.
ax.axvspan(
    pd.Timestamp("2024-05-01"),
    pd.Timestamp("2024-06-30"),
    alpha=0.15
)

ax.set_title(
    "WHEAT_AU_04: Calibrated Model-Observation Deviations"
)

ax.set_xlabel("Date")

ax.set_ylabel(
    "Deviation / historical robust error scale"
)

ax.legend(
    ncol=3,
    frameon=False
)

ax.grid(
    alpha=0.2
)

plt.tight_layout()

plt.savefig(
    "WHEAT_AU_04_PERSISTENT_DEVIATION.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()

print(
    "Saved WHEAT_AU_04_PERSISTENT_DEVIATION.png"
)
