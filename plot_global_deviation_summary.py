import pandas as pd
import matplotlib.pyplot as plt

INPUT = "GLOBAL_LATENT_V1_DEVIATION_SITE_SUMMARY.csv"

df = pd.read_csv(INPUT)

# Keep all 25 sites, ordered by number of multi-sensor events.
df = df.sort_values(
    ["multi_sensor_months", "maximum_score"],
    ascending=[True, False]
).reset_index(drop=True)

fig, ax = plt.subplots(
    figsize=(10, 9)
)

y = range(len(df))

ax.barh(
    y,
    df["multi_sensor_months"],
    alpha=0.75,
    label="Multi-sensor deviation months"
)

# Mark persistent months separately.
ax.scatter(
    df["persistent_months"],
    y,
    marker="o",
    s=45,
    label="Persistent consecutive months"
)

ax.set_yticks(list(y))
ax.set_yticklabels(
    df["site_id"]
)

ax.set_xlabel(
    "Number of deviation-event months"
)

ax.set_ylabel(
    "Site"
)

ax.set_title(
    "Global multi-sensor deviation events across 25 crop sites"
)

ax.legend(
    frameon=False
)

ax.grid(
    axis="x",
    alpha=0.2
)

plt.tight_layout()

plt.savefig(
    "GLOBAL_25SITE_DEVIATION_SUMMARY.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()

print(
    "Saved GLOBAL_25SITE_DEVIATION_SUMMARY.png"
)

print("\nSites with multi-sensor events:")

print(
    df[
        df["multi_sensor_months"] > 0
    ][
        [
            "site_id",
            "crop",
            "country",
            "multi_sensor_months",
            "persistent_months",
            "maximum_score"
        ]
    ]
    .sort_values(
        "multi_sensor_months",
        ascending=False
    )
    .to_string(index=False)
)
