import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("stress_results_2024.csv")
df["date"] = pd.to_datetime(df["date"])

s2 = df[df["optical_deviation"].notna()].copy()

mean_deviation = s2["optical_deviation"].mean()

fig, ax = plt.subplots(figsize=(11, 5.5))

ax.scatter(
    s2["date"],
    s2["optical_deviation"],
    s=45,
    label="S2 normalized observation deviation"
)

ax.axhline(
    mean_deviation,
    linestyle="--",
    linewidth=1.2,
    label=f"2024 mean = {mean_deviation:.2f}"
)

ax.set_xlabel("Date")
ax.set_ylabel("Normalized observation deviation")
ax.set_title(
    "2024 Sentinel-2 Observation-Deviation Signal "
    "from the Assimilating Digital Twin"
)

ax.legend()
ax.grid(axis="y", alpha=0.25)

plt.tight_layout()

plt.savefig(
    "s2_deviation_2024_publication.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()

print("Saved: s2_deviation_2024_publication.png")
