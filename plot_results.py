import pandas as pd
import matplotlib.pyplot as plt

# 2024 holdout results
results = pd.DataFrame({
    "Sensor": ["S2", "S2", "S2", "S1", "S1", "S1"],
    "Method": [
        "Seasonal baseline",
        "Free-running twin",
        "Assimilating twin",
        "Seasonal baseline",
        "Free-running twin",
        "Assimilating twin"
    ],
    "RMSE": [
        0.4726600875,
        0.6222874,
        0.49469805,
        0.5258740830,
        0.6032182,
        0.5962394
    ]
})

# Plot
fig, ax = plt.subplots(figsize=(9, 6))

sensors = ["S2", "S1"]
methods = [
    "Seasonal baseline",
    "Free-running twin",
    "Assimilating twin"
]

x = range(len(sensors))
width = 0.24

for i, method in enumerate(methods):
    values = [
        results[
            (results["Sensor"] == sensor) &
            (results["Method"] == method)
        ]["RMSE"].iloc[0]
        for sensor in sensors
    ]

    positions = [p + (i - 1) * width for p in x]

    ax.bar(
        positions,
        values,
        width=width,
        label=method
    )

    for pos, value in zip(positions, values):
        ax.text(
            pos,
            value + 0.012,
            f"{value:.3f}",
            ha="center",
            va="bottom",
            fontsize=9
        )

ax.set_xticks(list(x))
ax.set_xticklabels(["Sentinel-2 (S2)", "Sentinel-1 (S1)"])
ax.set_ylabel("Observation-space RMSE")
ax.set_title("2024 Holdout Performance of the Crop Digital Twin")
ax.legend()
ax.grid(axis="y", alpha=0.25)

plt.tight_layout()

plt.savefig(
    "results_comparison_2024.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()

print("Saved: results_comparison_2024.png")
