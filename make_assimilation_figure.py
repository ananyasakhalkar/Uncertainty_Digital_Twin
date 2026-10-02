import os
import pandas as pd
import matplotlib.pyplot as plt

BASE = os.path.expanduser(
    "~/Downloads/Crop_Digital_Twin"
)

INPUT = os.path.join(
    BASE,
    "data/global/US_assimilation_ablation_summary.csv"
)

OUTPUT = os.path.join(
    BASE,
    "data/global/US_assimilation_ablation_figure.png"
)

df = pd.read_csv(INPUT)

# Mean RMSE across variables for each mode
mean_rmse = (
    df.groupby("mode")["rmse"]
    .mean()
    .reindex([
        "free_running",
        "S1_only",
        "S2_only",
        "S1_plus_S2"
    ])
)

# Variable-level table for plotting
pivot = df.pivot(
    index="feature",
    columns="mode",
    values="rmse"
)

pivot = pivot[
    [
        "free_running",
        "S1_only",
        "S2_only",
        "S1_plus_S2"
    ]
]

fig, ax = plt.subplots(
    figsize=(10, 6)
)

pivot.plot(
    kind="bar",
    ax=ax
)

ax.set_xlabel("Observation variable")
ax.set_ylabel("Next-month RMSE")
ax.set_title(
    "Next-month forecast error under assimilation configurations"
)

ax.legend(
    title="Configuration",
    labels=[
        "Free-running",
        "S1 only",
        "S2 only",
        "S1 + S2"
    ]
)

plt.xticks(
    rotation=0
)

plt.tight_layout()

plt.savefig(
    OUTPUT,
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print("Mean RMSE:")
print(mean_rmse.round(4))

print("\nSaved:")
print(OUTPUT)
