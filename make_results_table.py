import pandas as pd

results = {
    "Sensor": ["S2", "S2", "S1", "S1"],
    "Method": [
        "Seasonal baseline",
        "Assimilating digital twin",
        "Seasonal baseline",
        "Assimilating digital twin"
    ],
    "RMSE": [
        0.4726600875,
        0.49469805,
        0.5258740830,
        0.5962394
    ]
}

df = pd.DataFrame(results)

baseline = df[df["Method"] == "Seasonal baseline"].set_index("Sensor")["RMSE"]

df["RMSE_change_vs_baseline_%"] = df.apply(
    lambda row: (
        (row["RMSE"] - baseline[row["Sensor"]])
        / baseline[row["Sensor"]]
        * 100
    ),
    axis=1
)

print("\n==========================================")
print("2024 HOLDOUT RESULTS")
print("==========================================\n")
print(df.to_string(index=False))

print("\n==========================================")
print("DIGITAL TWIN VS SEASONAL BASELINE")
print("==========================================\n")

for sensor in ["S2", "S1"]:
    b = baseline[sensor]
    twin = df[
        (df["Sensor"] == sensor) &
        (df["Method"] == "Assimilating digital twin")
    ]["RMSE"].iloc[0]

    change = (b - twin) / b * 100

    print(f"{sensor}:")
    print(f"  Baseline RMSE : {b:.4f}")
    print(f"  Twin RMSE     : {twin:.4f}")
    print(f"  Improvement   : {change:.2f}%")
    print()

df.to_csv("results_table_2024.csv", index=False)

print("Saved: results_table_2024.csv")
