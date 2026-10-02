import pandas as pd

files = ["S1.csv", "S2.csv", "weather.csv"]

for f in files:
    print("\n" + "=" * 50)
    print(f" {f}")
    print("=" * 50)

    df = pd.read_csv("data/" + f)

    print("Shape:", df.shape)
    print("Date range:", df["date"].min(), "to", df["date"].max())
    print("Duplicate dates:", df["date"].duplicated().sum())

    print("\nMissing values:")
    print(df.isna().sum())

    if "crop" in df.columns:
        print("\nCrop values:")
        print(df["crop"].value_counts(dropna=False))

    print("\nColumns:")
    print(df.columns.tolist())
