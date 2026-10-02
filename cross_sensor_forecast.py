import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from pathlib import Path


# =========================================================
# REPRODUCIBILITY
# =========================================================

SEED = 42

np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


# =========================================================
# PATHS
# =========================================================

BASE = Path.home() / "Downloads" / "Crop_Digital_Twin"

DATA = BASE / "data" / "global"

DATA_FILE = DATA / "US_global_panel_2019_2024.csv"

OUT_FILE = DATA / "US_cross_sensor_forecast_clean_results.csv"

SUMMARY_FILE = DATA / "US_cross_sensor_forecast_clean_summary.csv"


# =========================================================
# DEVICE
# =========================================================

DEVICE = torch.device(
    "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)

print("DEVICE:", DEVICE)
print("SEED:", SEED)


# =========================================================
# FEATURES
# =========================================================

S1_FEATURES = [
    "VV",
    "VH",
    "VV_VH"
]

S2_FEATURES = [
    "NDVI",
    "EVI",
    "NDWI"
]

WEATHER_FEATURES = [
    "temperature_2m",
    "precipitation",
    "solar_radiation"
]


# =========================================================
# MODEL
# =========================================================

class CrossSensorTwin(nn.Module):

    def __init__(
        self,
        input_dim,
        output_dim,
        weather_dim=3,
        crop_dim=2,
        hidden_dim=32
    ):

        super().__init__()

        self.crop_embedding = nn.Embedding(
            crop_dim,
            4
        )

        self.network = nn.Sequential(

            nn.Linear(
                input_dim +
                weather_dim +
                4,
                hidden_dim
            ),

            nn.Tanh(),

            nn.Linear(
                hidden_dim,
                hidden_dim
            ),

            nn.Tanh(),

            nn.Linear(
                hidden_dim,
                output_dim
            )
        )


    def forward(
        self,
        state,
        weather,
        crop
    ):

        crop_emb = self.crop_embedding(
            crop
        )

        x = torch.cat(
            [
                state,
                weather,
                crop_emb
            ],
            dim=1
        )

        return self.network(x)


# =========================================================
# LOAD DATA
# =========================================================

df = pd.read_csv(
    DATA_FILE
)

df["date"] = pd.to_datetime(
    df["date"],
    errors="coerce"
)


for c in (
    S1_FEATURES +
    S2_FEATURES +
    WEATHER_FEATURES
):

    df[c] = pd.to_numeric(
        df[c],
        errors="coerce"
    )


df = df.sort_values(
    [
        "site_id",
        "date"
    ]
).reset_index(drop=True)


# =========================================================
# CROP ENCODING
# =========================================================

crop_names = sorted(
    df["crop"]
    .dropna()
    .unique()
)

crop_to_id = {
    crop: i
    for i, crop in enumerate(
        crop_names
    )
}


print(
    "\nCROP ENCODING:",
    crop_to_id
)


# =========================================================
# EXPERIMENTS
# =========================================================

experiments = [

    {
        "name": "S1_to_S2",
        "input": S1_FEATURES,
        "output": S2_FEATURES
    },

    {
        "name": "S2_to_S1",
        "input": S2_FEATURES,
        "output": S1_FEATURES
    },

    {
        "name": "S1S2_to_S2",
        "input": S1_FEATURES + S2_FEATURES,
        "output": S2_FEATURES
    },

    {
        "name": "S1S2_to_S1",
        "input": S1_FEATURES + S2_FEATURES,
        "output": S1_FEATURES
    }

]


# =========================================================
# BUILD CONSECUTIVE TRANSITIONS
# =========================================================

def build_transitions(
    data,
    input_features,
    output_features
):

    rows = []


    for site_id, group in (
        data.groupby("site_id")
    ):

        group = (
            group
            .sort_values("date")
            .reset_index(drop=True)
        )


        for i in range(
            len(group) - 1
        ):

            current = group.iloc[i]

            nxt = group.iloc[i + 1]


            expected_next = (
                current["date"]
                +
                pd.DateOffset(
                    months=1
                )
            )


            if (
                nxt["date"]
                != expected_next
            ):

                continue


            if (
                current[
                    input_features +
                    WEATHER_FEATURES
                ].isna().any()
            ):

                continue


            if (
                nxt[
                    output_features
                ].isna().any()
            ):

                continue


            rows.append({

                "site_id":
                    site_id,

                "date":
                    current["date"],

                "input":
                    current[
                        input_features
                    ].values.astype(float),

                "weather":
                    current[
                        WEATHER_FEATURES
                    ].values.astype(float),

                "target":
                    nxt[
                        output_features
                    ].values.astype(float),

                "crop":
                    current["crop"]

            })


    return rows


# =========================================================
# RUN EXPERIMENT
# =========================================================

def run_experiment(
    experiment
):

    name = experiment["name"]

    input_features = experiment["input"]

    output_features = experiment["output"]


    print("\n")
    print("========================================")
    print("EXPERIMENT:", name)
    print(
        "INPUT:",
        input_features
    )
    print(
        "TARGET:",
        output_features
    )
    print("========================================")


    # -----------------------------------------------------
    # TEMPORAL SPLIT
    # -----------------------------------------------------

    train_df = df[
        df["date"]
        <= pd.Timestamp(
            "2023-12-01"
        )
    ].copy()


    test_df = df[
        df["date"]
        >= pd.Timestamp(
            "2024-01-01"
        )
    ].copy()


    # -----------------------------------------------------
    # TRAIN-ONLY STANDARDIZATION
    # -----------------------------------------------------

    input_mean = (
        train_df[
            input_features
        ]
        .mean()
    )

    input_std = (
        train_df[
            input_features
        ]
        .std()
        .replace(0, 1)
        .fillna(1)
    )


    output_mean = (
        train_df[
            output_features
        ]
        .mean()
    )

    output_std = (
        train_df[
            output_features
        ]
        .std()
        .replace(0, 1)
        .fillna(1)
    )


    weather_mean = (
        train_df[
            WEATHER_FEATURES
        ]
        .mean()
    )

    weather_std = (
        train_df[
            WEATHER_FEATURES
        ]
        .std()
        .replace(0, 1)
        .fillna(1)
    )


    train_scaled = train_df.copy()

    test_scaled = test_df.copy()


    for c in input_features:

        train_scaled[c] = (
            train_scaled[c]
            -
            input_mean[c]
        ) / input_std[c]

        test_scaled[c] = (
            test_scaled[c]
            -
            input_mean[c]
        ) / input_std[c]


    for c in output_features:

        train_scaled[c] = (
            train_scaled[c]
            -
            output_mean[c]
        ) / output_std[c]

        test_scaled[c] = (
            test_scaled[c]
            -
            output_mean[c]
        ) / output_std[c]


    for c in WEATHER_FEATURES:

        train_scaled[c] = (
            train_scaled[c]
            -
            weather_mean[c]
        ) / weather_std[c]

        test_scaled[c] = (
            test_scaled[c]
            -
            weather_mean[c]
        ) / weather_std[c]


    # -----------------------------------------------------
    # TRAINING TRANSITIONS
    # -----------------------------------------------------

    transitions = build_transitions(
        train_scaled,
        input_features,
        output_features
    )


    print(
        "Training transitions:",
        len(transitions)
    )


    # -----------------------------------------------------
    # ARRAYS
    # -----------------------------------------------------

    X_np = np.stack(
        [
            x["input"]
            for x in transitions
        ]
    )

    W_np = np.stack(
        [
            x["weather"]
            for x in transitions
        ]
    )

    Y_np = np.stack(
        [
            x["target"]
            for x in transitions
        ]
    )

    C_np = np.array(
        [
            crop_to_id[
                x["crop"]
            ]
            for x in transitions
        ],
        dtype=np.int64
    )


    # -----------------------------------------------------
    # TORCH
    # -----------------------------------------------------

    X = torch.tensor(
        X_np,
        dtype=torch.float32,
        device=DEVICE
    )

    W = torch.tensor(
        W_np,
        dtype=torch.float32,
        device=DEVICE
    )

    Y = torch.tensor(
        Y_np,
        dtype=torch.float32,
        device=DEVICE
    )

    C = torch.tensor(
        C_np,
        dtype=torch.long,
        device=DEVICE
    )


    # -----------------------------------------------------
    # MODEL
    # -----------------------------------------------------

    model = CrossSensorTwin(
        input_dim=len(input_features),
        output_dim=len(output_features),
        weather_dim=3,
        crop_dim=len(crop_to_id),
        hidden_dim=32
    ).to(DEVICE)


    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=5e-4
    )

    loss_fn = nn.MSELoss()


    # -----------------------------------------------------
    # TRAIN
    # -----------------------------------------------------

    epochs = 500

    batch_size = 32

    n = len(X)


    for epoch in range(
        epochs
    ):

        model.train()

        permutation = torch.randperm(
            n,
            device=DEVICE
        )

        epoch_loss = 0.0

        batches = 0


        for start in range(
            0,
            n,
            batch_size
        ):

            idx = permutation[
                start:start + batch_size
            ]


            optimizer.zero_grad()


            prediction = model(
                X[idx],
                W[idx],
                C[idx]
            )


            loss = loss_fn(
                prediction,
                Y[idx]
            )


            # Guard against numerical instability.
            if not torch.isfinite(loss):

                print(
                    "WARNING: non-finite loss "
                    "at epoch",
                    epoch
                )

                return []


            loss.backward()


            # Prevent exploding gradients.
            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=1.0
            )


            optimizer.step()


            epoch_loss += loss.item()

            batches += 1


        if epoch in [
            0,
            100,
            200,
            300,
            400,
            499
        ]:

            print(
                f"Epoch {epoch:3d} "
                f"loss = "
                f"{epoch_loss / batches:.6f}"
            )


    # -----------------------------------------------------
    # TEST
    # -----------------------------------------------------

    model.eval()

    results = []


    for site_id, group in (
        test_scaled.groupby(
            "site_id"
        )
    ):

        group = (
            group
            .sort_values("date")
            .reset_index(drop=True)
        )


        for i in range(
            len(group) - 1
        ):

            current = group.iloc[i]

            nxt = group.iloc[i + 1]


            expected_next = (
                current["date"]
                +
                pd.DateOffset(
                    months=1
                )
            )


            if (
                nxt["date"]
                != expected_next
            ):

                continue


            if (
                current[
                    input_features +
                    WEATHER_FEATURES
                ].isna().any()
            ):

                continue


            if (
                nxt[
                    output_features
                ].isna().any()
            ):

                continue


            x = torch.tensor(
                current[
                    input_features
                ].values.astype(float),
                dtype=torch.float32,
                device=DEVICE
            ).unsqueeze(0)


            w = torch.tensor(
                current[
                    WEATHER_FEATURES
                ].values.astype(float),
                dtype=torch.float32,
                device=DEVICE
            ).unsqueeze(0)


            crop_id = crop_to_id[
                current["crop"]
            ]


            c = torch.tensor(
                [crop_id],
                dtype=torch.long,
                device=DEVICE
            )


            with torch.no_grad():

                prediction = model(
                    x,
                    w,
                    c
                )


            prediction = (
                prediction
                .cpu()
                .numpy()[0]
            )


            observation = (
                nxt[
                    output_features
                ]
                .values
                .astype(float)
            )


            for j, feature in enumerate(
                output_features
            ):

                results.append({

                    "experiment":
                        name,

                    "site_id":
                        site_id,

                    "crop":
                        current["crop"],

                    "date":
                        nxt["date"],

                    "feature":
                        feature,

                    "prediction":
                        prediction[j],

                    "observation":
                        observation[j]

                })


    return results


# =========================================================
# RUN ALL EXPERIMENTS
# =========================================================

all_results = []


for experiment in experiments:

    output = run_experiment(
        experiment
    )

    all_results.extend(
        output
    )


# =========================================================
# SAVE
# =========================================================

results = pd.DataFrame(
    all_results
)


results.to_csv(
    OUT_FILE,
    index=False
)


# =========================================================
# SUMMARY
# =========================================================

print("\n")
print("========================================")
print("CLEAN CROSS-SENSOR FORECAST RESULTS")
print("========================================")


summary_rows = []


for experiment in results[
    "experiment"
].unique():

    print("\n", experiment)


    experiment_df = results[
        results["experiment"]
        == experiment
    ]


    for feature in (
        experiment_df["feature"]
        .unique()
    ):

        subset = experiment_df[
            experiment_df["feature"]
            == feature
        ]


        y_true = (
            subset["observation"]
            .values
        )

        y_pred = (
            subset["prediction"]
            .values
        )


        rmse = np.sqrt(
            np.mean(
                (y_true - y_pred) ** 2
            )
        )


        mae = np.mean(
            np.abs(
                y_true - y_pred
            )
        )


        print(
            f"  {feature:8s} "
            f"RMSE = {rmse:.4f} "
            f"MAE = {mae:.4f}"
        )


        summary_rows.append({

            "experiment":
                experiment,

            "feature":
                feature,

            "rmse":
                rmse,

            "mae":
                mae,

            "n":
                len(subset)

        })


summary = pd.DataFrame(
    summary_rows
)


summary.to_csv(
    SUMMARY_FILE,
    index=False
)


# =========================================================
# FINAL
# =========================================================

print("\n========================================")

print(
    "Rows evaluated:",
    len(results)
)

print(
    "Saved:",
    OUT_FILE
)

print(
    "Saved:",
    SUMMARY_FILE
)

