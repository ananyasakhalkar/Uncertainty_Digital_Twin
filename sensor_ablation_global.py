import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from pathlib import Path


# =========================================================
# PATHS
# =========================================================

BASE = Path.home() / "Downloads" / "Crop_Digital_Twin"

DATA = BASE / "data" / "global"

DATA_FILE = DATA / "US_global_panel_2019_2024.csv"

OUT_FILE = DATA / "US_sensor_ablation_results.csv"

SUMMARY_FILE = DATA / "US_sensor_ablation_summary.csv"


# =========================================================
# DEVICE
# =========================================================

DEVICE = torch.device(
    "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)

print("DEVICE:", DEVICE)


# =========================================================
# FEATURE GROUPS
# =========================================================

S2_FEATURES = [
    "NDVI",
    "EVI",
    "NDWI"
]

S1_FEATURES = [
    "VV",
    "VH",
    "VV_VH"
]

ALL_FEATURES = S2_FEATURES + S1_FEATURES

WEATHER_FEATURES = [
    "temperature_2m",
    "precipitation",
    "solar_radiation"
]


# =========================================================
# MODEL
# =========================================================

class CropTwin(nn.Module):

    def __init__(
        self,
        state_dim,
        weather_dim=3,
        crop_dim=2,
        hidden_dim=32
    ):

        super().__init__()

        self.crop_embedding = nn.Embedding(
            crop_dim,
            4
        )

        self.transition = nn.Sequential(

            nn.Linear(
                state_dim +
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
                state_dim
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

        delta = self.transition(x)

        delta = (
            0.10 *
            torch.tanh(delta)
        )

        return state + delta


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


# =========================================================
# NUMERIC CONVERSION
# =========================================================

for c in ALL_FEATURES + WEATHER_FEATURES:

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
# BUILD CONSECUTIVE TRAINING TRANSITIONS
# =========================================================

def build_transitions(
    data,
    state_features
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


            # Require genuine consecutive months
            if (
                nxt["date"]
                != expected_next
            ):
                continue


            # Current state + weather
            if (
                current[
                    state_features +
                    WEATHER_FEATURES
                ].isna().any()
            ):
                continue


            # Next state
            if (
                nxt[
                    state_features
                ].isna().any()
            ):
                continue


            rows.append({

                "site_id":
                    site_id,

                "date":
                    current["date"],

                "state":
                    current[
                        state_features
                    ].values.astype(float),

                "weather":
                    current[
                        WEATHER_FEATURES
                    ].values.astype(float),

                "target":
                    nxt[
                        state_features
                    ].values.astype(float),

                "crop":
                    current["crop"]

            })


    return rows


# =========================================================
# TRAIN + EVALUATE ONE ABLATION
# =========================================================

def run_experiment(
    name,
    state_features
):

    print("\n")
    print("========================================")
    print("EXPERIMENT:", name)
    print("FEATURES:", state_features)
    print("========================================")


    # -----------------------------------------------------
    # TRAIN / TEST
    # -----------------------------------------------------

    train_df = df[
        df["date"]
        <= pd.Timestamp("2023-12-01")
    ].copy()


    test_df = df[
        df["date"]
        >= pd.Timestamp("2024-01-01")
    ].copy()


    # -----------------------------------------------------
    # STANDARDIZATION
    # -----------------------------------------------------

    state_mean = (
        train_df[
            state_features
        ]
        .mean()
    )

    state_std = (
        train_df[
            state_features
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


    for c in state_features:

        train_scaled[c] = (
            train_scaled[c]
            -
            state_mean[c]
        ) / state_std[c]

        test_scaled[c] = (
            test_scaled[c]
            -
            state_mean[c]
        ) / state_std[c]


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
        state_features
    )


    print(
        "Training transitions:",
        len(transitions)
    )


    # -----------------------------------------------------
    # ARRAYS
    # -----------------------------------------------------

    X_state = np.stack(
        [
            x["state"]
            for x in transitions
        ]
    )

    X_weather = np.stack(
        [
            x["weather"]
            for x in transitions
        ]
    )

    Y_target = np.stack(
        [
            x["target"]
            for x in transitions
        ]
    )

    X_crop = np.array(
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

    X_state_t = torch.tensor(
        X_state,
        dtype=torch.float32,
        device=DEVICE
    )

    X_weather_t = torch.tensor(
        X_weather,
        dtype=torch.float32,
        device=DEVICE
    )

    Y_target_t = torch.tensor(
        Y_target,
        dtype=torch.float32,
        device=DEVICE
    )

    X_crop_t = torch.tensor(
        X_crop,
        dtype=torch.long,
        device=DEVICE
    )


    # -----------------------------------------------------
    # MODEL
    # -----------------------------------------------------

    model = CropTwin(
        state_dim=len(state_features),
        weather_dim=len(
            WEATHER_FEATURES
        ),
        crop_dim=len(crop_to_id),
        hidden_dim=32
    ).to(DEVICE)


    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=1e-3
    )

    loss_fn = nn.MSELoss()


    # -----------------------------------------------------
    # TRAIN
    # -----------------------------------------------------

    epochs = 500

    batch_size = 32

    n = len(X_state_t)


    for epoch in range(
        epochs
    ):

        model.train()

        permutation = torch.randperm(
            n,
            device=DEVICE
        )

        for start in range(
            0,
            n,
            batch_size
        ):

            idx = permutation[
                start:start + batch_size
            ]


            state_batch = (
                X_state_t[idx]
            )

            weather_batch = (
                X_weather_t[idx]
            )

            crop_batch = (
                X_crop_t[idx]
            )

            target_batch = (
                Y_target_t[idx]
            )


            optimizer.zero_grad()


            prediction = model(
                state_batch,
                weather_batch,
                crop_batch
            )


            loss = loss_fn(
                prediction,
                target_batch
            )


            loss.backward()

            optimizer.step()


    print(
        "Final training loss:",
        loss.item()
    )


    # -----------------------------------------------------
    # FREE-RUNNING 2024
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


        # Need an initial complete state
        valid_initial = group.dropna(
            subset=state_features
        )


        if len(valid_initial) == 0:
            continue


        initial = valid_initial.iloc[0]


        current_state = torch.tensor(
            initial[
                state_features
            ].values.astype(float),
            dtype=torch.float32,
            device=DEVICE
        ).unsqueeze(0)


        crop_name = initial["crop"]

        crop_id = crop_to_id[
            crop_name
        ]


        crop_tensor = torch.tensor(
            [crop_id],
            dtype=torch.long,
            device=DEVICE
        )


        previous_date = (
            initial["date"]
        )


        for i in range(
            len(group)
        ):

            row = group.iloc[i]

            current_date = (
                row["date"]
            )


            # Skip the initial observation
            if current_date <= previous_date:
                continue


            expected_date = (
                previous_date
                +
                pd.DateOffset(
                    months=1
                )
            )


            if (
                current_date
                != expected_date
            ):

                previous_date = (
                    current_date
                )

                continue


            weather_values = (
                row[
                    WEATHER_FEATURES
                ]
                .values
                .astype(float)
            )


            if not np.all(
                np.isfinite(
                    weather_values
                )
            ):

                previous_date = (
                    current_date
                )

                continue


            weather_tensor = (
                torch.tensor(
                    weather_values,
                    dtype=torch.float32,
                    device=DEVICE
                )
                .unsqueeze(0)
            )


            with torch.no_grad():

                predicted = model(
                    current_state,
                    weather_tensor,
                    crop_tensor
                )


            prediction = (
                predicted
                .cpu()
                .numpy()[0]
            )


            observation = (
                row[
                    state_features
                ]
                .values
                .astype(float)
            )


            for j, feature in enumerate(
                state_features
            ):

                if np.isfinite(
                    observation[j]
                ):

                    results.append({

                        "experiment":
                            name,

                        "site_id":
                            site_id,

                        "crop":
                            crop_name,

                        "date":
                            current_date,

                        "feature":
                            feature,

                        "prediction":
                            prediction[j],

                        "observation":
                            observation[j]

                    })


            current_state = predicted

            previous_date = (
                current_date
            )


    return results


# =========================================================
# RUN THREE EXPERIMENTS
# =========================================================

all_results = []


experiments = [

    (
        "S2_only",
        S2_FEATURES
    ),

    (
        "S1_only",
        S1_FEATURES
    ),

    (
        "S1_plus_S2",
        ALL_FEATURES
    )

]


for name, features in experiments:

    experiment_results = run_experiment(
        name,
        features
    )

    all_results.extend(
        experiment_results
    )


# =========================================================
# SAVE RAW RESULTS
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
print("SENSOR ABLATION RESULTS")
print("========================================")


summary_rows = []


for experiment in [
    "S2_only",
    "S1_only",
    "S1_plus_S2"
]:

    exp_df = results[
        results["experiment"]
        == experiment
    ]


    print("\n", experiment)


    for feature in (
        exp_df["feature"]
        .unique()
    ):

        subset = exp_df[
            exp_df["feature"]
            == feature
        ]


        if len(subset) == 0:
            continue


        y_true = (
            subset[
                "observation"
            ].values
        )

        y_pred = (
            subset[
                "prediction"
            ].values
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

print("========================================")
