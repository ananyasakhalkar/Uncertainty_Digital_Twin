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

OUT_FILE = DATA / "US_spatial_holdout_clean_results.csv"

SUMMARY_FILE = DATA / "US_spatial_holdout_clean_summary.csv"


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
# FEATURES
# =========================================================

state_features = [
    "NDVI",
    "EVI",
    "NDWI",
    "VV",
    "VH",
    "VV_VH"
]

weather_features = [
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
        state_dim=6,
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
                state_dim + weather_dim + 4,
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

        crop_emb = self.crop_embedding(crop)

        x = torch.cat(
            [
                state,
                weather,
                crop_emb
            ],
            dim=1
        )

        delta = self.transition(x)

        delta = 0.10 * torch.tanh(delta)

        return state + delta


# =========================================================
# LOAD DATA
# =========================================================

df = pd.read_csv(DATA_FILE)

df["date"] = pd.to_datetime(
    df["date"],
    errors="coerce"
)


# =========================================================
# NUMERIC CONVERSION
# =========================================================

for c in state_features + weather_features:

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
# SITE / CROP INFORMATION
# =========================================================

sites = sorted(
    df["site_id"]
    .dropna()
    .unique()
)

crop_names = sorted(
    df["crop"]
    .dropna()
    .unique()
)

crop_to_id = {
    crop: i
    for i, crop in enumerate(crop_names)
}


print("\nSITES:")

for site in sites:
    print(" ", site)

print(
    "\nNUMBER OF SITES:",
    len(sites)
)

print(
    "\nCROP ENCODING:",
    crop_to_id
)


# =========================================================
# BUILD CONSECUTIVE TRANSITIONS
# =========================================================
#
# A transition is allowed only when:
#
# current month -> next month
#
# differs by exactly one calendar month.
#
# This prevents missing observations from being treated
# as direct one-month transitions.
# =========================================================

def build_transitions(data):

    transition_rows = []

    for site_id, group in data.groupby("site_id"):

        group = (
            group
            .sort_values("date")
            .reset_index(drop=True)
        )

        for i in range(len(group) - 1):

            current = group.iloc[i]

            nxt = group.iloc[i + 1]

            expected_next = (
                current["date"]
                + pd.DateOffset(months=1)
            )

            # Require genuine consecutive months
            if nxt["date"] != expected_next:
                continue

            # Current state, next state, and current weather
            # must all be available.
            required_current = (
                state_features +
                weather_features
            )

            required_next = state_features

            if (
                current[
                    required_current
                ].isna().any()
            ):
                continue

            if (
                nxt[
                    required_next
                ].isna().any()
            ):
                continue

            transition_rows.append({

                "site_id": site_id,

                "date": current["date"],

                "next_date": nxt["date"],

                "crop": current["crop"],

                "state": current[
                    state_features
                ].values.astype(float),

                "weather": current[
                    weather_features
                ].values.astype(float),

                "target": nxt[
                    state_features
                ].values.astype(float)

            })

    return transition_rows


# =========================================================
# METRIC
# =========================================================

def rmse(y_true, y_pred):

    return np.sqrt(
        np.mean(
            (y_true - y_pred) ** 2
        )
    )


# =========================================================
# LEAVE-ONE-SITE-OUT
# =========================================================

all_results = []


for held_out_site in sites:

    print("\n========================================")
    print(
        "HOLDING OUT:",
        held_out_site
    )
    print("========================================")


    # -----------------------------------------------------
    # SPLIT SITES
    # -----------------------------------------------------

    train_df = df[
        df["site_id"] != held_out_site
    ].copy()

    test_df = df[
        df["site_id"] == held_out_site
    ].copy()


    # -----------------------------------------------------
    # TRAIN-ONLY STANDARDIZATION
    # -----------------------------------------------------

    state_mean = (
        train_df[state_features]
        .mean()
    )

    state_std = (
        train_df[state_features]
        .std()
        .replace(0, 1)
        .fillna(1)
    )

    weather_mean = (
        train_df[weather_features]
        .mean()
    )

    weather_std = (
        train_df[weather_features]
        .std()
        .replace(0, 1)
        .fillna(1)
    )


    # -----------------------------------------------------
    # STANDARDIZE COPIES
    # -----------------------------------------------------

    train_scaled = train_df.copy()

    test_scaled = test_df.copy()


    for c in state_features:

        train_scaled[c] = (
            train_scaled[c]
            - state_mean[c]
        ) / state_std[c]

        test_scaled[c] = (
            test_scaled[c]
            - state_mean[c]
        ) / state_std[c]


    for c in weather_features:

        train_scaled[c] = (
            train_scaled[c]
            - weather_mean[c]
        ) / weather_std[c]

        test_scaled[c] = (
            test_scaled[c]
            - weather_mean[c]
        ) / weather_std[c]


    # -----------------------------------------------------
    # BUILD TRAINING TRANSITIONS
    # -----------------------------------------------------

    transitions = build_transitions(
        train_scaled
    )


    print(
        "Training transitions:",
        len(transitions)
    )


    if len(transitions) == 0:

        print(
            "No valid training transitions."
        )

        continue


    # -----------------------------------------------------
    # CONVERT TRAINING TRANSITIONS TO ARRAYS
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
            crop_to_id[x["crop"]]
            for x in transitions
        ],
        dtype=np.int64
    )


    # -----------------------------------------------------
    # TORCH TENSORS
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
        weather_dim=len(weather_features),
        crop_dim=len(crop_to_id),
        hidden_dim=32
    ).to(DEVICE)


    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=1e-3
    )

    loss_fn = nn.MSELoss()


    # -----------------------------------------------------
    # JOINT TRAINING
    # -----------------------------------------------------

    epochs = 500

    batch_size = 32

    n = len(X_state_t)


    for epoch in range(epochs):

        model.train()

        permutation = torch.randperm(
            n,
            device=DEVICE
        )

        epoch_loss = 0.0

        batch_count = 0


        for start in range(
            0,
            n,
            batch_size
        ):

            idx = permutation[
                start:start + batch_size
            ]


            state_batch = X_state_t[idx]

            weather_batch = (
                X_weather_t[idx]
            )

            crop_batch = X_crop_t[idx]

            target_batch = Y_target_t[idx]


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


            epoch_loss += loss.item()

            batch_count += 1


    print(
        "Final training loss:",
        epoch_loss / max(
            batch_count,
            1
        )
    )


    # -----------------------------------------------------
    # TEST: FREE-RUNNING UNSEEN SITE
    # -----------------------------------------------------

    model.eval()


    test_scaled = (
        test_scaled
        .sort_values("date")
        .reset_index(drop=True)
    )


    # -----------------------------------------------------
    # INITIAL STATE
    # -----------------------------------------------------
    #
    # Use the earliest complete state at the held-out site.
    # This initializes the twin but does not train it.
    # -----------------------------------------------------

    valid_initial = test_scaled.dropna(
        subset=state_features
    )


    if len(valid_initial) == 0:

        print(
            "No complete initial state."
        )

        continue


    initial = valid_initial.iloc[0]


    current_state_np = (
        initial[state_features]
        .values
        .astype(float)
    )


    crop_name = initial["crop"]

    crop_id = crop_to_id[crop_name]


    current_state = torch.tensor(
        current_state_np,
        dtype=torch.float32,
        device=DEVICE
    ).unsqueeze(0)


    crop_tensor = torch.tensor(
        [crop_id],
        dtype=torch.long,
        device=DEVICE
    )


    # -----------------------------------------------------
    # FREE-RUN THROUGH SITE
    # -----------------------------------------------------

    previous_date = initial["date"]


    for i in range(
        test_scaled.index[
            test_scaled["date"]
            >= initial["date"]
        ].min(),
        len(test_scaled)
    ):

        row = test_scaled.iloc[i]

        current_date = row["date"]


        # Only evaluate true consecutive months.
        expected_date = (
            previous_date
            + pd.DateOffset(months=1)
        )


        if current_date != expected_date:

            previous_date = current_date

            continue


        weather_values = (
            row[
                weather_features
            ]
            .values
            .astype(float)
        )


        if not np.all(
            np.isfinite(
                weather_values
            )
        ):

            previous_date = current_date

            continue


        weather_tensor = torch.tensor(
            weather_values,
            dtype=torch.float32,
            device=DEVICE
        ).unsqueeze(0)


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

                all_results.append({

                    "held_out_site":
                        held_out_site,

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


        # Free-running update.
        current_state = predicted

        previous_date = current_date


# =========================================================
# RESULTS DATAFRAME
# =========================================================

results = pd.DataFrame(
    all_results
)


# =========================================================
# SAVE RAW RESULTS
# =========================================================

results.to_csv(
    OUT_FILE,
    index=False
)


# =========================================================
# OVERALL METRICS
# =========================================================

print("\n========================================")
print("CLEAN SPATIAL HOLDOUT RESULTS")
print("========================================")


summary_rows = []


for feature in state_features:

    subset = results[
        results["feature"] == feature
    ]


    if len(subset) == 0:
        continue


    y_true = (
        subset["observation"]
        .values
    )

    y_pred = (
        subset["prediction"]
        .values
    )


    feature_rmse = rmse(
        y_true,
        y_pred
    )

    feature_mae = np.mean(
        np.abs(
            y_true - y_pred
        )
    )


    print(
        f"{feature:8s} "
        f"RMSE = {feature_rmse:.4f} "
        f"MAE = {feature_mae:.4f}"
    )


    summary_rows.append({

        "feature":
            feature,

        "rmse":
            feature_rmse,

        "mae":
            feature_mae,

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
# PER-SITE METRICS
# =========================================================

print("\n========================================")
print("PER-SITE RMSE")
print("========================================")


for site in sites:

    site_results = results[
        results["held_out_site"]
        == site
    ]


    if len(site_results) == 0:
        continue


    print("\n", site)


    for feature in state_features:

        subset = site_results[
            site_results["feature"]
            == feature
        ]


        if len(subset) == 0:
            continue


        site_rmse = rmse(
            subset["observation"].values,
            subset["prediction"].values
        )


        print(
            f"  {feature:8s} "
            f"RMSE = {site_rmse:.4f}"
        )


# =========================================================
# FINAL INFORMATION
# =========================================================

print("\n========================================")

print(
    "Rows evaluated:",
    len(results)
)

print(
    "Sites evaluated:",
    results[
        "held_out_site"
    ].nunique()
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
