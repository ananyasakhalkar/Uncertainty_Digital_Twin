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


# =========================================================
# PATHS
# =========================================================

BASE = Path.home() / "Downloads" / "Crop_Digital_Twin"

DATA = BASE / "data" / "global"

DATA_FILE = DATA / "US_global_panel_2019_2024.csv"

OUT_FILE = DATA / "US_assimilation_ablation_results.csv"

SUMMARY_FILE = DATA / "US_assimilation_ablation_summary.csv"


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

STATE_FEATURES = (
    S2_FEATURES +
    S1_FEATURES
)

WEATHER_FEATURES = [
    "temperature_2m",
    "precipitation",
    "solar_radiation"
]


# =========================================================
# ASSIMILATION SETTINGS
# =========================================================

ENSEMBLE_SIZE = 50

PROCESS_NOISE = 0.05

OBSERVATION_NOISE = 0.15


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


for c in (
    STATE_FEATURES +
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
# BUILD CONSECUTIVE TRAINING TRANSITIONS
# =========================================================

def build_training_transitions(
    data
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


            required_current = (
                STATE_FEATURES +
                WEATHER_FEATURES
            )


            if (
                current[
                    required_current
                ].isna().any()
            ):

                continue


            if (
                nxt[
                    STATE_FEATURES
                ].isna().any()
            ):

                continue


            rows.append({

                "state":
                    current[
                        STATE_FEATURES
                    ].values.astype(float),

                "weather":
                    current[
                        WEATHER_FEATURES
                    ].values.astype(float),

                "target":
                    nxt[
                        STATE_FEATURES
                    ].values.astype(float),

                "crop":
                    current["crop"]

            })


    return rows


# =========================================================
# TRAIN MODEL
# =========================================================

def train_model(
    train_scaled
):

    transitions = (
        build_training_transitions(
            train_scaled
        )
    )


    print(
        "Training transitions:",
        len(transitions)
    )


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

    Y = np.stack(
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


    X_state = torch.tensor(
        X_state,
        dtype=torch.float32,
        device=DEVICE
    )

    X_weather = torch.tensor(
        X_weather,
        dtype=torch.float32,
        device=DEVICE
    )

    Y = torch.tensor(
        Y,
        dtype=torch.float32,
        device=DEVICE
    )

    X_crop = torch.tensor(
        X_crop,
        dtype=torch.long,
        device=DEVICE
    )


    model = CropTwin().to(DEVICE)


    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=1e-3
    )

    loss_fn = nn.MSELoss()


    epochs = 500

    batch_size = 32

    n = len(X_state)


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
                X_state[idx],
                X_weather[idx],
                X_crop[idx]
            )


            loss = loss_fn(
                prediction,
                Y[idx]
            )


            loss.backward()


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


    return model


# =========================================================
# ENSEMBLE FORECAST
# =========================================================

def forecast_ensemble(
    model,
    state,
    weather,
    crop_id
):

    # Repeat current posterior state
    ensemble = (
        state
        .unsqueeze(0)
        .repeat(
            ENSEMBLE_SIZE,
            1
        )
    )


    # Add small process uncertainty
    noise = torch.randn_like(
        ensemble
    ) * PROCESS_NOISE


    ensemble = (
        ensemble + noise
    )


    weather_batch = (
        weather
        .unsqueeze(0)
        .repeat(
            ENSEMBLE_SIZE,
            1
        )
    )


    crop_batch = torch.full(
        (
            ENSEMBLE_SIZE,
        ),
        crop_id,
        dtype=torch.long,
        device=DEVICE
    )


    with torch.no_grad():

        forecast = model(
            ensemble,
            weather_batch,
            crop_batch
        )


    return forecast


# =========================================================
# ENKF UPDATE
# =========================================================

def enkf_update(
    ensemble,
    observation,
    obs_indices
):

    if len(obs_indices) == 0:

        return ensemble


    # -----------------------------------------------------
    # Ensemble observation matrix
    # -----------------------------------------------------

    Y = ensemble[
        :,
        obs_indices
    ]


    # -----------------------------------------------------
    # Observation noise
    # -----------------------------------------------------

    obs_noise = (
        torch.randn(
            ENSEMBLE_SIZE,
            len(obs_indices),
            device=DEVICE
        )
        *
        OBSERVATION_NOISE
    )


    perturbed_obs = (
        observation
        .unsqueeze(0)
        .repeat(
            ENSEMBLE_SIZE,
            1
        )
        +
        obs_noise
    )


    # -----------------------------------------------------
    # Ensemble means
    # -----------------------------------------------------

    x_mean = ensemble.mean(
        dim=0
    )

    y_mean = Y.mean(
        dim=0
    )


    X_anom = (
        ensemble
        -
        x_mean
    )

    Y_anom = (
        Y
        -
        y_mean
    )


    # -----------------------------------------------------
    # Covariances
    # -----------------------------------------------------

    denom = max(
        ENSEMBLE_SIZE - 1,
        1
    )


    P_xy = (
        X_anom.T
        @
        Y_anom
        /
        denom
    )


    P_yy = (
        Y_anom.T
        @
        Y_anom
        /
        denom
    )


    R = (
        torch.eye(
            len(obs_indices),
            device=DEVICE
        )
        *
        (
            OBSERVATION_NOISE ** 2
        )
    )


    # -----------------------------------------------------
    # Kalman gain
    # -----------------------------------------------------

    innovation_cov = (
        P_yy + R
    )


    K = (
        P_xy
        @
        torch.linalg.pinv(
            innovation_cov
        )
    )


    # -----------------------------------------------------
    # Update
    # -----------------------------------------------------

    innovation = (
        perturbed_obs
        -
        Y
    )


    updated = (
        ensemble
        +
        innovation
        @
        K.T
    )


    return updated


# =========================================================
# RUN ONE ASSIMILATION MODE
# =========================================================

def run_mode(
    model,
    test_scaled,
    mode,
    state_mean,
    state_std
):

    print("\n")
    print("========================================")
    print("ASSIMILATION MODE:", mode)
    print("========================================")


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


        # -------------------------------------------------
        # Find December 2023 initial state
        # -------------------------------------------------

        initial_candidates = group[
            group["date"]
            ==
            pd.Timestamp(
                "2023-12-01"
            )
        ]


        if len(initial_candidates) == 0:

            continue


        initial = (
            initial_candidates.iloc[0]
        )


        if (
            initial[
                STATE_FEATURES
            ].isna().any()
        ):

            continue


        crop_name = initial["crop"]

        crop_id = crop_to_id[
            crop_name
        ]


        initial_state = torch.tensor(
            initial[
                STATE_FEATURES
            ].values.astype(float),
            dtype=torch.float32,
            device=DEVICE
        )


        # -------------------------------------------------
        # Free-running state
        # -------------------------------------------------

        state = initial_state.clone()


        # -------------------------------------------------
        # 2024 months
        # -------------------------------------------------

        months = group[
            group["date"]
            >=
            pd.Timestamp(
                "2024-01-01"
            )
        ]


        previous_date = (
            pd.Timestamp(
                "2023-12-01"
            )
        )


        for _, row in months.iterrows():

            current_date = row["date"]


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


            # -------------------------------------------------
            # Weather for transition:
            #
            # Dec 2023 -> Jan 2024 uses Dec weather.
            # Jan -> Feb uses Jan weather, etc.
            #
            # Therefore we retrieve weather from the previous
            # month, not the target month.
            # -------------------------------------------------

            previous_rows = group[
                group["date"]
                ==
                previous_date
            ]


            if len(previous_rows) == 0:

                previous_date = (
                    current_date
                )

                continue


            previous_row = (
                previous_rows.iloc[0]
            )


            weather_values = (
                previous_row[
                    WEATHER_FEATURES
                ]
                .values.astype(float)
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


            weather = torch.tensor(
                weather_values,
                dtype=torch.float32,
                device=DEVICE
            )


            # -------------------------------------------------
            # PRIOR FORECAST
            # -------------------------------------------------

            ensemble = forecast_ensemble(
                model,
                state,
                weather,
                crop_id
            )


            prior_mean = (
                ensemble.mean(
                    dim=0
                )
            )


            # -------------------------------------------------
            # Evaluate PRIOR against current observation
            #
            # This is the important metric.
            # The current observation has NOT been assimilated.
            # -------------------------------------------------

            observation_values = (
                row[
                    STATE_FEATURES
                ]
                .values.astype(float)
            )


            for j, feature in enumerate(
                STATE_FEATURES
            ):

                if np.isfinite(
                    observation_values[j]
                ):

                    error = (
                        prior_mean[j].item()
                        -
                        observation_values[j]
                    )


                    results.append({

                        "mode":
                            mode,

                        "site_id":
                            site_id,

                        "crop":
                            crop_name,

                        "date":
                            current_date,

                        "feature":
                            feature,

                        "prior_prediction":
                            prior_mean[j].item(),

                        "observation":
                            observation_values[j],

                        "squared_error":
                            error ** 2

                    })


            # -------------------------------------------------
            # ASSIMILATION
            # -------------------------------------------------

            if mode == "free_running":

                # No observation update.
                state = prior_mean


            else:

                if mode == "S1_only":

                    obs_names = S1_FEATURES

                elif mode == "S2_only":

                    obs_names = S2_FEATURES

                elif mode == "S1_plus_S2":

                    obs_names = STATE_FEATURES

                else:

                    raise ValueError(
                        "Unknown mode"
                    )


                obs_indices = [
                    STATE_FEATURES.index(
                        x
                    )
                    for x in obs_names
                    if np.isfinite(
                        observation_values[
                            STATE_FEATURES.index(x)
                        ]
                    )
                ]


                if len(obs_indices) > 0:

                    observation_tensor = (
                        torch.tensor(
                            observation_values[
                                obs_indices
                            ],
                            dtype=torch.float32,
                            device=DEVICE
                        )
                    )


                    ensemble = enkf_update(
                        ensemble,
                        observation_tensor,
                        obs_indices
                    )


                # Posterior becomes next month's initial state.
                state = ensemble.mean(
                    dim=0
                )


            previous_date = (
                current_date
            )


    return results


# =========================================================
# TRAINING DATA
# =========================================================

train_df = df[
    df["date"]
    <=
    pd.Timestamp(
        "2023-12-01"
    )
].copy()


test_df = df[
    df["date"]
    >=
    pd.Timestamp(
        "2023-12-01"
    )
].copy()


# =========================================================
# TRAIN-ONLY STANDARDIZATION
# =========================================================

state_mean = (
    train_df[
        STATE_FEATURES
    ]
    .mean()
)

state_std = (
    train_df[
        STATE_FEATURES
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


for c in STATE_FEATURES:

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


# =========================================================
# TRAIN MODEL
# =========================================================

print("\n")
print("========================================")
print("TRAINING DYNAMIC TWIN")
print("========================================")


model = train_model(
    train_scaled
)


# =========================================================
# RUN FOUR MODES
# =========================================================

all_results = []


modes = [
    "free_running",
    "S1_only",
    "S2_only",
    "S1_plus_S2"
]


for mode in modes:

    output = run_mode(
        model,
        test_scaled,
        mode,
        state_mean,
        state_std
    )

    all_results.extend(
        output
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
print("NEXT-MONTH ASSIMILATION RESULTS")
print("========================================")


summary_rows = []


for mode in modes:

    mode_df = results[
        results["mode"]
        == mode
    ]


    print("\n", mode)


    for feature in STATE_FEATURES:

        subset = mode_df[
            mode_df["feature"]
            == feature
        ]


        if len(subset) == 0:
            continue


        rmse = np.sqrt(
            np.mean(
                subset[
                    "squared_error"
                ]
            )
        )


        print(
            f"  {feature:8s} "
            f"RMSE = {rmse:.4f}"
        )


        summary_rows.append({

            "mode":
                mode,

            "feature":
                feature,

            "rmse":
                rmse,

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
# MODE-LEVEL MEAN RMSE
# =========================================================

print("\n")
print("========================================")
print("MEAN RMSE ACROSS OBSERVED VARIABLES")
print("========================================")


for mode in modes:

    subset = summary[
        summary["mode"]
        == mode
    ]


    if len(subset) == 0:
        continue


    mean_rmse = (
        subset["rmse"]
        .mean()
    )


    print(
        f"{mode:16s} "
        f"{mean_rmse:.4f}"
    )


# =========================================================
# ASSIMILATION GAIN
# =========================================================

print("\n")
print("========================================")
print("RMSE REDUCTION RELATIVE TO FREE RUNNING")
print("========================================")


free = summary[
    summary["mode"]
    == "free_running"
].set_index(
    "feature"
)


for mode in [
    "S1_only",
    "S2_only",
    "S1_plus_S2"
]:

    current = summary[
        summary["mode"]
        == mode
    ].set_index(
        "feature"
    )


    print("\n", mode)


    for feature in STATE_FEATURES:

        if feature not in free.index:
            continue

        if feature not in current.index:
            continue


        free_rmse = free.loc[
            feature,
            "rmse"
        ]

        current_rmse = current.loc[
            feature,
            "rmse"
        ]


        reduction = (
            100 *
            (
                free_rmse
                -
                current_rmse
            )
            /
            free_rmse
        )


        print(
            f"  {feature:8s} "
            f"{reduction:.2f}%"
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
