import pandas as pd

rows = [

    {
        "Experiment": "Dataset",
        "Metric": "Site-month records",
        "Value": 1750,
        "Units": "records"
    },
    {
        "Experiment": "Dataset",
        "Metric": "Sites",
        "Value": 25,
        "Units": "sites"
    },
    {
        "Experiment": "Dataset",
        "Metric": "Crops",
        "Value": 5,
        "Units": "crops"
    },
    {
        "Experiment": "Dataset",
        "Metric": "Countries",
        "Value": 14,
        "Units": "countries"
    },
    {
        "Experiment": "Dataset",
        "Metric": "Complete multimodal records",
        "Value": 89.83,
        "Units": "%"
    },

    {
        "Experiment": "Temporal prediction",
        "Metric": "2024 EVI RMSE",
        "Value": 0.566370,
        "Units": "standardized observation units"
    },
    {
        "Experiment": "Temporal prediction",
        "Metric": "2024 NDVI RMSE",
        "Value": 0.490082,
        "Units": "standardized observation units"
    },
    {
        "Experiment": "Temporal prediction",
        "Metric": "2024 NDWI RMSE",
        "Value": 0.496894,
        "Units": "standardized observation units"
    },
    {
        "Experiment": "Temporal prediction",
        "Metric": "2024 VH RMSE",
        "Value": 0.640204,
        "Units": "standardized observation units"
    },
    {
        "Experiment": "Temporal prediction",
        "Metric": "2024 VV RMSE",
        "Value": 0.467636,
        "Units": "standardized observation units"
    },
    {
        "Experiment": "Temporal prediction",
        "Metric": "2024 VV/VH RMSE",
        "Value": 0.784678,
        "Units": "standardized observation units"
    },
    {
        "Experiment": "Temporal prediction",
        "Metric": "2024 mean RMSE",
        "Value": 0.584825,
        "Units": "mean across six variables"
    },

    {
        "Experiment": "Spatial generalization",
        "Metric": "Held-out sites",
        "Value": 5,
        "Units": "sites"
    },
    {
        "Experiment": "Spatial generalization",
        "Metric": "Held-out site RMSE",
        "Value": 1.993548,
        "Units": "observation-space RMSE"
    },

    {
        "Experiment": "Latent assimilation",
        "Metric": "Free-running RMSE",
        "Value": 0.580016,
        "Units": "observation-space mean RMSE"
    },
    {
        "Experiment": "Latent assimilation",
        "Metric": "S1-only RMSE",
        "Value": 0.581022,
        "Units": "observation-space mean RMSE"
    },
    {
        "Experiment": "Latent assimilation",
        "Metric": "S2-only RMSE",
        "Value": 0.580894,
        "Units": "observation-space mean RMSE"
    },
    {
        "Experiment": "Latent assimilation",
        "Metric": "S1+S2 RMSE",
        "Value": 0.580304,
        "Units": "observation-space mean RMSE"
    },

    {
        "Experiment": "Uncertainty",
        "Metric": "Spread-error Pearson correlation",
        "Value": 0.116720,
        "Units": "r"
    },

    {
        "Experiment": "Deviation analysis",
        "Metric": "2024 evaluated feature-months",
        "Value": 1218,
        "Units": "feature-months"
    },
    {
        "Experiment": "Deviation analysis",
        "Metric": "Multi-sensor deviation site-months",
        "Value": 17,
        "Units": "site-months"
    },
    {
        "Experiment": "Deviation analysis",
        "Metric": "Sites with multi-sensor events",
        "Value": 9,
        "Units": "sites"
    },
    {
        "Experiment": "Deviation analysis",
        "Metric": "Persistent consecutive multi-sensor months",
        "Value": 2,
        "Units": "site-months"
    },
    {
        "Experiment": "Deviation analysis",
        "Metric": "Persistent event site",
        "Value": "WHEAT_AU_04",
        "Units": ""
    }
]

out = pd.DataFrame(rows)

out.to_csv(
    "FINAL_GLOBAL_TWIN_RESULTS_SUMMARY.csv",
    index=False
)

print(
    out.to_string(index=False)
)

print(
    "\nSaved FINAL_GLOBAL_TWIN_RESULTS_SUMMARY.csv"
)
