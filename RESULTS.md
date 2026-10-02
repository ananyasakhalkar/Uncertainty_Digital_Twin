# 3. Results

## 3.1 Global multi-crop observation framework

The final study dataset comprised 25 geographically distributed sites representing five crops: cotton, maize, rice, soybean, and wheat, across 14 countries. The monthly study period extended from January 2019 through October 2024, yielding 1,750 site-month records. Sentinel-1, Sentinel-2, and ERA5-Land observations were retained as asynchronous observations rather than requiring simultaneous availability from both satellite sensors.

Following preprocessing and the replacement of one Australian wheat site with a location having complete weather coverage, 89.83% of the site-month records contained complete multimodal observations. The remaining records retained partial satellite availability and were not discarded solely because one satellite modality was unavailable.

This structure provided a common multi-crop framework for evaluating temporal prediction, spatial transfer, sequential observation updating, uncertainty representation, and model-observation deviation detection.

## 3.2 Temporal prediction of crop-condition observations

The latent transition model was trained using observations from 2019–2022, with 2023 used for validation and 2024 retained for evaluation. The model represented crop condition using a four-dimensional learned latent state and conditioned its monthly transition on three months of weather history and crop identity.

On the 2024 evaluation period, the model produced the following observation-space RMSE values after decoding the predicted latent state:

| Variable | RMSE |
|---|---:|
| EVI | 0.566 |
| NDVI | 0.490 |
| NDWI | 0.497 |
| VH | 0.640 |
| VV | 0.468 |
| VV/VH | 0.785 |
| Mean across six variables | 0.585 |

The reported variables were evaluated in standardized observation space. Consequently, the mean RMSE is used as a compact summary rather than as a physically homogeneous error metric across optical indices and radar backscatter variables.

The results demonstrate that the learned latent dynamics can propagate the crop-condition representation forward in time and reconstruct multiple optical and radar observations for an unseen temporal period.

## 3.3 Spatial generalization

A separate spatial holdout experiment was performed to assess geographic transfer. Five sites were excluded from model training, with one held-out site and country represented for each crop. The spatial training and test sets had no overlapping sites.

The model was evaluated on 5 geographically held-out sites and 239 site-month transitions. The overall observation-space RMSE was 1.994.

The site-level evaluation demonstrated that the shared model could be applied to locations that were not used during model fitting. Because the spatial and temporal experiments used different evaluation protocols, their RMSE values are reported separately and are not interpreted as directly comparable accuracy scores.

## 3.4 Sequential observation assimilation

A 50-member ensemble Kalman filter was used to evaluate sequential updating of the learned latent state using Sentinel-1 and Sentinel-2 observations. Four modes were evaluated: free-running prediction, Sentinel-1-only assimilation, Sentinel-2-only assimilation, and combined Sentinel-1/Sentinel-2 assimilation.

The resulting observation-space mean RMSE values were:

| Assimilation mode | RMSE |
|---|---:|
| Free running | 0.5800 |
| Sentinel-1 only | 0.5810 |
| Sentinel-2 only | 0.5809 |
| Sentinel-1 + Sentinel-2 | 0.5803 |

The differences between assimilation modes were small. The combined assimilation configuration therefore did not produce a measurable point-prediction improvement under the present experimental formulation.

These results demonstrate the operational integration of asynchronous observations into the latent-state framework, while also indicating that the current observation-update formulation requires further development before assimilation can be interpreted as an accuracy-enhancing component.

## 3.5 Ensemble uncertainty characterization

Ensemble dispersion was evaluated as an uncertainty indicator by comparing forecast ensemble spread with subsequent observation-space error.

The overall Pearson correlation between ensemble spread and absolute error was 0.117. Relationships were stronger for some optical variables than for the radar variables, but the overall association remained weak.

Accordingly, ensemble spread is treated in this study as an initial uncertainty representation rather than as a fully calibrated probabilistic forecast. Formal reliability and coverage calibration would require a dedicated calibration period independent of the final evaluation period.

## 3.6 Model-observation deviation detection

A historical model-error distribution derived from 2019–2023 was used to standardize 2024 model-observation deviations. For each observation variable, the absolute prediction error was compared with a feature-specific robust historical error scale based on the median absolute deviation.

The 2024 deviation analysis contained 1,218 evaluated feature-month observations. Seventeen site-months across nine sites contained at least two observation variables exceeding their respective historical 95th-percentile error reference.

The strongest persistent multi-sensor pattern occurred at the Australian wheat site WHEAT_AU_04. The site exhibited multi-sensor deviation events in February, April, May, June, August, and October 2024. May and June constituted two consecutive multi-sensor deviation months, with three and two features respectively exceeding their historical 95th-percentile error references.

Other multi-sensor events occurred at sites representing cotton, maize, soybean, rice, and wheat. Individual large deviations did not necessarily correspond to multi-sensor or persistent events. For example, the largest individual calibrated deviation occurred for EVI at the Vietnam rice site RICE_02 in April 2024, whereas this site did not produce a multi-sensor event.

These events are interpreted as departures between observations and the model's learned trajectory. They are not interpreted as validated crop-stress events because the present dataset does not contain independent ground-truth stress labels.

## 3.7 Summary of principal findings

The experiments establish a global, crop-conditioned latent dynamical framework capable of integrating Sentinel-1, Sentinel-2, and weather observations across multiple crops and geographic locations. The temporal experiment demonstrated forward prediction of unseen 2024 observations, while the spatial holdout demonstrated evaluation at geographically excluded sites.

Sequential assimilation was successfully implemented but produced near-neutral point-prediction changes under the present formulation. Ensemble uncertainty showed only a modest relationship with subsequent error and therefore remains insufficiently calibrated for probabilistic interpretation.

The deviation-analysis component provided a complementary monitoring capability by identifying model-observation departures and persistent multi-sensor events. The results support interpreting the digital twin as a dynamic prediction and monitoring framework, while independent validation of crop-stress meaning remains a necessary future step.
