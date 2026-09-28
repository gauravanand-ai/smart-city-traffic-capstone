# SHAP Explainability -- Top Drivers of the Next-Hour Traffic Forecast

1. **traffic volume one hour earlier** -- mean |SHAP| impact of 1,500 veh/hr. Higher values of this feature push the forecast UP (Pearson correlation between feature value and SHAP value: r=+0.99).
2. **hour of day (cyclical encoding)** -- mean |SHAP| impact of 322 veh/hr. This is a cyclical (sine/cosine) encoding, so its effect on the forecast depends on which specific hour/day it represents rather than a single 'higher is more' direction -- consistent with the rush-hour and weekday/weekend patterns found throughout this project.
3. **traffic volume 24 hours earlier (same hour, previous day)** -- mean |SHAP| impact of 195 veh/hr. Higher values of this feature push the forecast UP (Pearson correlation between feature value and SHAP value: r=+0.84).
4. **average traffic over the past 6 hours** -- mean |SHAP| impact of 87 veh/hr. Higher values of this feature push the forecast DOWN (Pearson correlation between feature value and SHAP value: r=-0.55).
5. **day of week (cyclical encoding)** -- mean |SHAP| impact of 29 veh/hr. This is a cyclical (sine/cosine) encoding, so its effect on the forecast depends on which specific hour/day it represents rather than a single 'higher is more' direction -- consistent with the rush-hour and weekday/weekend patterns found throughout this project.
6. **whether it's a weekend** -- mean |SHAP| impact of 22 veh/hr. Higher values of this feature push the forecast DOWN (Pearson correlation between feature value and SHAP value: r=-0.74).
