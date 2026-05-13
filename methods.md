# Flow Projector: Storage-Discharge Projection Methods

This document outlines the methodology used by the Flow Projector application to forecast future streamflow based on historical recession behavior and predicted precipitation. The model assumes a dynamic storage-discharge relationship for the watershed, identifying how fast the basin drains given its current storage state.

## 1. Data Selection & Filtering
* **Historical Window:** The model retrieves the last 10 years of historical daily flow data from the selected USGS streamgage.
* **Seasonal Matching:** It isolates historical days that match the exact same day-of-year window as the projection period. This controls for seasonal variations in evapotranspiration and soil moisture.
* **Precipitation Filtering:** Historical precipitation data is retrieved at the basin's centroid. Any historical days where rainfall exceeded 1 mm are excluded from the analysis to isolate periods of true streamflow recession.
* **Strict Recession Definition:** A given day is identified as part of a recession only if the streamflow is strictly positive ($Q > 0$) and actively decreasing ($dQ/dt < 0$).

## 2. Binning Approach (Kirchner 2009)
Because day-to-day values of the flow derivative ($dQ/dt$) can be highly noisy due to measurement error or minor diurnal fluctuations, the model uses a binning method based on Kirchner (2009):
* Valid pairs of $(Q, -dQ/dt)$ are sorted in descending order by flow rate ($Q$).
* The data is grouped into approximately 30 bins, ensuring a minimum of 10 data points per bin.
* For each bin, the mean flow, the mean flow derivative, and the standard error of the log derivative are calculated. This process reveals a cleaner, underlying functional relationship between flow and its rate of decline.

## 3. Curve Fitting
* **Dynamic Power Law:** Rather than a simple power law, which assumes a linear relationship in log-log space, the model fits the binned data to a dynamic power-law function as proposed by Wlostowski et al.
* The functional form is given by:
  $$\ln(-dQ/dt) = \ln(a) + b(Q) \cdot \ln(Q/\bar{Q})$$
  where $b(Q)$ varies between a lower and upper limit depending on the flow state.
* **Error-Weighted Fit:** The curve fitting algorithm weights each bin by its standard error ($\sigma$). Bins with less variance exert a stronger pull on the fitted curve, ensuring the model accurately captures the most reliable recession patterns.
* This fit yields a continuous sensitivity function, $g(Q) = -\frac{dQ/dt}{Q}$, which describes the relative rate of watershed drainage at any arbitrary flow state.

## 4. Projection via Numerical Integration
* To generate the actual flow forecast, the model defines the initial value problem:
  $$\frac{dQ}{dt} = -g(Q)(Q - P(t))$$
* **Forcing ($P(t)$):** This term represents the incoming rainfall forecast, taken from the chosen weather model (e.g., GFS, DWD, GEM) and converted from depth (mm) into a volumetric flow rate (cfs) over the total watershed area.
* **Integration:** Starting from the current, real-time measured flow ($Q_0$) at $t=0$, the application integrates this differential equation forward using numerical methods (SciPy's `solve_ivp`) across the duration of the projection period.

## References
* Kirchner, J. W. (2009). Catchments as simple dynamical systems: Catchment characterization, rainfall-runoff modeling, and doing hydrology backward. *Water Resources Research*, 45(2).
* Wlostowski et al. (Dynamic power law approaches to streamflow recession analysis)
* Dralle, D. N., Hahm, W. J., Rempe, D. M., Karst, N. J., Thompson, S. E., & Dietrich, W. E. (2018). Quantification of the seasonal hillslope water storage that does not drive streamflow. *Hydrological Processes*, 32(13), 1978-1992.
