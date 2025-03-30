# Flow Projector

A web application that forecasts future streamflow based on weather predictions and historical river data from USGS streamgages. This tool helps water resource managers, researchers, and the public visualize potential changes in river flow over the coming weeks and months.

## What it does

- Forecasts river flow up to 90+ days into the future
- Connects to real-time USGS streamflow data
- Uses rainfall predictions from major weather forecasting models (ECMWF, GFS, or DWD)
- Shows both historical flow data and future projections in interactive charts
- Provides downloadable data for further analysis
- Adapts to different watershed sizes with smart rainfall sampling

## Best Application Period

The app performs most accurately during the later wet season months and early spring, when vadose zone deficits are small and most rainfall is converted to groundwater recharge (versus being stored in the unsaturated zone).

## Methodology

The app applies a storage-discharge modeling approach, building on methods established in:

- Kirchner, J. W. (2009). Catchments as simple dynamical systems: Catchment characterization, rainfall-runoff modeling, and doing hydrology backward. Water Resources Research, 45(2).

- Dralle, D. N., Hahm, W. J., Rempe, D. M., Karst, N. J., Thompson, S. E., & Dietrich, W. E. (2018). Quantification of the seasonal hillslope water storage that does not drive streamflow. Hydrological Processes, 32(13), 1978-1992.

## Limitations

This model works best for rain-dominated watersheds and has limited applicability in snow-affected regions. All flow values represent daily averages rather than instantaneous flows.