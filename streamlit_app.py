from termios import TAB0
import streamlit as st
import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d
from pynhd import NLDI
from pygeohydro import NWIS
import datetime
import warnings
from streamlit import cache_data
import plotly.graph_objects as go
from scipy.optimize import curve_fit
import scipy.special
import requests
import pytz
import matplotlib.pyplot as plt
import seaborn as sns
import os
import base64

warnings.filterwarnings('ignore')

# Define Los Angeles timezone
LA_TIMEZONE = pytz.timezone('America/Los_Angeles')

# Ensure plots directory exists
PLOTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'plots')
os.makedirs(PLOTS_DIR, exist_ok=True)

# Function to get flow data
@cache_data
def getFlow(site, start, stop):
    nwis = NWIS()
    nldi = NLDI()
    
    # Get current date in LA timezone
    today = datetime.datetime.now(LA_TIMEZONE).date()
    yesterday = today - datetime.timedelta(days=1)
    yesterday_str = yesterday.strftime('%Y-%m-%d')
    two_days_ago = today - datetime.timedelta(days=2)
    two_days_ago_str = two_days_ago.strftime('%Y-%m-%d')
    
    # Get daily values (dv) up to yesterday
    df_dv = nwis.get_streamflow(site, (start, yesterday_str), freq="dv")
    df_dv.columns = ['q']
    df_dv.index = pd.to_datetime(df_dv.index)
    df_dv = df_dv.tz_convert(LA_TIMEZONE) if df_dv.index.tz is not None else df_dv.tz_localize(LA_TIMEZONE)
    
    # Get instantaneous values (iv) from 2 days ago to tomorrow to ensure we get today's data
    today_str = today.strftime('%Y-%m-%d')
    tomorrow = today + datetime.timedelta(days=1)
    tomorrow_str = tomorrow.strftime('%Y-%m-%d')
    # For IV data, we need to use 2 days ago to tomorrow to get complete data for today
    try:
        # Using 2-days-ago-to-tomorrow range to ensure we get all of today's and yesterday's data
        df_iv = nwis.get_streamflow(site, (two_days_ago_str, tomorrow_str), freq="iv")
        df_iv.columns = ['q']
        df_iv.index = pd.to_datetime(df_iv.index)
        df_iv = df_iv.tz_convert(LA_TIMEZONE) if df_iv.index.tz is not None else df_iv.tz_localize(LA_TIMEZONE)
      
        # Check if yesterday's daily value is missing in df_dv
        yesterday_timestamp = pd.Timestamp(yesterday).tz_localize(LA_TIMEZONE)
        yesterday_missing = yesterday_timestamp not in df_dv.index
        yesterday_df = None
        
        if yesterday_missing and not df_iv.empty:
            # Filter to get yesterday's instantaneous data
            yesterday_mask = (df_iv.index.date == yesterday)
            yesterday_iv = df_iv[yesterday_mask]
            
            if not yesterday_iv.empty:
                # Calculate the mean of yesterday's iv data
                yesterday_mean = yesterday_iv['q'].mean()
                yesterday_df = pd.DataFrame({'q': [yesterday_mean]}, index=[yesterday_timestamp])
                # st.info(f"Using average of yesterday's instantaneous values: {yesterday_mean:.2f} cms to fill missing daily value")
        
        # Calculate the mean of today's iv data
        today_df = None
        if not df_iv.empty:
            # Filter to get only today's data
            mask = (df_iv.index.date == today)
            today_iv = df_iv[mask]
            
            if not today_iv.empty:
                today_mean = today_iv['q'].mean()
                today_df = pd.DataFrame({'q': [today_mean]}, index=[pd.Timestamp(today).tz_localize(LA_TIMEZONE)])
            else:
                # st.warning(f"No instantaneous values found for today ({today}). Using yesterday's final instantaneous value.")
                # Filter to get yesterday's data
                yesterday_mask = (df_iv.index.date == yesterday)
                yesterday_iv = df_iv[yesterday_mask]
                
                if not yesterday_iv.empty:
                    # Get the last instantaneous value from yesterday
                    yesterday_final_value = yesterday_iv['q'].iloc[-1]
                    today_df = pd.DataFrame({'q': [yesterday_final_value]}, index=[pd.Timestamp(today).tz_localize(LA_TIMEZONE)])
                    # st.info(f"Using yesterday's final instantaneous value: {yesterday_final_value:.2f} cms")
                else:
                    # st.warning(f"No instantaneous values found for yesterday either. Using only historical data.")
                    pass
            
            # Combine historical data with yesterday's and today's data if available
            if yesterday_df is not None or today_df is not None:
                # Start with historical data
                combined_dfs = [df_dv]
                
                # Add yesterday's data if available
                if yesterday_df is not None:
                    combined_dfs.append(yesterday_df)
                
                # Add today's data if available
                if today_df is not None:
                    combined_dfs.append(today_df)
                    if 'today_mean' in locals():
                        st.session_state.flow_combination_message = f"Successfully combined historical daily values with today's average flow: {today_mean:.2f} cms"
                
                # Combine all available data
                df = pd.concat(combined_dfs)
                # Sort by date index to ensure chronological order
                df = df.sort_index()
            else:
                df = df_dv
        else:
            df = df_dv
    except Exception as e:
        # st.warning(f"Could not fetch instantaneous values for today: {e}")
        df = df_dv
    
    # Resample to daily frequency
    df = df.resample('D').mean()
    df.columns = [site]
    
    # Get basin information
    basin = nldi.get_basins(site).to_crs('epsg:26910')

    try:
        geoms = [item for item in list(basin.geometry[0])]
        idx = np.argmax([item.area for item in geoms])
        basin.geometry = [geoms[idx]]
        st.info('Found multipolygon - fixing')
    except:
        basin.geometry = basin.geometry
    area_mm2 = basin.to_crs('epsg:26910').geometry[0].area*1000**2
    df = df * 35.3147  # Convert from cms to cfs
    return df, basin

# Sensitivity function
@cache_data
def gQ(q, p):
    if np.size(np.array(q)) == 1:
        return np.exp(np.sum([p[i] * np.log(q) ** (len(p) - i - 1) for i in range(len(p))]))
    return [np.exp(np.sum([p[i] * np.log(qq) ** (len(p) - i - 1) for i in range(len(p))])) for qq in np.array(q)]

@cache_data
def predefined_binning_indices(df, num_bins=10, min_per_bin=10, loud=False):
    """
    Predefine bins for the DataFrame based on the logarithm of 'q' and
    return a list of row indices representing the bin boundaries.

    Each bin will have at least min_per_bin datapoints (if possible) by
    merging candidate bins that don't reach the threshold.

    Parameters:
      df (pd.DataFrame): DataFrame with at least a 'q' column.
      num_bins (int): Number of bins to attempt (using log-spaced edges).
      min_per_bin (int): Minimum number of data points required per bin.
      loud (bool): If True, prints debug information.

    Returns:
      List[int]: Sorted list of row indices that serve as bin boundaries.
    """
    # Sort the DataFrame in descending order of q and reset the index.
    df_sorted = df.sort_values('q', ascending=False).reset_index(drop=True)

    # Compute the natural logarithm of q.
    logQ = np.log(df_sorted['q']).values

    # Define bin edges in log space (from max to min of logQ).
    bin_edges = np.linspace(logQ.max(), logQ.min(), num_bins + 1)
    if loud:
        print("Predefined bin edges:", bin_edges)

    # Start with the first index (0).
    boundaries = [0]
    last_boundary = 0

    # Loop over the candidate bin edges (skip the first, which is logQ.max()).
    for edge in bin_edges[1:]:
        # Find the first index where logQ <= edge.
        indices = np.where(logQ <= edge)[0]
        if len(indices) == 0:
            continue
        candidate = indices[0]
        # Only add the candidate if it creates a bin with at least min_per_bin datapoints.
        if candidate - last_boundary < min_per_bin:
            if loud:
                print(f"Candidate boundary {candidate} skipped (bin size {candidate - last_boundary} < {min_per_bin})")
            continue
        # Accept candidate boundary.
        boundaries.append(candidate)
        last_boundary = candidate

    # Make sure the final bin (from last boundary to end of data) has at least min_per_bin points.
    total_points = len(df_sorted)
    if boundaries[-1] != total_points:
        if total_points - boundaries[-1] < min_per_bin and len(boundaries) > 1:
            if loud:
                print("Final bin too small; merging with previous bin.")
            boundaries.pop()  # Remove the last boundary so the bin becomes larger.
        boundaries.append(total_points)

    if loud:
        print("Final bin boundary indices:", boundaries)
    return boundaries

def generate_sample_points(basin_geometry, num_points):
    """
    Generate evenly distributed sample points within a watershed.
    
    Args:
        basin_geometry: The basin geometry (shapely Polygon or MultiPolygon)
        num_points: Number of points to generate
        
    Returns:
        List of (latitude, longitude) tuples for sample points
    """
    from shapely.geometry import Point
    import numpy as np
    
    # Convert MultiPolygon to single polygon if needed
    try:
        if hasattr(basin_geometry, 'geoms'):
            # It's a MultiPolygon, find the largest polygon
            geoms = list(basin_geometry.geoms)
            idx = np.argmax([g.area for g in geoms])
            geometry = geoms[idx]
        else:
            geometry = basin_geometry
    except Exception as e:
        # If any error, just use the original geometry
        geometry = basin_geometry
    
    # Get the bounds of the geometry (minx, miny, maxx, maxy)
    minx, miny, maxx, maxy = geometry.bounds
    
    # Generate points and keep those within the basin
    sample_points = []
    
    # Always include the centroid as one point
    centroid = geometry.centroid
    sample_points.append((centroid.y, centroid.x))  # lat, lon
    
    # Special case - if we only need 1 point, return just the centroid
    if num_points <= 1:
        return sample_points
    
    # We need more points, so keep generating random points until we have enough
    attempts = 0
    max_attempts = num_points * 10  # Limit attempts to avoid infinite loop
    
    while len(sample_points) < num_points and attempts < max_attempts:
        # Generate a random point within the bounding box
        x = minx + (maxx - minx) * np.random.random()
        y = miny + (maxy - miny) * np.random.random()
        point = Point(x, y)
        
        # Check if the point is within the basin
        if geometry.contains(point):
            # Check if it's not too close to existing points (basic spacing)
            is_far_enough = True
            for existing_point in sample_points:
                # Approximate distance check (could improve with proper geodesic distance)
                dist = ((existing_point[1] - x)**2 + (existing_point[0] - y)**2)**0.5
                if dist < (maxx - minx) / (num_points**0.5 * 2):  # Simple spacing heuristic
                    is_far_enough = False
                    break
                    
            if is_far_enough:
                sample_points.append((y, x))  # lat, lon
        
        attempts += 1
    
    # If we couldn't get enough points, we'll work with what we have
    return sample_points

def get_daily_rainfall_forecast_multi(basin_geometry, basin_utm=None, model="ecmwf_ifs025"):
    """
    Get daily rainfall forecast for multiple points in a watershed based on its size.
    
    Args:
        basin_geometry: The watershed geometry in WGS84 (EPSG:4326) for point sampling
        basin_utm: Optional basin GeoDataFrame in UTM projection for accurate area calculation
        model: Weather forecast model to use
        
    Returns:
        Dictionary with averaged forecast data
    """
    # Calculate basin area in km²
    if basin_utm is not None:
        # Use the provided UTM basin for accurate area calculation
        area_m2 = basin_utm.to_crs('epsg:26910').geometry[0].area
        area_km2 = area_m2 / 1000000
    else:
        # Fallback to rough estimate if UTM basin not provided
        st.warning("UTM basin not provided. Using approximate area calculation.")
        area_km2 = 200  # Default to 200 km²
    
    # Determine number of sample points based on watershed size
    if area_km2 < 40:  # Small watershed
        num_points = 1
    elif area_km2 < 200:  # Medium watershed
        num_points = 4  # Increased from 3
    elif area_km2 < 750:  # Large watershed
        num_points = 6  # Increased from 5
    elif area_km2 < 1500:  # Very large watershed
        num_points = 9  # Increased from 7
    else:  # Extremely large watershed - this will be caught in the main function
        num_points = 12  # Increased from 10
    
    # Generate sample points
    sample_points = generate_sample_points(basin_geometry, num_points)
    
    # Always store the watershed info in session state
    st.session_state.watershed_info_message = f"Watershed area: {area_km2:.1f} km² - Using {len(sample_points)} sampling points for rainfall forecast"
    
    # Collect forecasts for all points
    all_forecasts = []
    for i, (lat, lon) in enumerate(sample_points):
        try:
            forecast = get_daily_rainfall_forecast(lat, lon, model)
            all_forecasts.append(forecast)
        except Exception as e:
            st.warning(f"Error fetching forecast for point {i+1}: {e}")
    
    # We must have at least one successful forecast
    if not all_forecasts:
        raise Exception("Failed to get any rainfall forecasts")
    
    # Combine the forecasts (average the precipitation across all points)
    # Start with the first forecast as a template
    combined_forecast = all_forecasts[0].copy()
    
    # If we have multiple forecasts, average the precipitation values
    if len(all_forecasts) > 1:
        # Get the precipitation data from all forecasts
        all_precip = []
        for f in all_forecasts:
            all_precip.append(f['daily']['precipitation_sum'])
        
        # Calculate average precipitation for each day
        avg_precip = []
        for i in range(len(all_precip[0])):
            day_values = [precip[i] for precip in all_precip if i < len(precip)]
            avg_precip.append(sum(day_values) / len(day_values))
        
        # Replace the precipitation values in the combined forecast
        combined_forecast['daily']['precipitation_sum'] = avg_precip
    
    return combined_forecast

def get_historical_rainfall_data(latitude, longitude, start_date, end_date):
    """
    Get historical daily rainfall data using Open-Meteo API
    
    Args:
        latitude (float): Location latitude
        longitude (float): Location longitude
        start_date (str): Start date in format YYYY-MM-DD
        end_date (str): End date in format YYYY-MM-DD
        
    Returns:
        pandas.DataFrame: DataFrame with daily precipitation data
    """
    base_url = "https://archive-api.open-meteo.com/v1/archive"
    
    params = {
        "latitude": latitude,
        "longitude": longitude,
        "start_date": start_date,
        "end_date": end_date,
        "daily": "precipitation_sum",
        "timezone": "auto"
    }
    
    try:
        response = requests.get(base_url, params=params)
        
        if response.status_code == 200:
            data = response.json()
            
            # Create DataFrame from the response
            df = pd.DataFrame({
                "date": pd.to_datetime(data["daily"]["time"]),
                "precipitation_mm": data["daily"]["precipitation_sum"]
            })
            
            # Set date as index
            df.set_index("date", inplace=True)
            
            return df
        else:
            st.warning(f"Error retrieving historical rainfall data: {response.status_code}")
            return None
    except Exception as e:
        st.warning(f"Error retrieving historical rainfall data: {str(e)}")
        return None

def get_daily_rainfall_forecast(latitude, longitude, model="ecmwf_ifs025"):
    """
    Get daily rainfall forecast using Open-Meteo API for a single point.
    
    Models:
    - ecmwf_ifs025: ECMWF IFS-HRES (European Centre for Medium-Range Weather Forecasts)
    - gfs_seamless: GFS (Global Forecast System from NOAA)
    - icon_seamless: ICON (German Weather Service/DWD)
    """
    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": latitude,
        "longitude": longitude,
        "daily": "precipitation_sum",
        "timezone": "America/Los_Angeles",
        "models": [model]  # Correct format: "models" as a list
    }
    try:
        response = requests.get(url, params=params)
        response.raise_for_status()  # Raise exception for HTTP errors
        data = response.json()
        return data
    except requests.RequestException as e:
        # If the selected model fails, fall back to default model
        if model != "ecmwf_ifs025":
            st.warning(f"Error with {model} model: {str(e)}. Falling back to ECMWF model.")
            return get_daily_rainfall_forecast(latitude, longitude, "ecmwf_ifs025")
        else:
            # If even the default model fails, raise the exception
            raise e

@cache_data
def get_historical_rainfall(latitude, longitude, start_date, end_date, timezone="auto"):
    """
    Get historical daily rainfall data using Open-Meteo API
    
    Args:
        latitude (float): Location latitude
        longitude (float): Location longitude
        start_date (str): Start date in format YYYY-MM-DD
        end_date (str): End date in format YYYY-MM-DD
        timezone (str): Timezone name (default: auto)
        
    Returns:
        pandas.DataFrame: DataFrame with daily precipitation data
    """
    base_url = "https://archive-api.open-meteo.com/v1/archive"
    
    params = {
        "latitude": latitude,
        "longitude": longitude,
        "start_date": start_date,
        "end_date": end_date,
        "daily": "precipitation_sum",
        "timezone": timezone
    }
    
    try:
        response = requests.get(base_url, params=params)
        
        if response.status_code == 200:
            data = response.json()
            
            # Create DataFrame from the response
            df = pd.DataFrame({
                "date": pd.to_datetime(data["daily"]["time"]),
                "precipitation_mm": data["daily"]["precipitation_sum"]
            })
            
            # Set date as index
            df.set_index("date", inplace=True)
            
            return df
        else:
            st.warning(f"Error retrieving historical rainfall data: {response.status_code}")
            return None
    except Exception as e:
        st.warning(f"Error retrieving historical rainfall data: {str(e)}")
        return None

# Function to create a download link for a dataframe
def get_csv_download_link(df, filename, link_text):
    """Generates a link to download the dataframe as a CSV file"""
    csv = df.to_csv(index=True)
    b64 = base64.b64encode(csv.encode()).decode()  # Encode to base64
    href = f'<a href="data:file/csv;base64,{b64}" download="{filename}">{link_text}</a>'
    return href

# Function to create a download link for binary files (like images)
def get_binary_download_link(file_path, filename, link_text, mime_type):
    """Generates a link to download a binary file"""
    with open(file_path, "rb") as f:
        data = f.read()
    b64 = base64.b64encode(data).decode()
    href = f'<a href="data:{mime_type};base64,{b64}" download="{filename}">{link_text}</a>'
    return href

def main():
    st.title('Streamflow Projection App')
    st.markdown('<small><a href="https://github.com/daviddralle/flow_projector/blob/main/README.md" target="_blank">More details in the GitHub README</a></small>', unsafe_allow_html=True)
    
    # Put all model information in a single collapsible expander
    with st.expander("⚠️ Model Caveats and Details", expanded=False):
        st.markdown("""
        <div style="font-size:0.9em;">
            <p><strong>Important model limitations:</strong></p>
            <ul>
                <li>All flow values represent <strong>DAILY averages</strong>, not instantaneous flow. These values should not be directly compared to instantaneous values posted in CNRFC forecasts.</li>
                <li>This is a simple <strong>storage-discharge approach for rain-dominated watersheds</strong>. It assumes rainfall directly recharges hillslope groundwater tables that feed the stream.</li>
                <li>There is no accounting for <strong>vadose zone storage deficits</strong> that might result in less recharge.</li>
                <li>This model <strong>does not handle snow accumulation or melt processes</strong> appropriately and should be used with caution in snow-affected watersheds.</li>
                <li>The app was designed primarily for forecasting in <strong>Coastal California during the spring and early summer</strong>, when vadose zone deficits are small and most rainfall is converted to groundwater recharge (versus being stored in the unsaturated zone).</li>
                <li>Forecasts account for rain over a 7-day forecast period, after which the model assumes there is no rain and flows recede according to the underlying storage-discharge relationship (calculated behind the scenes from historical flow data).</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    st.sidebar.header('Input Parameters')
    
    # Initialize session state to store data between reruns
    if 'has_run_analysis' not in st.session_state:
        st.session_state.has_run_analysis = False
    if 'historical_data' not in st.session_state:
        st.session_state.historical_data = None
    if 'projected_data' not in st.session_state:
        st.session_state.projected_data = None
    if 'site_id' not in st.session_state:
        st.session_state.site_id = None
    if 'T_days' not in st.session_state:
        st.session_state.T_days = None
    if 'selected_model' not in st.session_state:
        st.session_state.selected_model = "ecmwf_ifs025"  # Default model
    if 'selected_model_name' not in st.session_state:
        st.session_state.selected_model_name = "ECMWF (European Centre for Medium-Range Weather Forecasts)"
    # Initialize message storage
    if 'flow_combination_message' not in st.session_state:
        st.session_state.flow_combination_message = None
    if 'watershed_info_message' not in st.session_state:
        st.session_state.watershed_info_message = None
    if 'plot_dates' not in st.session_state:
        st.session_state.plot_dates = {
            'stop_datetime': None,
            'future_stop_datetime': None
        }
        
    # Define commonly used USGS gages with descriptive names
    predefined_gages = {
        "Elder Creek nr Branscomb": "11475560",
        "SF Eel at Leggett": "11475800",
        "SF Eel at Miranda": "11476500",
        "MF Eel at Dos Rios": "11473900",
        "Van Duzen nr Bridgeville": "11478500",
        "Mad River nr Arcata": "11481000",
        "Custom/Other Gage...": "custom"
    }
    
    # Initialize gage selection in session state if it doesn't exist
    if 'gage_selection' not in st.session_state:
        st.session_state.gage_selection = list(predefined_gages.keys())[1]  # Default to SF Eel at Leggett
    if 'custom_gage' not in st.session_state:
        st.session_state.custom_gage = ""
    if 'previous_gage' not in st.session_state:
        st.session_state.previous_gage = st.session_state.gage_selection
    
    # Callback for gage selection changes
    def on_gage_change():
        # Reset analysis if gage changed
        if st.session_state.gage_dropdown != st.session_state.previous_gage:
            # The selection has changed
            st.session_state.previous_gage = st.session_state.gage_dropdown
            st.session_state.gage_selection = st.session_state.gage_dropdown
            # Reset analysis state
            if st.session_state.has_run_analysis:
                st.session_state.has_run_analysis = False
                st.session_state.trigger_rerun = True
    
    # Gage selection UI
    st.sidebar.subheader("Gage Selection")
    gage_selection = st.sidebar.selectbox(
        "Select a USGS Gage",
        options=list(predefined_gages.keys()),
        index=list(predefined_gages.keys()).index(st.session_state.gage_selection),
        key="gage_dropdown",
        on_change=on_gage_change
    )
    
    # Show custom gage input if "Custom" is selected
    if st.session_state.gage_dropdown == "Custom/Other Gage...":
        # Define callback for custom gage input changes
        def on_custom_gage_change():
            # Reset analysis if custom gage ID changed
            if st.session_state.has_run_analysis:
                st.session_state.has_run_analysis = False
                st.session_state.trigger_rerun = True
                
        custom_gage = st.sidebar.text_input(
            "Enter USGS Gage ID", 
            value=st.session_state.custom_gage if st.session_state.custom_gage else "",
            placeholder="e.g., 11475800",
            key="custom_gage_input",
            on_change=on_custom_gage_change
        )
        st.session_state.custom_gage = custom_gage
        site = custom_gage if custom_gage.strip() else "11475800"  # Default if empty
    else:
        site = predefined_gages[st.session_state.gage_dropdown]
        
    # Projection period selection
    T = st.sidebar.number_input('Projection Period (days)', min_value=7, max_value=90, value=60)
    
    # Weather model selection
    model_mapping = {
        "ECMWF (European Centre for Medium-Range Weather Forecasts)": "ecmwf_ifs025",
        "GFS (Global Forecast System - NOAA)": "gfs_seamless",
        "DWD (German Weather Service)": "icon_seamless",
        "No model (zero rainfall)": "no_model"
    }
    
    # Get the index of the previously selected model
    model_options = list(model_mapping.keys())
    default_index = model_options.index(st.session_state.selected_model_name) if st.session_state.selected_model_name in model_options else 0
    
    st.sidebar.subheader("Weather Model Selection")
    
    # Initialize trigger_rerun if needed
    if 'trigger_rerun' not in st.session_state:
        st.session_state.trigger_rerun = False
        
    # Check if we need to rerun based on previous callback
    if st.session_state.trigger_rerun:
        st.session_state.trigger_rerun = False
        st.rerun()
    
    # Use a callback to handle model changes
    def on_model_change():
        model_name = st.session_state.model_selector
        model_code = model_mapping[model_name]
        st.session_state.selected_model = model_code
        st.session_state.selected_model_name = model_name
        if st.session_state.has_run_analysis:
            st.session_state.has_run_analysis = False
            # Don't call st.rerun() inside callback
            st.session_state.trigger_rerun = True
    
    # Instead of setting both default index and session state,
    # just use a standard selectbox with the callback
    current_model = st.session_state.selected_model_name
    
    st.sidebar.selectbox(
        "Select Weather Forecast Model",
        model_options,
        index=model_options.index(current_model) if current_model in model_options else 0,
        help="Choose the weather forecast model for rainfall predictions",
        key="model_selector",
        on_change=on_model_change
    )
    
    # Get the current selected model
    selected_model = model_mapping[st.session_state.model_selector]
    
    def run_analysis():
        st.session_state.has_run_analysis = True
        st.session_state.site_id = site
        st.session_state.T_days = T
        # Save the current model selection
        st.session_state.selected_model = selected_model
        st.session_state.selected_model_name = st.session_state.model_selector
        # Reset message storage on new analysis
        st.session_state.flow_combination_message = None
        st.session_state.watershed_info_message = None
    
    if st.sidebar.button('Run Analysis', on_click=run_analysis) or st.session_state.has_run_analysis:
        # Date range for getting historical data
        now = pd.to_datetime(datetime.datetime.now(LA_TIMEZONE).strftime('%Y-%m-%d'))
        # Use 10 years of historical data instead of just 1 year
        start = now - pd.to_timedelta(10*365, unit='d')
        stop = now
        future_stop = now + pd.to_timedelta(T, unit='d')
        start_str = start.strftime('%Y-%m-%d')
        stop_str = stop.strftime('%Y-%m-%d')
        stop_datetime = pd.to_datetime(stop_str).tz_localize(LA_TIMEZONE)
        future_stop_str = (future_stop - pd.to_timedelta(1, unit='d')).strftime('%Y-%m-%d')
        future_stop_datetime = pd.to_datetime(future_stop_str).tz_localize(LA_TIMEZONE)

        with st.spinner(f"Retrieving data for gauge {site}..."):
            # Grab data
            df, basin = getFlow(site, start_str, stop_str)
            
            if df is not None and len(df) > 0:
                # Process data based on matching days of year
                QS = []
                DQS = []
                
                # Get days of year for projection period
                projection_days = pd.date_range(stop_datetime, future_stop_datetime)
                projection_doys = [d.dayofyear for d in projection_days]
                
                # Create mask for all data points that match projection period days of year
                mask = df.index.dayofyear.isin(projection_doys)
                seasonal_data = df[mask]
                
                # Get basin centroid coordinates
                geo = basin.to_crs('epsg:4326').geometry.values[0].centroid
                centroid_lat = geo.y
                centroid_lon = geo.x
                
                # Get historical rainfall data at basin centroid
                with st.spinner("Retrieving historical rainfall data for basin centroid..."):
                    rainfall_df = get_historical_rainfall(
                        latitude=centroid_lat,
                        longitude=centroid_lon,
                        start_date=start_str,
                        end_date=stop_str,
                        timezone=LA_TIMEZONE.zone
                    )
                
                if len(seasonal_data) > 0:
                    # Filter out days with rainfall if rainfall data is available
                    if rainfall_df is not None:
                        # Convert seasonal_data index to date only for matching with rainfall_df
                        seasonal_dates = seasonal_data.index.date
                        
                        # Create a mask for days without rainfall (< 1mm)
                        no_rain_mask = []
                        for date in seasonal_dates:
                            date_str = pd.Timestamp(date)
                            if date_str in rainfall_df.index:
                                # If rainfall is less than 1mm, consider it a dry day
                                no_rain_mask.append(rainfall_df.loc[date_str, 'precipitation_mm'] < 1.0)
                            else:
                                # If no rainfall data for this date, assume it's a dry day
                                no_rain_mask.append(True)
                        
                        # Apply the no-rain mask to the seasonal data
                        dry_seasonal_data = seasonal_data[no_rain_mask]
                        # st.info(f"Filtered out {len(seasonal_data) - len(dry_seasonal_data)} days with rainfall > 1mm from recession analysis")
                        
                        # Use the dry seasonal data for recession analysis
                        qs = dry_seasonal_data[site].values
                    else:
                        # If rainfall data retrieval failed, use all seasonal data
                        st.warning("Could not retrieve rainfall data. Using all seasonal data for recession analysis.")
                        qs = seasonal_data[site].values
                    
                    # Calculate gradients on the full time series first
                    all_qs = df[site].values
                    all_dqs = np.gradient(all_qs,86400)
                    
                    # Map gradients back to the seasonal data
                    if rainfall_df is not None and len(no_rain_mask) > 0:
                        # For dry days only
                        dry_seasonal_indices = np.where(mask)[0][no_rain_mask]
                        dqs = all_dqs[dry_seasonal_indices]
                    else:
                        # For all seasonal days
                        seasonal_indices = np.where(mask)[0]
                        dqs = all_dqs[seasonal_indices]
                    
                    # Find recessions (when flow is decreasing and positive)
                    idx = (dqs < 0) & (qs > 0)
                    QS = qs[idx]
                    DQS = dqs[idx]
                else:
                    st.warning("Not enough historical data matching the projection period's days of year. Using all available data instead.")
                    
                    # Filter out days with rainfall if rainfall data is available
                    if rainfall_df is not None:
                        # Convert df index to date only for matching with rainfall_df
                        all_dates = df.index.date
                        
                        # Create a mask for days without rainfall (< 1mm)
                        no_rain_mask = []
                        for date in all_dates:
                            date_str = pd.Timestamp(date)
                            if date_str in rainfall_df.index:
                                # If rainfall is less than 1mm, consider it a dry day
                                no_rain_mask.append(rainfall_df.loc[date_str, 'precipitation_mm'] < 1.0)
                            else:
                                # If no rainfall data for this date, assume it's a dry day
                                no_rain_mask.append(True)
                        
                        # Apply the no-rain mask to all data
                        dry_df = df[no_rain_mask]
                        st.info(f"Filtered out {len(df) - len(dry_df)} days with rainfall > 1mm from recession analysis")
                        
                        # Use the dry data for recession analysis
                        qs = dry_df[site].values
                    else:
                        # If rainfall data retrieval failed, use all data
                        st.warning("Could not retrieve rainfall data. Using all data for recession analysis.")
                        qs = df[site].values
                    
                    dqs = np.gradient(qs,86400)
                    idx = (dqs < 0) & (qs > 0)
                    QS = qs[idx]
                    DQS = dqs[idx]
                
                # Apply binning approach for more sophisticated fitting
                # Convert to numpy arrays
                QS = np.array(QS)
                DQS = np.array(DQS)
                
                # Create dataframe for binning
                df_kirchner = pd.DataFrame({'q':QS, 'dq':DQS})
                df_kirchner = df_kirchner.dropna()
                df_kirchner = df_kirchner.sort_values('q',ascending=False)
                
                # Set binning parameters
                num_bins = 30  # Default number of bins
                min_per_bin = 10  # Minimum points per bin
                
                # Get bin boundaries
                binBoundaries = predefined_binning_indices(df_kirchner, num_bins=num_bins, min_per_bin=min_per_bin)
                
                # Calculate mean flow and mean flow derivative for each bin
                qs = [np.mean(df_kirchner.q[binBoundaries[i]:binBoundaries[i+1]]) for i in range(len(binBoundaries)-1)]
                dqs = np.array([np.mean(df_kirchner.dq[binBoundaries[i]:binBoundaries[i+1]]) for i in range(len(binBoundaries)-1)])
                
                # Calculate standard errors for each bin
                sigmas = []
                for i in range(len(binBoundaries)-1):
                    # Get data for this bin
                    bin_data = df_kirchner.dq[binBoundaries[i]:binBoundaries[i+1]]
                    # Filter for negative dq values
                    neg_dq = bin_data.loc[bin_data < 0]
                    if len(neg_dq) > 1:  # Need at least 2 points to calculate std
                        # Calculate standard error: std/sqrt(n)
                        bin_se = np.std(np.log(-neg_dq)) / np.sqrt(len(neg_dq))
                        sigmas.append(bin_se)
                    else:
                        # If not enough points, use a small default value
                        sigmas.append(1e-2)
                
                # Convert to numpy array and add small constant to avoid division by zero
                sigmas = np.array(sigmas) + 1e-12
                
                # Convert to numpy arrays
                SS = np.array(sigmas)
                QS = np.array(qs)
                DQS = np.array(dqs)
                
                # Filter data
                idx = (DQS<0)&(np.isfinite(QS)) & (np.isfinite(DQS))
                QS = QS[idx]
                DQS = DQS[idx]
                SS = SS[idx]

                # now perform fit with Adam's function
                def bq(q,bl,bu):
                    return bl + (bu - bl)*0.5*(1+scipy.special.erf( (np.log(q) - logqbar)/(logsigma*np.sqrt(2))))
                logqbar = np.log(np.mean(QS))
                logsigma = np.log(np.std(QS))
                def eps(q,bl,bu,a):
                    qbar = np.exp(logqbar)
                    vals = a*(q/qbar)**bq(q,bl,bu)
                    return vals
                
                def logeps(q,bl,bu,a):
                    qbar = np.exp(logqbar)
                    vals = np.log(a) + bq(q,bl,bu)*np.log(q/qbar)
                    return vals
                    
                # new sensitivity function
                def newg(q,popt):
                    return eps(q,*popt)/q

                # Use standard errors in curve fitting with error handling
                try:
                    # First try with sigma weighting
                    popt, pcov = curve_fit(logeps, QS, np.log(-DQS), sigma=SS)
                except Exception as e:
                    st.warning(f"Curve fitting with sigma weighting failed: {e}. Falling back to unweighted fit.")
                    # Fall back to standard curve_fit without sigma weighting
                    popt, pcov = curve_fit(logeps, QS, np.log(-DQS))
                
                # Create and save dQ/dt vs Q plot with fit
                freq = site  # Use site ID as frequency identifier
                f, ax = plt.subplots(figsize=(5, 4))
                
                # Plot the raw scatter points with low alpha for context
                ax.scatter(np.log(QS), np.log(-DQS), ec='k', c='dodgerblue', alpha=0.1)
                
                # Plot the binned data with error bars
                log_qs = np.log(np.array(qs))
                log_dqs = np.log(-np.array(dqs))
                
                # Convert standard errors to log space
                log_sigmas = SS.copy()
                
                # Plot binned points with error bars
                ax.errorbar(log_qs, log_dqs, yerr=log_sigmas, fmt='o', 
                           color='dodgerblue', ecolor='black', capsize=3, 
                           markersize=5, markeredgecolor='black', zorder=10)
                
                # Save raw data to CSV in plots folder
                data_filename = f'data_{freq}.csv'
                data_path = os.path.join(PLOTS_DIR, data_filename)
                pd.DataFrame({'x': np.log(QS), 'y': np.log(-DQS)}).to_csv(data_path)
                
                # Save binned data to CSV in plots folder
                binned_data_filename = f'binned_data_{freq}.csv'
                binned_data_path = os.path.join(PLOTS_DIR, binned_data_filename)
                pd.DataFrame({
                    'log_q': log_qs, 
                    'log_dq': log_dqs,
                    'se': log_sigmas
                }).to_csv(binned_data_path)
                
                # Sort x values for smooth line plot
                x = np.log(QS)
                x_sorted = np.sort(x)
                
                # Calculate fit values
                y = np.log(eps(np.exp(x_sorted), *popt))
                
                # Save fit data to CSV in plots folder
                fit_data_filename = f'fit_{freq}.csv'
                fit_data_path = os.path.join(PLOTS_DIR, fit_data_filename)
                pd.DataFrame({'x': x_sorted, 'y': y}).to_csv(fit_data_path)
                
                # Plot the fit line
                ax.plot(x_sorted, y, c='k', lw=0.75, label='Dynamic power law (Wlostowski et al)')
                
                # Set labels and title
                ax.set_xlabel('ln(Q)')
                ax.set_ylabel(r'$\ln \left( -\mathrm{\frac{dQ}{dt}}\right)$')
                ax.set_title(f'USGS {site}', fontsize=12)
                
                # Add legend
                leg = ax.legend(fontsize=10, loc='best')
                leg.set_frame_on(False)
                
                # Layout and style
                f.tight_layout()
                sns.despine()
                
                # Save the figure to plots directory
                plot_filename = f'dqdt_vs_q_{freq}.png'
                plot_path = os.path.join(PLOTS_DIR, plot_filename)
                f.savefig(plot_path, dpi=300, bbox_inches='tight')
                plt.close(f)
                
                # Create HTML download link for the plot
                plot_link = get_binary_download_link(
                    file_path=plot_path,
                    filename=plot_filename,
                    link_text="Download dQ/dt vs Q Plot",
                    mime_type="image/png"
                )
                
                # Display the download link
                st.markdown(plot_link, unsafe_allow_html=True)
                geo = basin.to_crs('epsg:4326').geometry.values[0].centroid
                area_ft2 = basin.to_crs('epsg:26910').geometry.values[0].area*10.7639104167
                lat,lon = geo.y,geo.x
                
                # Get watershed geometry in WGS84 for rainfall sampling
                basin_wgs84 = basin.to_crs('epsg:4326')
                
                # Convert basin to UTM for accurate area calculation
                basin_utm = basin.to_crs('epsg:26910')
                
                # Calculate watershed area in km²
                area_m2 = basin_utm.geometry.values[0].area
                area_km2 = area_m2 / 1000000
                
                # Check if watershed area is too large for this model
                if area_km2 > 1500:
                    st.error(f"Watershed area ({area_km2:.1f} km²) exceeds the 1500 km² limit for this model. This model does not implement flow routing required for larger watersheds at the daily timestep. Please select a different gage with a watershed area < 1500 km²")
                    st.session_state.has_run_analysis = False
                    return
                
                # Get forecast from multiple points based on watershed size or use zero rainfall if 'no_model' is selected
                if st.session_state.selected_model == "no_model":
                    # Create a zero-rainfall forecast for the projection period
                    current_date = datetime.datetime.now(LA_TIMEZONE).date()
                    forecast_dates = [current_date + datetime.timedelta(days=i) for i in range(T)]
                    forecast_dates_str = [date.strftime("%Y-%m-%d") for date in forecast_dates]
                    forecast = {
                        'daily': {
                            'time': forecast_dates_str,
                            'precipitation_sum': [0.0] * len(forecast_dates_str)
                        }
                    }
                    st.sidebar.success(f"Using {st.session_state.selected_model_name} (assuming zero rainfall)")
                else:
                    # Pass both the WGS84 geometry for sampling points and the UTM basin for area calculation
                    forecast = get_daily_rainfall_forecast_multi(
                        basin_geometry=basin_wgs84.geometry.values[0],
                        basin_utm=basin_utm,
                        model=st.session_state.selected_model
                    )
                    st.sidebar.success(f"Using {st.session_state.selected_model_name} for rainfall forecasts")
                time = forecast['daily']['time']
                precipitation_sum = forecast['daily']['precipitation_sum']
                df_forecast = pd.DataFrame({'ppt': precipitation_sum}, index=pd.to_datetime(time).tz_localize(LA_TIMEZONE))
          
                # Get today's date in LA timezone for comparison
                today = datetime.datetime.now(LA_TIMEZONE).date()
                
                # Check if today is in the forecast dates
                today_in_forecast = False
                forecast_start_idx = 0
                
                # Identify today's position in the forecast data
                for i, date in enumerate(pd.to_datetime(df_forecast.index.date)):
                    if date.date() == today:
                        today_in_forecast = True
                        forecast_start_idx = i
                        break
                
                if not today_in_forecast:
                    forecast_start_idx = 0
                    
                # Create a DataFrame for display later if needed
                forecast_display = pd.DataFrame({
                    'Date': pd.to_datetime(df_forecast.index).date,
                    'Rainfall (mm)': df_forecast['ppt'].values
                })
                
                # Create time array for integration and rainfall values
                forecast_times = np.array(range(forecast_start_idx, len(df_forecast)))
                adjusted_forecast_times = forecast_times - forecast_start_idx
                
                # Define t for integration
                t = np.linspace(0, T-1, T)*86400
                
                # Initialize rainfall values array
                rain_vals = np.zeros_like(t)
                
                # Convert forecast mm/day into ft3/s increments
                # Use only the forecast from today onwards
                forecast_rain = df_forecast['ppt'].values[forecast_start_idx:]
                rain_vals[adjusted_forecast_times] = 3.79727e-8 * area_ft2 * forecast_rain
                
                # Save integration info for displaying later
                integration_info = {
                    'start_date': today,
                    'initial_flow': df[site].values[-1],
                    'forecast_start': forecast_display['Date'].iloc[forecast_start_idx] if len(forecast_rain) > 0 else None,
                    'forecast_start_val': forecast_rain[0] if len(forecast_rain) > 0 else 0,
                    'forecast_end': forecast_display['Date'].iloc[-1] if len(forecast_rain) > 0 else None,
                    'forecast_end_val': forecast_rain[-1] if len(forecast_rain) > 0 else 0
                }
                
                # Create forcing function for integration
                forcing = interp1d(t, rain_vals, kind='previous',fill_value='extrapolate')
                
                def fun(time, q):
                    return -newg(q, popt)*(q - forcing(time))
                
                # Use the latest flow value as initial condition
                q0 = df[site].values[-1]
                
                sol = solve_ivp(fun, [0, t[-1]], [q0], rtol=1e-5)
                sol_int = interp1d(sol.t, sol.y[0], fill_value=0, bounds_error=False)
                natQ = sol_int(t)
                
                # Create projection dataframe starting from today
                today_datetime = pd.Timestamp(today).tz_localize(LA_TIMEZONE)
                projection_end = today_datetime + pd.to_timedelta(T-1, unit='d')
                
                idx = pd.date_range(
                    today_datetime,
                    projection_end,
                    freq='D',
                    tz=LA_TIMEZONE
                )
                
                # Create the projection dataframe
                natQ_df = pd.DataFrame({'Flow projection': natQ}, index=idx)
                
                # Interactive Plotly plot
                st.header('Daily average (not instantaneous) flow forecast')
                
                # Create the figure
                fig = go.Figure()
                
                # Add historical flow data
                fig.add_trace(go.Scatter(
                    x=df.index,
                    y=df[site],
                    mode='lines',
                    name='Historical Flow',
                    line=dict(color='black'),
                    hovertemplate='<b>Date</b>: %{x}<br><b>Flow</b>: %{y:.2f} cfs<extra></extra>'
                ))
                
                # Add projection
                fig.add_trace(go.Scatter(
                    x=natQ_df.index,
                    y=natQ_df['Flow projection'],
                    mode='lines',
                    name='Projected Flow',
                    line=dict(color='red'),
                    hovertemplate='<b>Date</b>: %{x}<br><b>Projected Flow</b>: %{y:.2f} cfs<extra></extra>'
                ))
                
                # No longer needed - we'll use the star marker instead
                
                # Try another approach for the projection start point - add a clearer marker
                start_date_str = pd.to_datetime(natQ_df.index[0]).strftime('%Y-%m-%d')
                start_flow = float(natQ_df['Flow projection'].values[0])
                
                fig.add_trace(go.Scatter(
                    x=[natQ_df.index[0]],
                    y=[start_flow],
                    mode='markers',
                    marker=dict(symbol='star', color='yellow', size=10, line=dict(color='black', width=2)),
                    name='Projection Start Point',
                    hoverinfo='text',
                    text=f"Projection Start: {start_date_str}<br>Flow: {start_flow:.2f} cfs"
                ))
                
                # Add rainfall forecast to the plot on secondary y-axis
                rainfall_dates = pd.to_datetime(forecast['daily']['time'])
                
                # Create color array to highlight today's rainfall
                bar_colors = ['rgba(0, 100, 255, 0.6)'] * len(rainfall_dates)  # Default blue
                if today_in_forecast:
                    # Highlight today's rainfall with a different color
                    bar_colors[forecast_start_idx] = 'rgba(255, 140, 0, 0.8)'  # Orange for today
                
                # Set the default blue color used for rainfall bars
                default_blue_color = 'rgba(0, 100, 255, 0.6)'  # Default blue for rainfall
                
                # Create bar chart for rainfall with custom colors
                fig.add_trace(go.Bar(
                    x=rainfall_dates,
                    y=df_forecast['ppt'],
                    name='Rainfall Forecast',
                    marker_color=bar_colors,
                    marker=dict(color=bar_colors, line=dict(color=bar_colors)),  # Removed colorbar
                    yaxis='y2',  # Use secondary y-axis
                    hovertemplate='<b>Date</b>: %{x}<br><b>Rainfall</b>: %{y:.1f} mm<extra></extra>',
                    showlegend=False  # Hide this from legend initially
                ))
                
                # Add a separate visible trace for legend only
                fig.add_trace(go.Bar(
                    x=[None],
                    y=[None],
                    name='Rainfall Forecast',
                    marker=dict(color=default_blue_color),
                    showlegend=True,
                    hoverinfo='skip'
                ))
                
                # Add legend items for the shaded regions as colored squares
                # 1. Rainfall forecast period (green)
                rainfall_color = 'rgba(200, 255, 200, 0.8)'  # Light green, more opaque for legend visibility
                fig.add_trace(go.Scatter(
                    x=[None],
                    y=[None],
                    mode='markers',
                    marker=dict(
                        size=15,
                        color=rainfall_color,
                        symbol='square',
                        line=dict(width=1, color='rgba(0, 0, 0, 0.3)')
                    ),
                    name='Rainfall Forecast Period',
                    showlegend=True,
                    hoverinfo='skip'
                ))
                
                # 2. Recession forecast period (blue)
                recession_color = 'rgba(200, 230, 255, 0.8)'  # Light blue, more opaque for legend visibility
                fig.add_trace(go.Scatter(
                    x=[None],
                    y=[None],
                    mode='markers',
                    marker=dict(
                        size=15,
                        color=recession_color,
                        symbol='square',
                        line=dict(width=1, color='rgba(0, 0, 0, 0.3)')
                    ),
                    name='Recession Forecast Period',
                    showlegend=True,
                    hoverinfo='skip'
                ))
                
                # Add annotation for today's rainfall if it exists
                if today_in_forecast and df_forecast['ppt'].values[forecast_start_idx] > 0:
                    today_rainfall = df_forecast['ppt'].values[forecast_start_idx]
                    fig.add_annotation(
                        x=rainfall_dates[forecast_start_idx],
                        y=today_rainfall,
                        text=f"Today: {today_rainfall:.1f} mm",
                        showarrow=True,
                        arrowhead=1,
                        ax=0,
                        ay=-40,
                        yref="y2"
                    )
                
                # Get the rainfall forecast date range for the title
                forecast_start_date = rainfall_dates[0].strftime('%b %d, %Y')
                forecast_end_date = rainfall_dates[-1].strftime('%b %d, %Y')
                
                # Set up the layout with log scale for primary y-axis and secondary y-axis for rainfall
                # Now display the actual plot
                fig.update_layout(
                    title={
                        'text': f"Flow Projection for USGS Gage {site}<br>Starting {today}",
                        'y':0.98,
                        'x':0.5,
                        'xanchor': 'center',
                        'yanchor': 'top'
                    },
                    xaxis_title='Date',
                    yaxis=dict(
                        title='Flow (cfs)',
                        type='log',
                        side='left'
                    ),
                    yaxis2=dict(
                        title='Rainfall (mm)',
                        side='right',
                        overlaying='y',
                        showgrid=False,
                        range=[0, max(df_forecast['ppt'].max() * 1.1, 0.1)],  # Ensure proper y-axis range with padding
                        zeroline=True,  # Show the zero line
                        showline=True,  # Show the axis line
                        showticklabels=True,  # Make sure tick labels are shown
                        automargin=True,  # Ensure margin for labels
                        dtick=5  # Set tick interval to 5mm
                    ),
                    xaxis=dict(range=[stop_datetime - pd.to_timedelta(60, unit='d'), future_stop_datetime]),
                    hovermode='closest',

                    # Increase margins, especially top margin to prevent title overlap with buttons
                    margin=dict(l=50, r=60, t=80, b=80),
                    height=570,
                    # Improve legend layout
                    legend=dict(
                        orientation='h',  # Horizontal layout
                        yanchor='top',
                        y=-0.15,  # Position below the chart
                        xanchor='center',
                        x=0.5,
                        bgcolor='rgba(255, 255, 255, 0.8)',  # Semi-transparent background
                        bordercolor='rgba(0, 0, 0, 0.1)',
                        borderwidth=1
                    ),
                    # Add shaded regions to distinguish different periods
                    shapes=[
                        # 1. Determine the end of rainfall forecast period
                        # For actual weather forecasts or when zero rainfall is assumed, use the last date
                        dict(
                            type="rect",
                            xref="x",
                            yref="paper",
                            x0=stop_datetime,
                            y0=0,
                            # Use the last rainfall date as the boundary
                            x1=rainfall_dates[-1] if len(rainfall_dates) > 0 else stop_datetime,
                            y1=1,
                            fillcolor="rgba(200, 255, 200, 0.3)",  # Light green for rain forecast period
                            line_width=0,
                            layer="below"
                        ),
                        # 2. Add different shading for flow recession period (after rainfall forecast ends)
                        dict(
                            type="rect",
                            xref="x",
                            yref="paper",
                            # Start from the end of rainfall forecast
                            x0=rainfall_dates[-1] if len(rainfall_dates) > 0 else stop_datetime,
                            y0=0,
                            x1=future_stop_datetime,
                            y1=1,
                            fillcolor="rgba(200, 230, 255, 0.2)",  # Light blue for recession period
                            line_width=0,
                            layer="below"
                        )
                    ]
                )
                
                # Save the plot path for later reference
                plot_filename = f"flow_projection_{site}_{datetime.datetime.now().strftime('%Y%m%d')}.html"
                plot_path = os.path.join(PLOTS_DIR, plot_filename)
                st.session_state.plot_path = plot_path
                
                # Show the plot
                st.plotly_chart(fig, use_container_width=True)
                
                # Save the plot to the plots directory
                fig.write_html(plot_path)
                
                # Display additional information at the bottom
                with st.expander("Projection Details", expanded=False):
                    # Display watershed information if available
                    if 'watershed_info_message' in st.session_state and st.session_state.watershed_info_message:
                        st.info(st.session_state.watershed_info_message)
                        
                    # Display flow combination message if available
                    if 'flow_combination_message' in st.session_state and st.session_state.flow_combination_message:
                        st.success(st.session_state.flow_combination_message)
                    
                    st.write(f"Integration starting from {integration_info['start_date']} with initial flow: {integration_info['initial_flow']:.2f} cfs")
                    
                    # Show information about the distinct projection periods
                    st.markdown("**Projection Periods on Chart:**")
                    st.markdown("- **Historical Period**: No shading")
                    st.markdown("- **Rainfall Forecast Period**: Light green shading")
                    st.markdown("- **Flow Recession Period**: Light blue shading")
                    
                    if integration_info['forecast_start'] is not None:
                        st.write(f"Using rainfall forecast from {integration_info['forecast_start']} ({integration_info['forecast_start_val']:.2f} mm) to {integration_info['forecast_end']} ({integration_info['forecast_end_val']:.2f} mm)")
                        
                        # Highlight today's rainfall if available
                        if today_in_forecast:
                            today_rainfall = df_forecast['ppt'].values[forecast_start_idx]
                            st.info(f"Today's rainfall forecast: {today_rainfall:.2f} mm")
                    
                    # Add precipitation table
                    st.subheader("Precipitation Forecast Table")
                    
                    # Create a clean dataframe for display
                    precip_table = pd.DataFrame({
                        'Date': pd.to_datetime(df_forecast.index).date,
                        'Rainfall (mm)': df_forecast['ppt'].values.round(2)
                    })
                    
                    # Highlight today's row with custom styling
                    def highlight_today(row):
                        if row.name in precip_table[precip_table['Date'] == today].index:
                            return ['background-color: rgba(255, 165, 0, 0.2)'] * len(row)
                        return [''] * len(row)
                    
                    # Display the table with styling
                    st.dataframe(
                        precip_table.style.apply(highlight_today, axis=1),
                        use_container_width=True,
                        hide_index=True
                    )
                
                # Store the data in session state
                st.session_state.historical_data = df.copy()
                st.session_state.projected_data = natQ_df.copy()
                st.session_state.plot_dates = {
                    'stop_datetime': stop_datetime,
                    'future_stop_datetime': future_stop_datetime
                }
                
                # Save data to files in plots directory
                date_str = datetime.datetime.now().strftime("%Y%m%d")
                historical_filename = f"historical_flow_{site}_{date_str}.csv"
                projection_filename = f"projected_flow_{site}_{date_str}_{current_model}.csv"
                
                # Save files to plots directory
                historical_path = os.path.join(PLOTS_DIR, historical_filename)
                projection_path = os.path.join(PLOTS_DIR, projection_filename)
                df.to_csv(historical_path)
                natQ_df.to_csv(projection_path)
                
                # Create HTML download links instead of Streamlit download buttons
                historical_link = get_csv_download_link(df, historical_filename, "Download Historical Data")
                projection_link = get_csv_download_link(natQ_df, projection_filename, "Download Projected Data")
                
                # Create columns for download links to place them side by side
                col1, col2 = st.columns(2)
                
                # Display the download links
                with col1:
                    st.markdown(historical_link, unsafe_allow_html=True)
                
                with col2:
                    st.markdown(projection_link, unsafe_allow_html=True)
                
                # Display data tables
                table_col1, table_col2 = st.columns(2)
                with table_col1:
                    st.subheader("Historical Data (last 30 days)")
                    st.dataframe(df.tail(30))
                
                with table_col2:
                    st.subheader(f"Projected Flow ({T} days)")
                    st.dataframe(natQ_df)
            else:
                if gage_selection == "Custom/Other Gage...":
                    st.error(f"Failed to retrieve data for custom gauge {site}. Please verify the USGS gauge ID and try again.")
                else:
                    st.error(f"Failed to retrieve data for gauge {gage_selection} (ID: {site}). Please try another gauge or check the USGS system.")
                # Reset analysis state if data retrieval failed
                st.session_state.has_run_analysis = False

if __name__ == '__main__':
    main()
