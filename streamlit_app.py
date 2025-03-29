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

warnings.filterwarnings('ignore')

# Function to get flow data
@cache_data
def getFlow(site, start, stop):
    nwis = NWIS()
    nldi = NLDI()
    df = nwis.get_streamflow(site, (start, stop), freq="dv")
    df.columns = ['q']
    df.index = pd.to_datetime(df.index)
    df = df.tz_localize(None)
    df = df.resample('D').mean()
    basin = nldi.get_basins(site).to_crs('epsg:26910')

    try:
        geoms = [item for item in list(basin.geometry[0])]
        idx = np.argmax([item.area for item in geoms])
        basin.geometry = [geoms[idx]]
        st.info('Found multipolygon - fixing')
    except:
        basin.geometry = basin.geometry
    area_mm2 = basin.to_crs('epsg:26910').geometry[0].area*1000**2
    df.q = 35.3147*df.q
    df.columns = [site]
    return df, basin

# Sensitivity function
@cache_data
def gQ(q, p):
    if np.size(np.array(q)) == 1:
        return np.exp(np.sum([p[i] * np.log(q) ** (len(p) - i - 1) for i in range(len(p))]))
    return [np.exp(np.sum([p[i] * np.log(qq) ** (len(p) - i - 1) for i in range(len(p))])) for qq in np.array(q)]

def get_daily_rainfall_forecast(latitude, longitude):
    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": latitude,
        "longitude": longitude,
        "daily": "precipitation_sum",
        "timezone": "auto"
    }
    response = requests.get(url, params=params)
    data = response.json()
    return data

def main():
    st.title('Streamflow Projection App')
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
    if 'plot_dates' not in st.session_state:
        st.session_state.plot_dates = {
            'stop_datetime': None,
            'future_stop_datetime': None
        }
        
    # User inputs
    site = st.sidebar.text_input('Enter USGS Gauge ID', '11476500')
    T = st.sidebar.number_input('Projection Period (days)', min_value=1, value=90)
    
    def run_analysis():
        st.session_state.has_run_analysis = True
        st.session_state.site_id = site
        st.session_state.T_days = T
    
    if st.sidebar.button('Run Analysis', on_click=run_analysis) or st.session_state.has_run_analysis:
        # Date range for getting historical data
        now = pd.to_datetime(datetime.datetime.now().strftime('%Y-%m-%d'))
        # Use 10 years of historical data instead of just 1 year
        start = now - pd.to_timedelta(10*365, unit='d')
        stop = now
        future_stop = now + pd.to_timedelta(T, unit='d')
        start_str = start.strftime('%Y-%m-%d')
        stop_str = stop.strftime('%Y-%m-%d')
        stop_datetime = pd.to_datetime(stop_str)
        future_stop_str = (future_stop - pd.to_timedelta(1, unit='d')).strftime('%Y-%m-%d')
        future_stop_datetime = pd.to_datetime(future_stop_str)

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
                
                if len(seasonal_data) > 0:
                    qs = seasonal_data[site].values
                    # Calculate gradients on the full time series first
                    all_qs = df[site].values
                    all_dqs = np.gradient(all_qs)
                    # Map gradients back to the seasonal data
                    seasonal_indices = np.where(mask)[0]
                    dqs = all_dqs[seasonal_indices]
                    
                    # Find recessions (when flow is decreasing and positive)
                    idx = (dqs < 0) & (qs > 0)
                    QS = qs[idx]
                    DQS = dqs[idx]
                else:
                    st.warning("Not enough historical data matching the projection period's days of year. Using all available data instead.")
                    qs = df[site].values
                    dqs = np.gradient(qs)
                    idx = (dqs < 0) & (qs > 0)
                    QS = qs[idx]
                    DQS = dqs[idx]
                
                # Integrate to get natural flow projection
                t = np.linspace(0, T-1, T)

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

                popt, _ = curve_fit(logeps, QS, np.log(-DQS))
                geo = basin.to_crs('epsg:4326').geometry.values[0].centroid
                area_ft2 = basin.to_crs('epsg:26910').geometry.values[0].area*10.7639104167
                lat,lon = geo.y,geo.x
                forecast = get_daily_rainfall_forecast(lat, lon)
                time = forecast['daily']['time']
                precipitation_sum = forecast['daily']['precipitation_sum']
                df_forecast = pd.DataFrame({'ppt': precipitation_sum}, index=pd.to_datetime(time))
                forecast_times = np.array(range(0,len(df_forecast)))
                rain_vals = np.zeros_like(t)
                # convert forecast mm/day into ft3/s increments
                rain_vals[forecast_times] = 3.79727e-8*area_ft2*df_forecast['ppt'].values
                print(rain_vals)
                rain_vals[1] = rain_vals[0] + rain_vals[1]
                rain_vals[0] = 0
                print(rain_vals)
                forcing = interp1d(t, rain_vals, fill_value='extrapolate')
                def fun(time,q):
                    return -newg(q, popt)*(q - forcing(time))
                
                q0 = df[site].values[-1]
                sol = solve_ivp(fun, [0, t[-1]], [q0], rtol=1e-5)
                sol_int = interp1d(sol.t, sol.y[0], fill_value=0, bounds_error=False)
                natQ = sol_int(t)
                
                # Create projection dataframe as in original
                idx = pd.date_range(stop_datetime-pd.to_timedelta(1,unit='d'), future_stop_datetime - pd.to_timedelta(1,unit='d'), freq='D')
                natQ_df = pd.DataFrame({'Flow projection': natQ}, index=idx)
                
                # Interactive Plotly plot
                st.subheader('Flow Data and Projection with Rainfall Forecast (Interactive)')
                
                # Create plotly figure with secondary y-axis for rainfall
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
                    marker=dict(symbol='star', color='yellow', size=14, line=dict(color='black', width=2)),
                    name='Projection Start Point',
                    hoverinfo='text',
                    text=f"Projection Start: {start_date_str}<br>Flow: {start_flow:.2f} cfs"
                ))
                
                # Add rainfall forecast to the plot on secondary y-axis
                rainfall_dates = pd.to_datetime(forecast['daily']['time'])
                
                # Create bar chart for rainfall with custom blue color
                fig.add_trace(go.Bar(
                    x=rainfall_dates,
                    y=df_forecast['ppt'],
                    name='Rainfall Forecast',
                    marker_color='rgba(0, 100, 255, 0.6)',
                    yaxis='y2',  # Use secondary y-axis
                    hovertemplate='<b>Date</b>: %{x}<br><b>Rainfall</b>: %{y:.1f} mm<extra></extra>'
                ))
                
                # Get the rainfall forecast date range for the title
                forecast_start_date = rainfall_dates[0].strftime('%b %d, %Y')
                forecast_end_date = rainfall_dates[-1].strftime('%b %d, %Y')
                
                # Set up the layout with log scale for primary y-axis and secondary y-axis for rainfall
                fig.update_layout(
                    title={
                        'text': f"Flow Projection for USGS Gage {site}<br>with Rainfall Forecast ({forecast_start_date} to {forecast_end_date})",
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
                        rangemode='nonnegative',
                        showgrid=False
                    ),
                    xaxis=dict(range=[stop_datetime - pd.to_timedelta(60, unit='d'), future_stop_datetime]),
                    hovermode='closest',
                    # Move legend below the chart instead of at the top
                    legend=dict(
                        orientation='h', 
                        yanchor='top', 
                        y=-0.15, 
                        xanchor='center', 
                        x=0.5
                    ),
                    # Increase margins, especially top margin to prevent title overlap with buttons
                    margin=dict(l=50, r=60, t=80, b=80),
                    height=570,
                    # Add a light blue rectangle to highlight the forecast period
                    shapes=[
                        dict(
                            type="rect",
                            xref="x",
                            yref="paper",
                            x0=stop_datetime,
                            y0=0,
                            x1=future_stop_datetime,
                            y1=1,
                            fillcolor="rgba(200, 230, 255, 0.2)",
                            line_width=0,
                            layer="below"
                        )
                    ]
                )
                
                # Show the plot
                st.plotly_chart(fig, use_container_width=True)
                
                # Store the data in session state
                st.session_state.historical_data = df
                st.session_state.projected_data = natQ_df
                st.session_state.plot_dates = {
                    'stop_datetime': stop_datetime,
                    'future_stop_datetime': future_stop_datetime
                }
                
                # Add download buttons
                date_str = datetime.datetime.now().strftime("%Y%m%d")
                
                # Create columns for download buttons to place them side by side
                col1, col2 = st.columns(2)
                
                # Add a download button for the historical data
                with col1:
                    csv_historical = df.to_csv()
                    historical_filename = f"historical_flow_{site}_{date_str}.csv"
                    st.download_button(
                        label="Download Historical Data",
                        data=csv_historical,
                        file_name=historical_filename,
                        mime="text/csv",
                    )
                
                # Add a download button for the projected data
                with col2:
                    csv_projection = natQ_df.to_csv()
                    projection_filename = f"projected_flow_{site}_{date_str}.csv"
                    st.download_button(
                        label="Download Projected Data",
                        data=csv_projection,
                        file_name=projection_filename,
                        mime="text/csv",
                    )
                
                # Display data tables
                table_col1, table_col2 = st.columns(2)
                with table_col1:
                    st.subheader("Historical Data (last 30 days)")
                    st.dataframe(df.tail(30))
                
                with table_col2:
                    st.subheader(f"Projected Flow ({T} days)")
                    st.dataframe(natQ_df)
            else:
                st.error(f"Failed to retrieve data for gauge {site}. Please check the gauge ID and try again.")
                # Reset analysis state if data retrieval failed
                st.session_state.has_run_analysis = False

if __name__ == '__main__':
    main()
