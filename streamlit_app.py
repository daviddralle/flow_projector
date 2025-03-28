import streamlit as st
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from scipy.optimize import curve_fit
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d
from pynhd import NLDI
from pygeohydro import NWIS
import datetime
import warnings
from streamlit import cache_data
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

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

# Main Streamlit app
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
        # Date range - exactly as in original script
        now = pd.to_datetime(datetime.datetime.now().strftime('%Y-%m-%d'))
        start = now - pd.to_timedelta(365, unit='d')
        stop = now
        future_stop = now + pd.to_timedelta(T, unit='d')
        start_str = start.strftime('%Y-%m-%d')
        stop_str = stop.strftime('%Y-%m-%d')
        stop_datetime = pd.to_datetime(stop_str)
        future_stop_str = (future_stop - pd.to_timedelta(1, unit='d')).strftime('%Y-%m-%d')
        future_stop_datetime = pd.to_datetime(future_stop_str)
        twomonths = (stop_datetime - pd.to_timedelta(90, unit='d')).strftime('%Y-%m-%d')

        with st.spinner(f"Retrieving data for gauge {site}..."):
            # Grab data
            df, basin = getFlow(site, start_str, stop_str)
            
            if df is not None and len(df) > 0:
                # Process data exactly as in original script
                QS = []
                DQS = []
                
                qs = df[site].loc[twomonths:stop_str].values
                dqs = np.gradient(qs)
                idx = (dqs < 0) & (qs > 0)
                QS = qs[idx]
                DQS = dqs[idx]
                
                # Polynomial fit exactly as in original
                p = np.polyfit(x=np.log(QS), y=np.log(-DQS), deg=2)
                # Critical step from original script
                p[1] = p[1] - 1
                
                # Integrate to get natural flow projection
                t = np.linspace(0, T, T)
                
                def fun(time, q):
                    return -gQ(q, p) * q
                    
                q0 = df[site].values[-1]
                sol = solve_ivp(fun, [0, t[-1]], [q0], rtol=1e-5)
                sol_int = interp1d(sol.t, sol.y[0], fill_value=0, bounds_error=False)
                natQ = sol_int(t)
                
                # Create projection dataframe as in original
                idx = pd.date_range(stop_datetime, future_stop_datetime, freq='D')
                natQ_df = pd.DataFrame({'Flow projection': natQ}, index=idx)
                
                # Interactive Plotly plot
                st.subheader('Flow Data and Projection (Interactive)')
                
                # Create plotly figure
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
                
                # Set up the layout with log scale for y-axis
                fig.update_layout(
                    title=f"Flow Projection for USGS Gage {site}",
                    xaxis_title='Date',
                    yaxis_title='Flow (cfs)',
                    yaxis_type='log',
                    xaxis=dict(range=[stop_datetime - pd.to_timedelta(60, unit='d'), future_stop_datetime]),
                    hovermode='closest',
                    legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
                    margin=dict(l=40, r=40, t=40, b=40),
                    height=500,
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
