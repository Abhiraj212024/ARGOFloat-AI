import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px

st.set_page_config(layout="wide", page_title="FloatChat Dashboard")

tabs = st.tabs(["🌍 Map", "📊 Profiles", "📈 Time Series"])
df = pd.DataFrame({
    "lat": [10, 20, -15, 30],
    "lon": [60, 80, -45, 120],
    "temperature": [100,0,25,40],
    "cycle": [1,2,3,4],
    "float_id": ["F001", "F002", "F003", "F004"],"depth": [0, 50, 100, 200],
        "salinity": [35, 35.1, 35.2, 35.3]
    }) #I need this from you guys
with tabs[0]:
    st.header("Map View")
    
    import plotly.express as px
    # Define the function (as provided in your prompt)
    def show_map(df):
        fig = px.scatter_geo(
            df, lat="lat", lon="lon", color="temperature",
            hover_name="float_id", size_max=10,
            projection="natural earth"
        )
        return fig
    st.header("ARGO Floats Map")
    st.map(df, latitude="lat", longitude="lon", zoom=2)

with tabs[1]:
    st.header("Float Profile (Temperature & Salinity vs Depth)")
    def show_profile(df):
        fig = make_subplots(rows=1, cols=2, shared_yaxes=True,
                            subplot_titles=("Temperature (°C)", "Salinity (PSU)"))

        fig.add_trace(go.Scatter(x=df["temperature"], y=df["depth"], mode="lines+markers"), col=1, row=1)
        fig.add_trace(go.Scatter(x=df["salinity"], y=df["depth"], mode="lines+markers"), col=2, row=1)

        fig.update_yaxes(autorange="reversed", title="Depth (m)")
        return fig
    fig = show_profile(df)
    st.plotly_chart(fig, use_container_width=True)
with tabs[2]:
    st.header("Time Series Plots")

    fig_ts = make_subplots(rows=1, cols=2, subplot_titles=("Temperature vs Cycle", "Salinity vs Cycle"))

    fig_ts.add_trace(go.Scatter(x=df["cycle"], y=df["temperature"],
                                mode="lines+markers", name="Temperature"), row=1, col=1)

    fig_ts.add_trace(go.Scatter(x=df["cycle"], y=df["salinity"],
                                mode="lines+markers", name="Salinity"), row=1, col=2)

    fig_ts.update_xaxes(title="Cycle #")
    fig_ts.update_yaxes(title="Temperature (°C)", row=1, col=1)
    fig_ts.update_yaxes(title="Salinity (PSU)", row=1, col=2)

    st.plotly_chart(fig_ts, use_container_width=True)
