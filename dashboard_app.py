import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px

st.set_page_config(layout="wide", page_title="FloatChat Dashboard")

tabs = ["🌍 Map", "📊 Profiles", "📈 Time Series"]
df = pd.DataFrame({
    "lat": [10, 20, -15, 30],
    "lon": [60, 80, -45, 120],
    "temperature": [100,0,25,40],
    "cycle": [1,2,3,4],
    "float_id": ["F001", "F002", "F003", "F004"],"depth": [0, 50, 100, 200],
        "salinity": [35, 35.1, 35.2, 35.3]
    }) #I need this from you guys
# with tabs[0]:
    # st.header("Map View")
    
    # import plotly.express as px
    # Define the function (as provided in your prompt)
def show_map(df):
    st.header("ARGO Floats Map")
    fig = px.scatter_geo(
        df, lat="lat", lon="lon", color="temperature",
        hover_name="float_id", size_max=10,
        projection="natural earth"
    )
    return fig
    # st.map(df, latitude="lat", longitude="lon", zoom=2)

# with tabs[1]:
def show_profile(df):
    st.header("Float Profile (Temperature & Salinity vs Depth)")
    fig = make_subplots(rows=1, cols=2, shared_yaxes=True,
        subplot_titles=("Temperature (°C)", "Salinity (PSU)"))

    fig.add_trace(go.Scatter(x=df["temperature"], y=df["depth"], mode="lines+markers"), col=1, row=1)
    fig.add_trace(go.Scatter(x=df["salinity"], y=df["depth"], mode="lines+markers"), col=2, row=1)

    fig.update_yaxes(autorange="reversed", title="Depth (m)")
    return fig
    # fig = show_profile(df)
    # st.plotly_chart(fig, use_container_width=True,key="profile_chart")
# with tabs[2]:
def show_timeseries(df):
    st.header("Time Series Plots")
    fig_ts = make_subplots(rows=1, cols=2, subplot_titles=("Temperature vs Cycle", "Salinity vs Cycle"))

    fig_ts.add_trace(go.Scatter(x=df["cycle"], y=df["temperature"],
                                mode="lines+markers", name="Temperature"), row=1, col=1)

    fig_ts.add_trace(go.Scatter(x=df["cycle"], y=df["salinity"],
                                mode="lines+markers", name="Salinity"), row=1, col=2)

    fig_ts.update_xaxes(title="Cycle #")
    fig_ts.update_yaxes(title="Temperature (°C)", row=1, col=1)
    fig_ts.update_yaxes(title="Salinity (PSU)", row=1, col=2)
    return fig_ts
    # fig_ts=show_timeseries(df)
    # st.plotly_chart(fig_ts, use_container_width=True,key="timeseries_chart")
# --- CONVERSATIONAL INTERFACE (Task 2 Logic) ---

st.title("FloatChat 🌊")
st.markdown("Ask me about the data! Try queries like 'show me the map', 'show the profiles', or 'plot time series'.")

# Initialize chat history in session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "unhandled_queries" not in st.session_state:
    st.session_state.unhandled_queries = []    

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        if "chart_type" in message:
            # Re-render the chart based on the stored type
            if message["chart_type"] == "map":
                st.map(df, latitude="lat", longitude="lon", zoom=2)
            elif message["chart_type"] == "profile":
                fig = show_profile(df)
                st.plotly_chart(fig, use_container_width=True)
            elif message["chart_type"] == "timeseries":
                fig = show_timeseries(df)
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.markdown(message["content"])

# Accept user input
if prompt := st.chat_input("What do you want to see?"):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(prompt)

    # Determine which visualization to render based on the prompt
    with st.chat_message("assistant"):
        prompt_lower = prompt.lower()
        if "map" in prompt_lower:
            st.markdown("Here is the map view of the floats:")
            st.map(df, latitude="lat", longitude="lon", zoom=2)
            # Store the chart type to redraw on rerun
            st.session_state.messages.append({"role": "assistant", "chart_type": "map"})
        elif "profile" in prompt_lower:
            st.markdown("Here are the temperature and salinity profiles vs depth:")
            fig = show_profile(df)
            st.plotly_chart(fig, use_container_width=True)
            # Store the chart type to redraw on rerun
            st.session_state.messages.append({"role": "assistant", "chart_type": "profile"})
        elif "time series" in prompt_lower or "timeseries" in prompt_lower:
            st.markdown("Here are the time series plots for temperature and salinity over cycles:")
            fig = show_timeseries(df)
            st.plotly_chart(fig, use_container_width=True)
            # Store the chart type to redraw on rerun
            st.session_state.messages.append({"role": "assistant", "chart_type": "timeseries"})
        else:
            response = "I'm sorry, I can only show maps, profiles, or time series plots. Please try again with a different query."
            st.markdown(response)
            # Store the text message
            st.session_state.messages.append({"role": "assistant", "content": response})
            st.session_state.unhandled_queries.append(prompt) 
st.sidebar.subheader("Unhandled Queries")
st.sidebar.write(st.session_state.unhandled_queries)
