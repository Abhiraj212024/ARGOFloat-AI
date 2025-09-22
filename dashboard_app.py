import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
import sys
import os
import uuid

# Add the current directory to Python path to import rag module
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import our RAG pipeline
try:
    from rag import create_rag_pipeline, RAGPipeline
    RAG_AVAILABLE = True
except ImportError as e:
    st.error(f"Could not import RAG pipeline: {e}")
    RAG_AVAILABLE = False

st.set_page_config(layout="wide", page_title="FloatChat Dashboard")

# ----------------------------
# Visualization Functions
# ----------------------------
# def show_map(df):
    # """Create geographic map visualization of float locations."""
    # if df.empty:
    #     st.error("No data available for map visualization")
    #     return None
    
    # required_cols = ["lat", "lon"]
    # missing_cols = [col for col in required_cols if col not in df.columns]
    
    # if missing_cols:
    #     st.error(f"Map requires columns: {', '.join(missing_cols)}")
    #     return None
    
    # try:
    #     # Create the map
    #     fig = px.scatter_geo(
    #         df, 
    #         lat="lat", 
    #         lon="lon", 
    #         color="temperature" if "temperature" in df.columns else None,
    #         hover_name="float_id" if "float_id" in df.columns else None,
    #         hover_data={col: True for col in df.columns if col not in ["lat", "lon"]},
    #         size_max=15,
    #         projection="natural earth",
    #         title="ARGO Floats Geographic Distribution"
    #     )
    #     fig.update_layout(
    #         height=600,
    #         showlegend=True
    #     )
        
    #     return fig
        
    # except Exception as e:
    #     st.error(f"Error creating map: {str(e)}")
    #     return None
# import plotly.express as px

def show_map(df):
    if df.empty:
        st.error("No data available for map visualization")
        return None
    
    required_cols = ["lat", "lon"]
    missing_cols = [col for col in required_cols if col not in df.columns]
    
    if missing_cols:
        st.error(f"Map requires columns: {', '.join(missing_cols)}")
        return None
    
    try:
        fig = px.scatter_mapbox(
            df,
            lat="lat",
            lon="lon",
            color="temperature" if "temperature" in df.columns else None,
            hover_name="float_id" if "float_id" in df.columns else None,
            hover_data={col: True for col in df.columns if col not in ["lat", "lon"]},
            size_max=15,
            zoom=1,
            mapbox_style="carto-darkmatter"  # 🌑 Dark background
        )
        fig.update_layout(height=600, showlegend=True)
        return fig
    except Exception as e:
        st.error(f"Error creating map: {str(e)}")
        return None
    
def show_profile(df):
    def show_map(df):
        if df.empty:
            st.error("No data available for map visualization")
        return None
    
    required_cols = ["lat", "lon"]
    missing_cols = [col for col in required_cols if col not in df.columns]
    
    if missing_cols:
        st.error(f"Map requires columns: {', '.join(missing_cols)}")
        return None
    
    try:
        fig = px.scatter_mapbox(
            df,
            lat="lat",
            lon="lon",
            color="temperature" if "temperature" in df.columns else None,
            hover_name="float_id" if "float_id" in df.columns else None,
            hover_data={col: True for col in df.columns if col not in ["lat", "lon"]},
            size_max=15,
            zoom=1,
            mapbox_style="carto-darkmatter"  # 🌑 Dark background
        )
        fig.update_layout(height=600, showlegend=True)
        return fig
    except Exception as e:
        st.error(f"Error creating map: {str(e)}")
        return None

def show_timeseries(df):
    """Create time series visualization for temperature and salinity."""
    if df.empty:
        st.error("No data available for time series visualization")
        return None
    
    required_cols = ["cycle", "temperature", "salinity"]
    missing_cols = [col for col in required_cols if col not in df.columns]
    
    if missing_cols:
        st.error(f"Time series plot requires columns: {', '.join(missing_cols)}")
        return None
    
    try:
        fig_ts = make_subplots(
            rows=1, cols=2, 
            subplot_titles=("Temperature vs Cycle", "Salinity vs Cycle"),
            horizontal_spacing=0.1
        )

        # Temperature time series
        fig_ts.add_trace(
            go.Scatter(
                x=df["cycle"], 
                y=df["temperature"],
                mode="lines+markers", 
                name="Temperature",
                line=dict(color="red", width=2),
                marker=dict(size=6)
            ), 
            row=1, col=1
        )

        # Salinity time series
        fig_ts.add_trace(
            go.Scatter(
                x=df["cycle"], 
                y=df["salinity"],
                mode="lines+markers", 
                name="Salinity",
                line=dict(color="blue", width=2),
                marker=dict(size=6)
            ), 
            row=1, col=2
        )

        # Update layout
        fig_ts.update_xaxes(title="Cycle #")
        fig_ts.update_yaxes(title="Temperature (°C)", row=1, col=1)
        fig_ts.update_yaxes(title="Salinity (PSU)", row=1, col=2)
        fig_ts.update_layout(
            height=500,
            title="Time Series Analysis",
            showlegend=False
        )
        
        return fig_ts
        
    except Exception as e:
        st.error(f"Error creating time series plot: {str(e)}")
        return None

# ----------------------------
# Query Classification
# ----------------------------
def classify_query(query: str) -> str:
    """
    Classify user query to determine appropriate response type.
    
    Args:
        query: User's input query
        
    Returns:
        Query type: 'map', 'profile', 'timeseries', or 'rag'
    """
    query_lower = query.lower().strip()
    
    # Map-related keywords
    map_keywords = ["map", "location", "geographic", "geography", "where", "lat", "lon", "coordinate"]
    if any(keyword in query_lower for keyword in map_keywords):
        return "map"
    
    # Profile-related keywords  
    profile_keywords = ["profile", "depth", "vertical", "deep", "shallow", "meter", "pressure"]
    if any(keyword in query_lower for keyword in profile_keywords):
        return "profile"
    
    # Time series keywords
    timeseries_keywords = ["time", "cycle", "temporal", "trend", "over time", "series", "timeline"]
    if any(keyword in query_lower for keyword in timeseries_keywords):
        return "timeseries"
    
    # Default to RAG for complex queries
    return "rag"

# ----------------------------
# RAG Pipeline Integration
# ----------------------------
@st.cache_resource
def load_rag_pipeline():
    """Load and cache the RAG pipeline instance."""
    if not RAG_AVAILABLE:
        return None
    
    try:
        with st.spinner("Initializing RAG pipeline..."):
            rag = create_rag_pipeline()
            if rag.test_connection():
                return rag
            else:
                st.error("Failed to connect to database or LLM")
                return None
    except Exception as e:
        st.error(f"Error initializing RAG pipeline: {e}")
        return None

def handle_rag_query(rag_pipeline: RAGPipeline, query: str) -> dict:
    """
    Process query through RAG pipeline.
    
    Args:
        rag_pipeline: Initialized RAG pipeline instance
        query: User query
        
    Returns:
        Result dictionary with answer, SQL, data, etc.
    """
    if not rag_pipeline:
        return {
            "answer": "RAG pipeline is not available. Please check your configuration.",
            "sql": "",
            "data": pd.DataFrame(),
            "success": False
        }
    
    return rag_pipeline.process_query(query)

# ----------------------------
# Main Application
# ----------------------------
def main():
    st.title("FloatChat 🌊")
    st.markdown("**Your AI Assistant for Oceanographic Data Analysis**")
    st.markdown("Ask me to show visualizations or answer detailed questions about ocean float data!")
    
    # Load RAG pipeline
    rag_pipeline = load_rag_pipeline() if RAG_AVAILABLE else None
    
    # Sample data for basic visualizations (fallback)
    df_sample = pd.DataFrame({
        "lat": [],
        "lon": [],
        "temperature": [],
        "cycle": [],
        "float_id": [],
        "depth": [],
        "salinity": []
    })
    # df_sample = pd.DataFrame({
    #     "lat": [10.5, 20.3, -15.7, 30.2, 5.1],
    #     "lon": [60.8, 80.1, -45.3, 120.7, 75.4],
    #     "temperature": [28.5, 15.2, 22.1, 18.9, 26.3],
    #     "cycle": [1, 2, 3, 4, 5],
    #     "float_id": ["F001", "F002", "F003", "F004", "F005"],
    #     "depth": [0, 50, 100, 150, 200],
    #     "salinity": [35.1, 35.2, 35.0, 35.3, 35.1]
    # })
    
    # Initialize session state
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    # Display chat history
    for idx, message in enumerate(st.session_state.messages):
        with st.chat_message(message["role"]):
            if message["role"] == "assistant":
                # Handle different types of assistant messages
                if "chart_type" in message:
                    # Recreate visualization using stored chart_id
                    chart_type = message["chart_type"]
                    data = message.get("data", df_sample)
                    chart_id = message.get("chart_id", f"fallback_{idx}")  # Fallback for old messages
                    
                    if chart_type == "map":
                        st.markdown("🗺️ **Geographic Distribution of Ocean Floats:**")
                        fig = show_map(data)
                        if fig:
                            st.plotly_chart(fig, use_container_width=True, key=chart_id)
                    
                    elif chart_type == "profile":
                        st.markdown("📊 **Oceanographic Profiles vs Depth:**")
                        fig = show_profile(data)
                        if fig:
                            st.plotly_chart(fig, use_container_width=True, key=chart_id)
                    
                    elif chart_type == "timeseries":
                        st.markdown("📈 **Time Series Analysis:**")
                        fig = show_timeseries(data)
                        if fig:
                            st.plotly_chart(fig, use_container_width=True, key=chart_id)
                
                elif "rag_response" in message:
                    # RAG response with expandable sections
                    st.markdown(message["content"])
                    
                    if message.get("sql"):
                        with st.expander("🔍 View Generated SQL Query"):
                            st.code(message["sql"], language="sql")
                    
                    if not message.get("data", pd.DataFrame()).empty:
                        with st.expander("📋 View Raw Data Results"):
                            st.dataframe(message["data"], use_container_width=True)
                
                else:
                    # Simple text message
                    st.markdown(message["content"])
            else:
                # User message
                st.markdown(message["content"])
    
    # Chat input
    if prompt := st.chat_input("What would you like to know about the ocean data?"):
        # Add user message to history
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Display user message
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Process the query
        with st.chat_message("assistant"):
            query_type = classify_query(prompt)
            
            if query_type == "map":
                st.markdown("🗺️ **Geographic Distribution of Ocean Floats:**")
                fig = show_map(df_sample)
                if fig:
                    chart_id = str(uuid.uuid4())  # Generate unique ID
                    st.plotly_chart(fig, use_container_width=True, key=chart_id)
                    # Store in session state with unique chart ID
                    st.session_state.messages.append({
                        "role": "assistant",
                        "chart_type": "map",
                        "data": df_sample,
                        "chart_id": chart_id
                    })
                else:
                    error_msg = "Unable to generate map visualization."
                    st.error(error_msg)
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": error_msg
                    })
            
            elif query_type == "profile":
                st.markdown("📊 **Oceanographic Profiles vs Depth:**")
                fig = show_profile(df_sample)
                if fig:
                    chart_id = str(uuid.uuid4())  # Generate unique ID
                    st.plotly_chart(fig, use_container_width=True, key=chart_id)
                    # Store in session state with unique chart ID
                    st.session_state.messages.append({
                        "role": "assistant",
                        "chart_type": "profile",
                        "data": df_sample,
                        "chart_id": chart_id
                    })
                else:
                    error_msg = "Unable to generate profile visualization."
                    st.error(error_msg)
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": error_msg
                    })
            
            elif query_type == "timeseries":
                st.markdown("📈 **Time Series Analysis:**")
                fig = show_timeseries(df_sample)
                if fig:
                    chart_id = str(uuid.uuid4())  # Generate unique ID
                    st.plotly_chart(fig, use_container_width=True, key=chart_id)
                    # Store in session state with unique chart ID
                    st.session_state.messages.append({
                        "role": "assistant",
                        "chart_type": "timeseries", 
                        "data": df_sample,
                        "chart_id": chart_id
                    })
                else:
                    error_msg = "Unable to generate time series visualization."
                    st.error(error_msg)
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": error_msg
                    })
            
            else:  # RAG query
                if not RAG_AVAILABLE or not rag_pipeline:
                    fallback_msg = """
                    🚫 **Advanced data analysis is currently unavailable.**
                    
                    I can help you with basic visualizations:
                    - Say "show map" for geographic distribution
                    - Say "show profiles" for depth profiles  
                    - Say "show time series" for temporal analysis
                    
                    For complex queries, please ensure the RAG pipeline is properly configured.
                    """
                    st.markdown(fallback_msg)
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": fallback_msg
                    })
                else:
                    with st.spinner("🤔 Analyzing your question..."):
                        result = handle_rag_query(rag_pipeline, prompt)
                        
                        # Display the answer
                        st.markdown(result["answer"])
                        
                        # Show SQL query if available
                        if result.get("sql") and result["success"]:
                            with st.expander("🔍 View Generated SQL Query"):
                                st.code(result["sql"], language="sql")
                        
                        # Show raw data if available
                        if not result.get("data", pd.DataFrame()).empty and "Error" not in result["data"].columns:
                            with st.expander("📋 View Raw Data Results"):
                                st.dataframe(result["data"], use_container_width=True)
                        
                        # Store in session state
                        st.session_state.messages.append({
                            "role": "assistant",
                            "content": result["answer"],
                            "sql": result.get("sql", ""),
                            "data": result.get("data", pd.DataFrame()),
                            "rag_response": True
                        })

    # Sidebar with helpful information
    with st.sidebar:
        st.subheader("💡 How to Use FloatChat")
        
        st.markdown("""
        **Quick Visualizations:**
        - "Show me the map" → Geographic distribution
        - "Display profiles" → Temperature/Salinity vs depth
        - "Plot time series" → Temporal trends
        
        **Data Analysis Questions:**
        - "What's the average temperature at 500m?"
        - "Show salinity data from 2012 onwards"
        - "Find the deepest measurements"
        - "How many readings do we have?"
        """)
        
        # System status
        st.subheader("🔧 System Status")
        if RAG_AVAILABLE and rag_pipeline:
            st.success("✅ RAG Pipeline: Active")
            
            # Show available columns if possible
            try:
                columns = rag_pipeline.get_available_columns()
                if columns:
                    with st.expander("📊 Available Data Columns"):
                        for col in columns:
                            st.text(f"• {col}")
            except:
                pass
        else:
            st.warning("⚠️ RAG Pipeline: Unavailable")
        
        st.info("💾 Visualizations: Sample Data")
        
        # Quick stats
        st.subheader("📈 Quick Stats")
        st.metric("Sample Float Count", len(df_sample))
        st.metric("Temperature Range", f"{df_sample['temperature'].min():.1f}°C - {df_sample['temperature'].max():.1f}°C")
        st.metric("Depth Range", f"{df_sample['depth'].min()}m - {df_sample['depth'].max()}m")

if __name__ == "__main__":
    main()