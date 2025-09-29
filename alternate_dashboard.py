import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
import sys
import os
import uuid

# Add the current directory to Python path to import modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import our modules
try:
    from rag import create_rag_pipeline, RAGPipeline
    from data_manager import DataManager
    RAG_AVAILABLE = True
except ImportError as e:
    st.error(f"Could not import required modules: {e}")
    RAG_AVAILABLE = False

st.set_page_config(layout="wide", page_title="FloatChat Dashboard")

# ----------------------------
# Visualization Functions
# ----------------------------
def show_map(df):
    """Create geographic map visualization of float locations."""
    if df.empty:
        st.error("No data available for map visualization")
        return None
    
    required_cols = ["lat", "lon"]
    missing_cols = [col for col in required_cols if col not in df.columns]
    
    if missing_cols:
        st.error(f"Map requires columns: {', '.join(missing_cols)}")
        return None
    
    try:
        # Create the map
        fig = px.scatter_geo(
            df, 
            lat="lat", 
            lon="lon", 
            color="temperature" if "temperature" in df.columns else None,
            hover_name="float_id" if "float_id" in df.columns else None,
            hover_data={col: True for col in df.columns if col not in ["lat", "lon"] and col in ["temperature", "salinity", "depth", "time"]},
            size_max=15,
            projection="natural earth",
            title="Ocean Float Geographic Distribution"
        )
        
        fig.update_layout(
            height=600,
            showlegend=True
        )
        
        return fig
        
    except Exception as e:
        st.error(f"Error creating map: {str(e)}")
        return None

def show_profile(df):
    """Create depth profile visualization for temperature and salinity."""
    if df.empty:
        st.error("No data available for profile visualization")
        return None
    
    required_cols = ["temperature", "salinity", "depth"]
    missing_cols = [col for col in required_cols if col not in df.columns]
    
    if missing_cols:
        st.error(f"Profile plot requires columns: {', '.join(missing_cols)}")
        return None
    
    try:
        # Get a single representative profile
        # Option 1: Use the profile with most depth levels
        if 'float_id' in df.columns:
            # Count depth measurements per float
            float_counts = df.groupby('float_id').size()
            best_float = float_counts.idxmax()
            profile_data = df[df['float_id'] == best_float].copy()
        else:
            # If no float_id, just use all data but group by depth
            profile_data = df.copy()
        
        # Sort by depth and remove duplicates at same depth (take mean)
        profile_data = profile_data.groupby('depth').agg({
            'temperature': 'mean',
            'salinity': 'mean'
        }).reset_index()
        
        # Sort by depth for proper profile
        profile_data = profile_data.sort_values('depth')
        
        # Remove any NaN values
        profile_data = profile_data.dropna(subset=['temperature', 'salinity', 'depth'])
        
        if profile_data.empty:
            st.error("No valid profile data after filtering")
            return None
        
        fig = make_subplots(
            rows=1, cols=2, 
            shared_yaxes=True,
            subplot_titles=("Temperature (°C)", "Salinity (PSU)"),
            horizontal_spacing=0.1
        )

        # Temperature profile
        fig.add_trace(
            go.Scatter(
                x=profile_data["temperature"], 
                y=profile_data["depth"], 
                mode="lines+markers",
                name="Temperature",
                line=dict(color="red", width=2),
                marker=dict(size=4)
            ), 
            col=1, row=1
        )
        
        # Salinity profile
        fig.add_trace(
            go.Scatter(
                x=profile_data["salinity"], 
                y=profile_data["depth"], 
                mode="lines+markers",
                name="Salinity",
                line=dict(color="blue", width=2),
                marker=dict(size=4)
            ), 
            col=2, row=1
        )

        # Update layout
        fig.update_yaxes(autorange="reversed", title="Depth (m)")
        fig.update_xaxes(title="Temperature (°C)", col=1)
        fig.update_xaxes(title="Salinity (PSU)", col=2)
        fig.update_layout(
            height=600,
            title="Oceanographic Profiles vs Depth",
            showlegend=False
        )
        
        return fig
        
    except Exception as e:
        st.error(f"Error creating profile plot: {str(e)}")
        return None

def show_timeseries(df):
    """Create time series visualization for temperature and salinity."""
    if df.empty:
        st.error("No data available for time series visualization")
        return None
    
    has_time_axis = 'time' in df.columns or 'cycle' in df.columns
    has_data = 'temperature' in df.columns or 'salinity' in df.columns

    if not has_time_axis:
        st.error("Time series plot requires time or cycle column")
        return None
    
    if not has_data:
        st.error("Time series plot requires temperature or salinity data")
        return None
    
    try:
        # Convert time column to datetime if it exists
        if 'time' in df.columns:
            df = df.copy()
            df['time'] = pd.to_datetime(df['time'], errors='coerce')
            df = df.dropna(subset=['time'])
            
            # Sort by time
            df = df.sort_values('time')
            
            # Aggregate by month for clearer trends (especially for multi-year data)
            df['year_month'] = df['time'].dt.to_period('M')
            
            # Group by month and calculate mean values
            monthly_data = df.groupby('year_month').agg({
                'temperature': 'mean',
                'salinity': 'mean'
            }).reset_index()
            
            # Convert period back to timestamp for plotting
            monthly_data['time'] = monthly_data['year_month'].dt.to_timestamp()
            
            x_col = 'time'
            x_title = "Time (Monthly Averages)"
            plot_data = monthly_data
            
        else:
            # Use cycle as fallback
            x_col = 'cycle'
            x_title = "Cycle #"
            plot_data = df.sort_values('cycle')
        
        # Create subplots only if we have both temperature and salinity
        has_temp = 'temperature' in plot_data.columns
        has_sal = 'salinity' in plot_data.columns
        
        if has_temp and has_sal:
            fig_ts = make_subplots(
                rows=1, cols=2, 
                subplot_titles=("Temperature vs " + x_title, "Salinity vs " + x_title),
                horizontal_spacing=0.1
            )

            # Temperature time series
            fig_ts.add_trace(
                go.Scatter(
                    x=plot_data[x_col], 
                    y=plot_data["temperature"],
                    mode="lines+markers", 
                    name="Temperature",
                    line=dict(color="red", width=2),
                    marker=dict(size=4)
                ), 
                row=1, col=1
            )

            # Salinity time series
            fig_ts.add_trace(
                go.Scatter(
                    x=plot_data[x_col], 
                    y=plot_data["salinity"],
                    mode="lines+markers", 
                    name="Salinity",
                    line=dict(color="blue", width=2),
                    marker=dict(size=4)
                ), 
                row=1, col=2
            )

            fig_ts.update_xaxes(title=x_title)
            fig_ts.update_yaxes(title="Temperature (°C)", row=1, col=1)
            fig_ts.update_yaxes(title="Salinity (PSU)", row=1, col=2)
            fig_ts.update_layout(
                height=500,
                title="Time Series Analysis",
                showlegend=False
            )
        else:
            # Single plot for either temperature or salinity
            y_col = 'temperature' if has_temp else 'salinity'
            y_title = "Temperature (°C)" if has_temp else "Salinity (PSU)"
            color = "red" if has_temp else "blue"
            
            fig_ts = go.Figure()
            fig_ts.add_trace(
                go.Scatter(
                    x=plot_data[x_col],
                    y=plot_data[y_col],
                    mode="lines+markers",
                    name=y_col.title(),
                    line=dict(color=color, width=2),
                    marker=dict(size=4)
                )
            )
            
            fig_ts.update_layout(
                title=f"{y_col.title()} vs {x_title}",
                xaxis_title=x_title,
                yaxis_title=y_title,
                height=500
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

    #Rag-related keywords
    rag_keywords = ["describe", "changing", "rag", "trend", "over time"]
    if any(keyword in query_lower for keyword in rag_keywords):
        return "rag"
    
    # Map-related keywords
    map_keywords = ["map", "location", "geographic", "geography", "where", "lat", "lon", "coordinate"]
    if any(keyword in query_lower for keyword in map_keywords):
        return "map"
    
    # Profile-related keywords  
    profile_keywords = ["profile", "depth", "vertical", "deep", "shallow", "meter", "pressure"]
    if any(keyword in query_lower for keyword in profile_keywords):
        return "profile"
    
    # Time series keywords
    timeseries_keywords = ["display", "time", "cycle", "temporal", "trend", "series", "timeline", "plot"]
    if any(keyword in query_lower for keyword in timeseries_keywords):
        return "timeseries"
    
    # Default to RAG for complex queries
    return "rag"

# ----------------------------
# Component Loading
# ----------------------------
@st.cache_resource
def load_data_manager():
    """Load and cache the DataManager instance."""
    try:
        return DataManager(db_path="./DB_files/data.duckdb", table_name="ocean_profiles", default_limit=1000000)
    except Exception as e:
        st.error(f"Error initializing DataManager: {e}")
        return None

@st.cache_resource
def load_rag_pipeline():
    """Load and cache the RAG pipeline instance."""
    if not RAG_AVAILABLE:
        return None
    
    try:
        with st.spinner("Initializing RAG pipeline..."):
            rag = create_rag_pipeline(db_path="./DB_files/data.duckdb", table_name="ocean_profiles")
            if rag.test_connection():
                return rag
            else:
                st.error("Failed to connect to database or LLM")
                return None
    except Exception as e:
        st.error(f"Error initializing RAG pipeline: {e}")
        return None

def handle_rag_query(rag_pipeline: RAGPipeline, data_manager: DataManager, query: str) -> dict:
    """
    Process query through RAG pipeline and update data manager.
    
    Args:
        rag_pipeline: Initialized RAG pipeline instance
        data_manager: DataManager instance
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
    
    # Check for requests to see sample data
    if any(keyword in query.lower() for keyword in ["sample", "example", "show data", "see data"]):
        try:
            sample_data = rag_pipeline.get_sample_data()
            if not sample_data.empty and "Error" not in sample_data.columns:
                # Update data manager with sample data
                fake_rag_result = {
                    "success": True,
                    "data": sample_data,
                    "sql": f"SELECT * FROM {rag_pipeline.TABLE_NAME} LIMIT 100;",
                    "answer": "Here's a sample of the available data:"
                }
                data_manager.update_from_rag_result(fake_rag_result, "sample")
                return fake_rag_result
        except:
            pass
    
    # Process normal query
    result = rag_pipeline.process_query(query)
    
    # If successful, update data manager
    if result["success"] and not result["data"].empty and "Error" not in result["data"].columns:
        data_manager.update_from_rag_result(result, classify_query(query))
    
    # If there's a date/time error, provide helpful suggestions
    if not result["success"] and "date" in str(result.get("error", "")).lower():
        try:
            time_analysis = rag_pipeline.analyze_time_column_format()
            suggestion = f"""
            **Date/Time Format Issue Detected** 🕐
            
            The database seems to have date/time data in a different format than expected.
            
            **Sample time values:** {time_analysis['samples']}
            
            **Suggestions:**
            - Try: "Show me sample data first"
            - Rephrase using year: "salinity data from year 2015 onwards"  
            - Or specify format: "salinity where time contains '2015'"
            
            {' '.join(time_analysis['suggestions'])}
            """
            result["answer"] = suggestion
        except:
            result["answer"] = """
            **Date/Time Format Issue** 🕐
            
            It looks like the date/time column format doesn't match what was expected.
            
            Try:
            - "Show me sample data" to see the date format
            - Rephrase using "year 2015" instead of "2015 onwards"
            - Use "time contains '2015'" for text-based date matching
            """
    
    return result

# ----------------------------
# Main Application
# ----------------------------
def main():
    st.title("FloatChat 🌊")
    st.markdown("**Your AI Assistant for Oceanographic Data Analysis**")
    st.markdown("Ask me to show visualizations or answer detailed questions about ocean float data!")
    
    # Load components
    rag_pipeline = load_rag_pipeline() if RAG_AVAILABLE else None
    data_manager = load_data_manager()
    
    if not data_manager:
        st.error("Failed to initialize data management system")
        return
    
    # Initialize session state
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    # Display chat history
    for idx, message in enumerate(st.session_state.messages):
        with st.chat_message(message["role"]):
            if message["role"] == "assistant":
                # Handle different types of assistant messages
                if "chart_type" in message:
                    # Recreate visualization using stored chart_id and current data
                    chart_type = message["chart_type"]
                    chart_id = message.get("chart_id", f"fallback_{idx}")
                    
                    # Get appropriate data for visualization type
                    if chart_type == "map":
                        st.markdown("🗺️ **Geographic Distribution of Ocean Floats:**")
                        viz_data = data_manager.get_map_data()
                        fig = show_map(viz_data)
                        if fig:
                            st.plotly_chart(fig, use_container_width=True, key=chart_id)
                    
                    elif chart_type == "profile":
                        st.markdown("📊 **Oceanographic Profiles vs Depth:**")
                        viz_data = data_manager.get_profile_data()
                        fig = show_profile(viz_data)
                        if fig:
                            st.plotly_chart(fig, use_container_width=True, key=chart_id)
                    
                    elif chart_type == "timeseries":
                        st.markdown("📈 **Time Series Analysis:**")
                        viz_data = data_manager.get_timeseries_data()
                        fig = show_timeseries(viz_data)
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
                viz_data = data_manager.get_map_data()
                fig = show_map(viz_data)
                if fig:
                    chart_id = str(uuid.uuid4())
                    st.plotly_chart(fig, use_container_width=True, key=chart_id)
                    # Store in session state with unique chart ID
                    st.session_state.messages.append({
                        "role": "assistant",
                        "chart_type": "map",
                        "chart_id": chart_id
                    })
                else:
                    error_msg = "Unable to generate map visualization with current data."
                    st.error(error_msg)
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": error_msg
                    })
            
            elif query_type == "profile":
                st.markdown("📊 **Oceanographic Profiles vs Depth:**")
                viz_data = data_manager.get_profile_data()
                fig = show_profile(viz_data)
                if fig:
                    chart_id = str(uuid.uuid4())
                    st.plotly_chart(fig, use_container_width=True, key=chart_id)
                    # Store in session state with unique chart ID
                    st.session_state.messages.append({
                        "role": "assistant",
                        "chart_type": "profile",
                        "chart_id": chart_id
                    })
                else:
                    error_msg = "Unable to generate profile visualization with current data."
                    st.error(error_msg)
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": error_msg
                    })
            
            elif query_type == "timeseries":
                st.markdown("📈 **Time Series Analysis:**")
                viz_data = data_manager.get_timeseries_data()
                fig = show_timeseries(viz_data)
                if fig:
                    chart_id = str(uuid.uuid4())
                    st.plotly_chart(fig, use_container_width=True, key=chart_id)
                    # Store in session state with unique chart ID
                    st.session_state.messages.append({
                        "role": "assistant",
                        "chart_type": "timeseries",
                        "chart_id": chart_id
                    })
                else:
                    error_msg = "Unable to generate time series visualization with current data."
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
                        result = handle_rag_query(rag_pipeline, data_manager, prompt)
                        
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
        st.subheader("📊 Current Data Status")
        data_info = data_manager.get_data_info()
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Data Source", data_info["source"].title())
            st.metric("Total Points", data_info["row_count"])
        with col2:
            st.metric("Float Count", data_info["float_count"])
            if data_info.get("date_range"):
                st.metric("Date Range", f"{data_info['date_range']['start']} to {data_info['date_range']['end']}")
        
        # Data capabilities
        st.subheader("🎯 Available Visualizations")
        viz_types = ["map", "profile", "timeseries"]
        for viz_type in viz_types:
            suitable, reason = data_manager.is_suitable_for_visualization(viz_type)
            status = "✅" if suitable else "❌"
            st.text(f"{status} {viz_type.title()}: {reason}")
        
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
                            
                    # Also show sample data
                    with st.expander("🔍 Sample Data Preview"):
                        try:
                            sample_data = rag_pipeline.get_sample_data(3)
                            if not sample_data.empty and "Error" not in sample_data.columns:
                                st.dataframe(sample_data, use_container_width=True)
                            else:
                                st.text("Could not retrieve sample data")
                        except:
                            st.text("Sample data unavailable")
            except:
                pass
        else:
            st.warning("⚠️ RAG Pipeline: Unavailable")
        
        st.success("✅ Data Manager: Active")
        
        # Quick stats from DataManager
        st.subheader("📈 Data Summary")
        if data_info.get("temp_range"):
            temp_range = f"{data_info['temp_range']['min']:.1f}°C - {data_info['temp_range']['max']:.1f}°C"
            st.metric("Temperature Range", temp_range)
        
        if data_info.get("depth_range"):
            depth_range = f"{data_info['depth_range']['min']:.0f}m - {data_info['depth_range']['max']:.0f}m"
            st.metric("Depth Range", depth_range)
        
        # Reset button
        if st.button("🔄 Reset to Default Data"):
            data_manager.reset_to_default()
            st.rerun()

if __name__ == "__main__":
    main()