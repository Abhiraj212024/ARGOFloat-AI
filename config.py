# This file will contain the main workflow when everything is completed
"""
Configuration file for FloatChat Dashboard and RAG Pipeline
"""

import os
from pathlib import Path

# Database Configuration
DB_CONFIG = {
    "db_path": "./DB_files/data.duckdb",
    "table_name": "ocean_profiles",
    "connection_timeout": 30
}

# LLM Configuration
LLM_CONFIG = {
    "model_name": "llama3:8b-instruct-q4_K_M",
    "temperature": 0.1,
    "max_tokens": 2000,
    "timeout": 60
}

# LangChain Configuration
LANGCHAIN_CONFIG = {
    "tracing": True,
    "endpoint": "https://api.smith.langchain.com",
    "api_key_env": "LANGCHAIN_API_KEY"
}

# Streamlit Configuration
STREAMLIT_CONFIG = {
    "page_title": "FloatChat Dashboard",
    "page_icon": "🌊",
    "layout": "wide",
    "initial_sidebar_state": "expanded"
}

# Query Classification Keywords
QUERY_KEYWORDS = {
    "map": ["map", "location", "geographic", "geography", "where", "lat", "lon", "coordinate", "position"],
    "profile": ["profile", "depth", "vertical", "deep", "shallow", "meter", "pressure", "vs depth"],
    "timeseries": ["time", "cycle", "temporal", "trend", "over time", "series", "timeline", "historical"]
}

# Visualization Configuration
VIZ_CONFIG = {
    "map": {
        "projection": "natural earth",
        "height": 600,
        "size_max": 15
    },
    "profile": {
        "height": 600,
        "line_width": 2,
        "marker_size": 6
    },
    "timeseries": {
        "height": 500,
        "line_width": 2,
        "marker_size": 6
    }
}

# Sample Data Configuration
SAMPLE_DATA = {
    "use_sample": True,
    "sample_size": 5,
    "data": {
        "lat": [10.5, 20.3, -15.7, 30.2, 5.1],
        "lon": [60.8, 80.1, -45.3, 120.7, 75.4],
        "temperature": [28.5, 15.2, 22.1, 18.9, 26.3],
        "cycle": [1, 2, 3, 4, 5],
        "float_id": ["F001", "F002", "F003", "F004", "F005"],
        "depth": [0, 50, 100, 150, 200],
        "salinity": [35.1, 35.2, 35.0, 35.3, 35.1]
    }
}

# Error Messages
ERROR_MESSAGES = {
    "rag_unavailable": "RAG pipeline is not available. Please check your configuration.",
    "db_connection": "Failed to connect to database. Please check the database path and permissions.",
    "llm_connection": "Failed to initialize language model. Please ensure Ollama is running.",
    "missing_columns": "Required columns are missing from the dataset.",
    "no_data": "No data available for this visualization.",
    "query_failed": "Query execution failed. Please try rephrasing your question."
}

# Logging Configuration
LOGGING_CONFIG = {
    "level": "INFO",
    "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    "file_path": "./logs/floatchat.log"
}

# File Paths
PATHS = {
    "logs": Path("./logs"),
    "db": Path("./DB_files"),
    "cache": Path("./.cache"),
    "config": Path("./config")
}

def setup_environment():
    """Setup environment variables and create necessary directories."""
    # Set LangChain environment variables
    if LANGCHAIN_CONFIG["tracing"]:
        os.environ['LANGCHAIN_TRACING_V2'] = 'true'
        os.environ['LANGCHAIN_ENDPOINT'] = LANGCHAIN_CONFIG["endpoint"]
    
    # Create necessary directories
    for path in PATHS.values():
        path.mkdir(exist_ok=True)
    
    return True

def validate_config():
    """Validate configuration settings."""
    errors = []
    
    # Check database path
    db_path = Path(DB_CONFIG["db_path"])
    if not db_path.parent.exists():
        errors.append(f"Database directory does not exist: {db_path.parent}")
    
    # Check required environment variables
    if LANGCHAIN_CONFIG["api_key_env"] and not os.getenv(LANGCHAIN_CONFIG["api_key_env"]):
        errors.append(f"Environment variable not set: {LANGCHAIN_CONFIG['api_key_env']}")
    
    return errors

def get_database_config():
    """Get database configuration."""
    return DB_CONFIG.copy()

def get_llm_config():
    """Get LLM configuration."""
    return LLM_CONFIG.copy()

def get_streamlit_config():
    """Get Streamlit configuration."""
    return STREAMLIT_CONFIG.copy()

def get_sample_data():
    """Get sample data configuration."""
    return SAMPLE_DATA.copy()

# Initialize on import
if __name__ != "__main__":
    setup_environment()