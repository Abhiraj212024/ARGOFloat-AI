import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Optional, Union
from datetime import datetime, timedelta
import duckdb

logger = logging.getLogger(__name__)

class DataManager:
    """
    Manages visualization data for the FloatChat dashboard.
    Uses real database data instead of generating random samples.
    """
    
    def __init__(self, db_path: str = "./DB_files/data.duckdb", table_name: str = "ocean_profiles", default_limit: int = 1000):
        """
        Initialize DataManager with real data from database.
        
        Args:
            db_path: Path to the DuckDB database file
            table_name: Name of the table containing ocean data
            default_limit: Number of default data points to load
        """
        self.db_path = db_path
        self.table_name = table_name
        self.default_limit = default_limit
        self.current_data = None
        self.data_source = "default"  # "default" or "rag"
        self.last_updated = datetime.now()
        
        # Column mappings for your actual database schema
        self.column_mappings = {
            'latitude' : 'lat',
            'longitude' : 'lon', 
            'temp' : 'temperature',
            'psal' : 'salinity',
            'pres' : 'pres',
            'n_levels' : 'depth',
            'time' : 'time',
            'n_prof' : 'float_id'
        }
        
        # Required columns for different visualizations
        self.viz_requirements = {
            'map': ['lat', 'lon'],
            'profile': ['temperature', 'salinity', 'depth'],
            'timeseries': ['temperature', 'salinity', 'time']
        }
        
        # Initialize with real data from database
        self._load_default_data()
        logger.info(f"DataManager initialized with {len(self.current_data) if self.current_data is not None else 0} real data points")
    
    def _load_default_data(self):
        """Load first N data points from the actual database."""
        try:
            con = duckdb.connect(self.db_path)
            
            # Load first N records with all columns
            query = f"""
            SELECT 
                n_prof,
                n_levels,
                pres,
                temp, 
                psal,
                latitude,
                longitude,
                time
            FROM {self.table_name} 
            ORDER BY n_prof, pres  -- Order by profile and pressure/depth
            LIMIT {self.default_limit};
            """
            
            raw_data = con.execute(query).fetchdf()
            con.close()
            
            if raw_data.empty:
                logger.error("No data found in database table")
                self._create_minimal_fallback()
                return
            
            # Apply column mappings and enhancements
            self.current_data = self._standardize_dataframe(raw_data)
            self.data_source = "default"
            
            logger.info(f"Loaded {len(self.current_data)} real data points from database")
            
        except Exception as e:
            logger.error(f"Error loading default data from database: {e}")
            self._create_minimal_fallback()
    
    def _standardize_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Standardize column names and add missing columns needed for visualizations.
        
        Args:
            df: Raw dataframe from database or RAG
            
        Returns:
            Standardized dataframe
        """
        try:
            standardized_df = df.copy()
            
            # Apply column name mappings
            for old_name, new_name in self.column_mappings.items():
                if old_name in standardized_df.columns and new_name not in standardized_df.columns:
                    standardized_df[new_name] = standardized_df[old_name]
            
            # Add missing essential columns
            self._add_missing_viz_columns(standardized_df)
            
            # Add computed columns
            self._add_computed_columns(standardized_df)
            
            return standardized_df
            
        except Exception as e:
            logger.error(f"Error standardizing dataframe: {e}")
            return df
    
    def _add_missing_viz_columns(self, df: pd.DataFrame):
        """Add columns needed for visualizations that might be missing."""
        
        # Add depth column (use pressure as proxy if available)
        if 'depth' not in df.columns:
            if 'pressure' in df.columns:
                # Convert pressure (dbar) to depth (meters) approximately
                df['depth'] = df['pressure'] * 1.0  # 1 dbar ≈ 1 meter depth
            elif 'pres' in df.columns:
                df['depth'] = df['pres'] * 1.0
            else:
                # Default depth values if no pressure data
                df['depth'] = np.linspace(0, 2000, len(df))
        
        # Add float_id if missing (use n_prof if available)
        if 'float_id' not in df.columns:
            if 'n_prof' in df.columns:
                df['float_id'] = df['n_prof'].astype(str).apply(lambda x: f"PROF_{x}")
            else:
                # Generate sequential float IDs
                unique_locations = df.groupby(['lat', 'lon']).ngroup() + 1
                df['float_id'] = unique_locations.apply(lambda x: f"FLOAT_{x:04d}")
        
        # Add cycle column if missing (use n_levels or create sequence)
        if 'cycle' not in df.columns:
            if 'n_levels' in df.columns:
                df['cycle'] = df['n_levels']
            else:
                # Create cycle based on depth ordering within each float
                if 'float_id' in df.columns and 'depth' in df.columns:
                    df['cycle'] = df.groupby('float_id')['depth'].rank().astype(int)
                else:
                    df['cycle'] = range(1, len(df) + 1)
    
    
    def _add_computed_columns(self, df: pd.DataFrame):
        """Add computed columns that might be useful."""
        try:
            # Add region classification based on coordinates
            if 'lat' in df.columns and 'lon' in df.columns:
                df['region'] = df.apply(self._classify_ocean_region, axis=1)
            
            # Add depth category
            if 'depth' in df.columns:
                df['depth_category'] = pd.cut(df['depth'], 
                                            bins=[0, 200, 1000, 2000, float('inf')],
                                            labels=['Surface', 'Intermediate', 'Deep', 'Abyssal'])
                                            
        except Exception as e:
            logger.warning(f"Error adding computed columns: {e}")
    
    def _classify_ocean_region(self, row) -> str:
        """Classify ocean region based on lat/lon coordinates."""
        lat, lon = row['lat'], row['lon']
        
        # Simple ocean region classification
        if lat > 60:
            return "Arctic"
        elif lat < -40:
            return "Southern Ocean"
        elif -60 < lon < 20:  # Atlantic
            if lat > 0:
                return "North Atlantic"
            else:
                return "South Atlantic"
        elif 20 < lon < 110:  # Indian Ocean
            return "Indian Ocean"
        else:  # Pacific
            if lat > 0:
                return "North Pacific"
            else:
                return "South Pacific"
    
    def _create_minimal_fallback(self):
        """Create minimal fallback data if database connection fails."""
        logger.warning("Creating minimal fallback data due to database issues")
        
        fallback_data = pd.DataFrame({
            'lat': [10.0, 20.0, -15.0],
            'lon': [60.0, 80.0, -45.0],
            'temperature': [25.0, 15.0, 22.0],
            'salinity': [35.0, 35.1, 35.2],
            'depth': [0.0, 50.0, 100.0],
            'pressure': [0.0, 5.0, 10.0],
            'float_id': ['FALLBACK_001', 'FALLBACK_002', 'FALLBACK_003'],
            'cycle': [1, 2, 3],
            'time': [datetime.now().strftime('%Y-%m-%d')] * 3,
            'region': ['Fallback'] * 3
        })
        
        self.current_data = fallback_data
        self.data_source = "fallback"
    
    def update_from_rag_result(self, rag_result: Dict, query_type: str = "unknown") -> bool:
        """
        Update current data with RAG query results.
        
        Args:
            rag_result: Result dictionary from RAG pipeline
            query_type: Type of query that generated this data
            
        Returns:
            bool: True if data was successfully updated
        """
        try:
            if not rag_result.get("success", False):
                logger.warning("RAG result was not successful, keeping current data")
                return False
            
            new_data = rag_result.get("data", pd.DataFrame())
            
            if new_data.empty or "Error" in new_data.columns:
                logger.warning("RAG result contains no valid data, keeping current data")
                return False
            
            # Standardize the new data
            standardized_data = self._standardize_dataframe(new_data)
            
            if not standardized_data.empty:
                # Limit data size for performance
                if len(standardized_data) > 1000:
                    
                    standardized_data = standardized_data.head(1000)
                    logger.info(f"Limited RAG data to 1000 rows for performance")
                
                self.current_data = standardized_data
                self.data_source = "rag"
                self.last_updated = datetime.now()
                logger.info(f"Updated data from RAG result: {len(standardized_data)} rows")
                return True
            else:
                logger.warning("Standardized RAG data is empty, keeping current data")
                return False
                
        except Exception as e:
            logger.error(f"Error updating from RAG result: {e}")
            return False
    
    def get_current_data(self) -> pd.DataFrame:
        """Get current data for visualizations."""
        return self.current_data.copy() if self.current_data is not None else pd.DataFrame()
    
    def get_map_data(self) -> pd.DataFrame:
        """Get data optimized for map visualization (surface measurements only)."""
        data = self.get_current_data()
        if data.empty:
            return data
        
        # For map visualization, show surface measurements (shallowest depth for each profile)
        if 'depth' in data.columns and 'float_id' in data.columns:
            # Get shallowest measurement for each float
            surface_data = data.loc[data.groupby('float_id')['depth'].idxmin()]
            return surface_data.reset_index(drop=True)
        else:
            # If no depth info, remove duplicates by location
            return data.drop_duplicates(subset=['lat', 'lon']).reset_index(drop=True)
    
    def get_profile_data(self, float_id: str = None) -> pd.DataFrame:
        """
        Get data for profile visualization (depth vs temperature/salinity).
        
        Args:
            float_id: Specific float ID, or None for first available profile
            
        Returns:
            Profile data sorted by depth
        """
        data = self.get_current_data()
        if data.empty:
            return data
        
        # Get profile for specific float or first available
        if float_id and 'float_id' in data.columns and float_id in data['float_id'].values:
            profile_data = data[data['float_id'] == float_id].copy()
        else:
            # Get first float's complete profile
            if 'float_id' in data.columns:
                first_float = data['float_id'].iloc[0]
                profile_data = data[data['float_id'] == first_float].copy()
            else:
                profile_data = data.copy()
        
        # Sort by depth for proper profile visualization
        if 'depth' in profile_data.columns:
            profile_data = profile_data.sort_values('depth')
        
        return profile_data.reset_index(drop=True)
    
    def get_timeseries_data(self) -> pd.DataFrame:
        """Get data for time series visualization."""
        data = self.get_current_data()
        if data.empty:
            return data
        
        # Group by time and calculate means if we have time data
        if 'time' in data.columns:
            timeseries_data = data.groupby('time').agg({
                'temperature': 'mean',
                'salinity': 'mean',
                'depth': 'mean',
                'lat': 'mean',
                'lon': 'mean'
            }).reset_index()
            
            # Add cycle numbers based on time order
            timeseries_data = timeseries_data.sort_values('time')
            timeseries_data['cycle'] = range(1, len(timeseries_data) + 1)
            
        elif 'cycle' in data.columns:
            # Use cycle as time proxy and group by cycle
            timeseries_data = data.groupby('cycle').agg({
                'temperature': 'mean',
                'salinity': 'mean',
                'depth': 'mean'
            }).reset_index()
            
        else:
            # Create simple time series from data order
            timeseries_data = data.copy()
            timeseries_data['cycle'] = range(1, len(timeseries_data) + 1)
        
        return timeseries_data
    
    def get_data_info(self) -> Dict:
        """Get comprehensive information about current data."""
        data = self.get_current_data()
        
        info = {
            "source": self.data_source,
            "last_updated": self.last_updated,
            "row_count": len(data) if not data.empty else 0,
            "columns": list(data.columns) if not data.empty else [],
            "float_count": data['float_id'].nunique() if 'float_id' in data.columns else 0,
            "date_range": None,
            "depth_range": None,
            "temp_range": None,
            "has_location": all(col in data.columns for col in ['lat', 'lon']),
            "has_depth": 'depth' in data.columns,
            "has_temperature": 'temperature' in data.columns,
            "has_salinity": 'salinity' in data.columns,
            "has_time": 'time' in data.columns
        }
        
        if not data.empty:
            # Date range
            if 'time' in data.columns:
                try:
                    time_data = pd.to_datetime(data['time'], errors='coerce').dropna()
                    if not time_data.empty:
                        info["date_range"] = {
                            "start": time_data.min().strftime('%Y-%m-%d'),
                            "end": time_data.max().strftime('%Y-%m-%d')
                        }
                except:
                    pass
            
            # Depth range
            if 'depth' in data.columns:
                depth_data = data['depth'].dropna()
                if not depth_data.empty:
                    info["depth_range"] = {
                        "min": float(depth_data.min()),
                        "max": float(depth_data.max())
                    }
            
            # Temperature range
            if 'temperature' in data.columns:
                temp_data = data['temperature'].dropna()
                if not temp_data.empty:
                    info["temp_range"] = {
                        "min": float(temp_data.min()),
                        "max": float(temp_data.max())
                    }
        
        return info
    
    def reset_to_default(self):
        """Reset to default data from database."""
        self._load_default_data()
        logger.info("Data reset to default (first 100 database records)")
    
    def is_suitable_for_visualization(self, viz_type: str) -> tuple[bool, str]:
        """
        Check if current data is suitable for a specific visualization.
        
        Args:
            viz_type: 'map', 'profile', or 'timeseries'
            
        Returns:
            Tuple of (is_suitable, reason)
        """
        data = self.get_current_data()
        
        if data.empty:
            return False, "No data available"
        
        required_cols = self.viz_requirements.get(viz_type, [])
        missing_cols = [col for col in required_cols if col not in data.columns]
        
        if missing_cols:
            return False, f"Missing required columns: {', '.join(missing_cols)}"
        
        return True, f"✓ {viz_type.title()} visualization ready"
    
    def refresh_default_data(self, limit: int = None):
        """
        Refresh default data from database with optional new limit.
        
        Args:
            limit: New limit for default data, or None to use existing limit
        """
        if limit:
            self.default_limit = limit
        
        self._load_default_data()
        