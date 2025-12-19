import time
import pandas as pd
import numpy as np
from typing import Dict, List
import duckdb
from rag import create_rag_pipeline
import json
from datetime import datetime

class ProjectBenchmark:
    """Benchmark suite for FloatChat project to quantify impact."""
    
    def __init__(self, db_path: str = "./DB_files/data.duckdb"):
        self.db_path = db_path
        self.results = {
            "database_metrics": {},
            "query_performance": {},
            "rag_performance": {},
            "comparison_metrics": {}
        }
    
    # ==========================================
    # 1. DATABASE METRICS
    # ==========================================
    
    def measure_database_stats(self):
        """Measure database size, compression, and data volume."""
        import os
        
        con = duckdb.connect(self.db_path)
        
        # Get file size
        db_size_mb = os.path.getsize(self.db_path) / (1024 * 1024)
        
        # Get row count
        row_count = con.execute("SELECT COUNT(*) FROM ocean_profiles").fetchone()[0]
        
        # Get unique floats
        float_count = con.execute("SELECT COUNT(DISTINCT n_prof) FROM ocean_profiles").fetchone()[0]
        
        # Get date range
        date_range = con.execute("SELECT MIN(time), MAX(time) FROM ocean_profiles").fetchone()
        
        # Get column count
        columns = con.execute("DESCRIBE ocean_profiles").fetchdf()
        col_count = len(columns)
        
        # Estimate original data size (assume 50GB as mentioned)
        original_size_gb = 50
        compression_ratio = (original_size_gb * 1024) / db_size_mb
        
        con.close()
        
        self.results["database_metrics"] = {
            "processed_db_size_mb": round(db_size_mb, 2),
            "original_data_size_gb": original_size_gb,
            "compression_ratio": round(compression_ratio, 2),
            "compression_percentage": round((1 - db_size_mb/(original_size_gb*1024)) * 100, 2),
            "total_records": row_count,
            "unique_floats": float_count,
            "columns": col_count,
            "date_range": {
                "start": str(date_range[0]),
                "end": str(date_range[1])
            }
        }
        
        return self.results["database_metrics"]
    
    # ==========================================
    # 2. QUERY PERFORMANCE BENCHMARKS
    # ==========================================
    
    def benchmark_sql_queries(self, num_runs: int = 10) -> Dict:
        """Benchmark direct SQL query performance."""
        
        test_queries = [
            ("Simple SELECT", "SELECT * FROM ocean_profiles LIMIT 1000"),
            ("Aggregation", "SELECT AVG(temp), AVG(psal) FROM ocean_profiles"),
            ("Filtering", "SELECT * FROM ocean_profiles WHERE temp > 20 LIMIT 1000"),
            ("Grouping", "SELECT n_prof, AVG(temp) FROM ocean_profiles GROUP BY n_prof LIMIT 100"),
            ("Complex Join-like", "SELECT * FROM ocean_profiles WHERE latitude BETWEEN -10 AND 10 AND temp > 25 LIMIT 1000")
        ]
        
        results = {}
        con = duckdb.connect(self.db_path)
        
        for query_name, sql in test_queries:
            times = []
            
            for _ in range(num_runs):
                start = time.time()
                df = con.execute(sql).fetchdf()
                end = time.time()
                times.append(end - start)
            
            results[query_name] = {
                "avg_time_ms": round(np.mean(times) * 1000, 2),
                "min_time_ms": round(np.min(times) * 1000, 2),
                "max_time_ms": round(np.max(times) * 1000, 2),
                "std_dev_ms": round(np.std(times) * 1000, 2)
            }
        
        con.close()
        self.results["query_performance"]["sql_direct"] = results
        return results
    
    # ==========================================
    # 3. RAG PIPELINE PERFORMANCE
    # ==========================================
    
    def benchmark_rag_pipeline(self, num_runs: int = 5) -> Dict:
        """Benchmark end-to-end RAG query performance."""
        
        test_queries = [
            "What's the average temperature?",
            "Show me salinity data above 35 PSU",
            "Find measurements in the Pacific Ocean",
            "Get temperature profiles deeper than 1000m",
            "How many floats do we have?"
        ]
        
        rag = create_rag_pipeline(self.db_path)
        results = {}
        
        for query in test_queries:
            times = {
                "total": [],
                "sql_generation": [],
                "sql_execution": [],
                "summarization": []
            }
            
            for _ in range(num_runs):
                # Measure SQL generation
                start = time.time()
                sql = rag.generate_sql(query)
                sql_gen_time = time.time() - start
                
                # Measure SQL execution
                start = time.time()
                data = rag.run_sql(sql, query)
                sql_exec_time = time.time() - start
                
                # Measure summarization
                start = time.time()
                summary = rag.summarize_query(query, data)
                summary_time = time.time() - start
                
                total_time = sql_gen_time + sql_exec_time + summary_time
                
                times["total"].append(total_time)
                times["sql_generation"].append(sql_gen_time)
                times["sql_execution"].append(sql_exec_time)
                times["summarization"].append(summary_time)
            
            results[query] = {
                "avg_total_time_sec": round(np.mean(times["total"]), 3),
                "avg_sql_generation_sec": round(np.mean(times["sql_generation"]), 3),
                "avg_sql_execution_sec": round(np.mean(times["sql_execution"]), 3),
                "avg_summarization_sec": round(np.mean(times["summarization"]), 3),
                "breakdown_percentages": {
                    "sql_generation": round(np.mean(times["sql_generation"]) / np.mean(times["total"]) * 100, 1),
                    "sql_execution": round(np.mean(times["sql_execution"]) / np.mean(times["total"]) * 100, 1),
                    "summarization": round(np.mean(times["summarization"]) / np.mean(times["total"]) * 100, 1)
                }
            }
        
        self.results["rag_performance"] = results
        return results
    
    # ==========================================
    # 4. VECTOR DB PERFORMANCE
    # ==========================================
    
    def benchmark_vector_search(self, num_runs: int = 20) -> Dict:
        """Benchmark ChromaDB similarity search performance."""
        import chromadb
        
        client = chromadb.PersistentClient(path="./sql_query_vectors")
        collection = client.get_collection("duckdb_sql_queries")
        
        test_queries = [
            "show temperature data",
            "find deep ocean measurements",
            "get salinity trends over time",
            "average temperature by region",
            "profiles in tropical waters"
        ]
        
        results = {}
        
        for query in test_queries:
            times = []
            
            for _ in range(num_runs):
                start = time.time()
                results_search = collection.query(
                    query_texts=[query],
                    n_results=5
                )
                end = time.time()
                times.append(end - start)
            
            results[query] = {
                "avg_search_time_ms": round(np.mean(times) * 1000, 2),
                "min_time_ms": round(np.min(times) * 1000, 2),
                "max_time_ms": round(np.max(times) * 1000, 2)
            }
        
        self.results["query_performance"]["vector_search"] = results
        return results
    
    # ==========================================
    # 5. COMPARISON WITH MANUAL APPROACH
    # ==========================================
    
    def estimate_manual_analysis_time(self) -> Dict:
        """Estimate time saved compared to manual CSV analysis."""
        
        # Assumptions for manual analysis
        manual_steps = {
            "load_csv": 180,  # 3 minutes to load 50GB CSV
            "filter_data": 120,  # 2 minutes to filter
            "calculate_stats": 60,  # 1 minute to calculate
            "create_visualization": 300,  # 5 minutes to create viz
            "total": 660  # 11 minutes total
        }
        
        # FloatChat average time (from RAG benchmarks)
        floatchat_avg = 5  # seconds (estimate)
        
        time_saved_per_query = manual_steps["total"] - floatchat_avg
        speedup_factor = manual_steps["total"] / floatchat_avg
        
        # Estimate queries per r