import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
import numpy as np

# -----------------------------
# CONFIGURATION
# -----------------------------
CHROMA_DIR = "./sql_query_vectors"   # <-- persistent vector DB folder
TEXT_MODEL = "all-MiniLM-L6-v2"
# -----------------------------

# Step 1: Define 50 SQL query-natural language pairs
sql_query_pairs = [
    {
        "natural_language": "Show me all ocean profiles with temperature above 20 degrees Celsius",
        "sql_query": "SELECT * FROM ocean_profiles WHERE temp > 20"
    },
    {
        "natural_language": "Find profiles in the Pacific Ocean with high salinity",
        "sql_query": "SELECT * FROM ocean_profiles WHERE longitude BETWEEN -180 AND -70 AND psal > 35"
    },
    {
        "natural_language": "Get all measurements from the last 30 days",
        "sql_query": "SELECT * FROM ocean_profiles WHERE time >= CURRENT_DATE - INTERVAL 30 DAY"
    },
    {
        "natural_language": "Show profiles with more than 100 depth levels",
        "sql_query": "SELECT * FROM ocean_profiles WHERE n_levels > 100"
    },
    {
        "natural_language": "Find the deepest measurements with pressure above 2000 dbar",
        "sql_query": "SELECT * FROM ocean_profiles WHERE pres > 2000 ORDER BY pres DESC"
    },
    {
        "natural_language": "Get average temperature for each latitude band",
        "sql_query": "SELECT ROUND(latitude/10)*10 AS lat_band, AVG(temp) FROM ocean_profiles GROUP BY lat_band"
    },
    {
        "natural_language": "Show profiles in the Arctic region with low temperature",
        "sql_query": "SELECT * FROM ocean_profiles WHERE latitude > 66.5 AND temp < 5"
    },
    {
        "natural_language": "Find all profiles collected in 2023",
        "sql_query": "SELECT * FROM ocean_profiles WHERE EXTRACT(YEAR FROM time) = 2023"
    },
    {
        "natural_language": "Get the maximum depth for each profile",
        "sql_query": "SELECT n_prof, MAX(pres) as max_depth FROM ocean_profiles GROUP BY n_prof"
    },
    {
        "natural_language": "Show profiles with abnormally high salinity above 40 PSU",
        "sql_query": "SELECT * FROM ocean_profiles WHERE psal > 40"
    },
    {
        "natural_language": "Find profiles near the equator between 10N and 10S latitude",
        "sql_query": "SELECT * FROM ocean_profiles WHERE latitude BETWEEN -10 AND 10"
    },
    {
        "natural_language": "Get count of profiles by month for this year",
        "sql_query": "SELECT EXTRACT(MONTH FROM time) as month, COUNT(*) FROM ocean_profiles WHERE EXTRACT(YEAR FROM time) = 2024 GROUP BY month"
    },
    {
        "natural_language": "Show profiles with temperature between 15 and 25 degrees",
        "sql_query": "SELECT * FROM ocean_profiles WHERE temp BETWEEN 15 AND 25"
    },
    {
        "natural_language": "Find the coldest ocean measurements",
        "sql_query": "SELECT * FROM ocean_profiles ORDER BY temp ASC LIMIT 10"
    },
    {
        "natural_language": "Get profiles in the Mediterranean Sea region",
        "sql_query": "SELECT * FROM ocean_profiles WHERE latitude BETWEEN 30 AND 46 AND longitude BETWEEN -6 AND 36"
    },
    {
        "natural_language": "Show profiles with missing temperature data",
        "sql_query": "SELECT * FROM ocean_profiles WHERE temp IS NULL"
    },
    {
        "natural_language": "Find profiles collected during summer months",
        "sql_query": "SELECT * FROM ocean_profiles WHERE EXTRACT(MONTH FROM time) IN (6, 7, 8)"
    },
    {
        "natural_language": "Get average salinity for each ocean basin",
        "sql_query": "SELECT CASE WHEN longitude BETWEEN -180 AND -30 THEN 'Atlantic' WHEN longitude BETWEEN -30 AND 120 THEN 'Indian' ELSE 'Pacific' END as basin, AVG(psal) FROM ocean_profiles GROUP BY basin"
    },
    {
        "natural_language": "Show the most recent 100 profiles",
        "sql_query": "SELECT * FROM ocean_profiles ORDER BY time DESC LIMIT 100"
    },
    {
        "natural_language": "Find profiles with shallow measurements under 100 dbar pressure",
        "sql_query": "SELECT * FROM ocean_profiles WHERE pres < 100"
    },
    {
        "natural_language": "Get temperature statistics by depth ranges",
        "sql_query": "SELECT CASE WHEN pres < 500 THEN 'shallow' WHEN pres < 2000 THEN 'mid' ELSE 'deep' END as depth_range, AVG(temp), MIN(temp), MAX(temp) FROM ocean_profiles GROUP BY depth_range"
    },
    {
        "natural_language": "Show profiles in the Southern Ocean below 60S latitude",
        "sql_query": "SELECT * FROM ocean_profiles WHERE latitude < -60"
    },
    {
        "natural_language": "Find profiles with exactly 50 depth levels",
        "sql_query": "SELECT * FROM ocean_profiles WHERE n_levels = 50"
    },
    {
        "natural_language": "Get daily profile counts for the last week",
        "sql_query": "SELECT DATE(time) as day, COUNT(*) FROM ocean_profiles WHERE time >= CURRENT_DATE - INTERVAL 7 DAY GROUP BY day ORDER BY day"
    },
    {
        "natural_language": "Show profiles with high pressure measurements above 5000 dbar",
        "sql_query": "SELECT * FROM ocean_profiles WHERE pres > 5000"
    },
    {
        "natural_language": "Find profiles in tropical waters between 23N and 23S",
        "sql_query": "SELECT * FROM ocean_profiles WHERE latitude BETWEEN -23.5 AND 23.5"
    },
    {
        "natural_language": "Get the oldest profile in the database",
        "sql_query": "SELECT * FROM ocean_profiles ORDER BY time ASC LIMIT 1"
    },
    {
        "natural_language": "Show profiles with low salinity under 30 PSU",
        "sql_query": "SELECT * FROM ocean_profiles WHERE psal < 30"
    },
    {
        "natural_language": "Find profiles collected on weekends",
        "sql_query": "SELECT * FROM ocean_profiles WHERE EXTRACT(DOW FROM time) IN (0, 6)"
    },
    {
        "natural_language": "Get temperature range for each profile",
        "sql_query": "SELECT n_prof, MAX(temp) - MIN(temp) as temp_range FROM ocean_profiles GROUP BY n_prof"
    },
    {
        "natural_language": "Show profiles in the North Atlantic above 40N latitude",
        "sql_query": "SELECT * FROM ocean_profiles WHERE latitude > 40 AND longitude BETWEEN -80 AND 0"
    },
    {
        "natural_language": "Find profiles with complete data (no null values)",
        "sql_query": "SELECT * FROM ocean_profiles WHERE temp IS NOT NULL AND psal IS NOT NULL AND pres IS NOT NULL"
    },
    {
        "natural_language": "Get hourly profile distribution",
        "sql_query": "SELECT EXTRACT(HOUR FROM time) as hour, COUNT(*) FROM ocean_profiles GROUP BY hour ORDER BY hour"
    },
    {
        "natural_language": "Show profiles with moderate temperature between 10 and 20 degrees",
        "sql_query": "SELECT * FROM ocean_profiles WHERE temp BETWEEN 10 AND 20"
    },
    {
        "natural_language": "Find the saltiest measurements in the database",
        "sql_query": "SELECT * FROM ocean_profiles ORDER BY psal DESC LIMIT 10"
    },
    {
        "natural_language": "Get profiles collected during winter months in Northern Hemisphere",
        "sql_query": "SELECT * FROM ocean_profiles WHERE EXTRACT(MONTH FROM time) IN (12, 1, 2) AND latitude > 0"
    },
    {
        "natural_language": "Show profiles with medium depth levels between 50 and 150",
        "sql_query": "SELECT * FROM ocean_profiles WHERE n_levels BETWEEN 50 AND 150"
    },
    {
        "natural_language": "Find profiles in coastal waters near continents",
        "sql_query": "SELECT * FROM ocean_profiles WHERE (longitude BETWEEN -130 AND -110 AND latitude BETWEEN 20 AND 50) OR (longitude BETWEEN -10 AND 10 AND latitude BETWEEN 35 AND 65)"
    },
    {
        "natural_language": "Get annual temperature trends",
        "sql_query": "SELECT EXTRACT(YEAR FROM time) as year, AVG(temp) FROM ocean_profiles GROUP BY year ORDER BY year"
    },
    {
        "natural_language": "Show profiles with anomalous pressure readings",
        "sql_query": "SELECT * FROM ocean_profiles WHERE pres < 0 OR pres > 10000"
    },
    {
        "natural_language": "Find profiles collected at midnight",
        "sql_query": "SELECT * FROM ocean_profiles WHERE EXTRACT(HOUR FROM time) = 0"
    },
    {
        "natural_language": "Get salinity statistics by geographic regions",
        "sql_query": "SELECT CASE WHEN latitude > 0 THEN 'Northern' ELSE 'Southern' END as hemisphere, AVG(psal), STDDEV_SAMP(psal) FROM ocean_profiles GROUP BY hemisphere"
    },
    {
        "natural_language": "Show profiles with few depth measurements under 20 levels",
        "sql_query": "SELECT * FROM ocean_profiles WHERE n_levels < 20"
    },
    {
        "natural_language": "Find profiles in the Indian Ocean region",
        "sql_query": "SELECT * FROM ocean_profiles WHERE longitude BETWEEN 20 AND 147 AND latitude BETWEEN -70 AND 30"
    },
    {
        "natural_language": "Get monthly temperature averages",
        "sql_query": "SELECT EXTRACT(MONTH FROM time) as month, AVG(temp) FROM ocean_profiles GROUP BY month ORDER BY month"
    },
    {
        "natural_language": "Show profiles with extreme salinity values outside normal range",
        "sql_query": "SELECT * FROM ocean_profiles WHERE psal < 30 OR psal > 40"
    },
    {
        "natural_language": "Find the deepest ocean profile measurements",
        "sql_query": "SELECT * FROM ocean_profiles ORDER BY pres DESC LIMIT 5"
    },
    {
        "natural_language": "Get profile count by latitude bands",
        "sql_query": "SELECT FLOOR(latitude/10)*10 as lat_band, COUNT(*) FROM ocean_profiles GROUP BY lat_band ORDER BY lat_band"
    },
    {
        "natural_language": "Show profiles collected in the current month",
        "sql_query": "SELECT * FROM ocean_profiles WHERE EXTRACT(YEAR FROM time) = EXTRACT(YEAR FROM CURRENT_DATE) AND EXTRACT(MONTH FROM time) = EXTRACT(MONTH FROM CURRENT_DATE)"
    },
    {
        "natural_language": "Find profiles with consistent temperature throughout depth",
        "sql_query": "SELECT n_prof, STDDEV_SAMP(temp) as temp_variation FROM ocean_profiles GROUP BY n_prof HAVING STDDEV_SAMP(temp) < 1"
    }
]

print(f"📝 Created {len(sql_query_pairs)} SQL query-natural language pairs")

# Step 2: Create Chroma persistent client
client = chromadb.PersistentClient(path=CHROMA_DIR, settings=Settings(allow_reset=True))

# Step 3: Store SQL queries with natural language embeddings
print("📦 Computing embeddings for natural language queries...")
model = SentenceTransformer(TEXT_MODEL)

# Prepare data
natural_language_texts = [pair["natural_language"] for pair in sql_query_pairs]
sql_queries = [pair["sql_query"] for pair in sql_query_pairs]
ids = [f"query_{i}" for i in range(len(sql_query_pairs))]

# Create metadata with SQL queries (explicitly note DuckDB dialect)
metadata = [{"sql_query": sql, "engine": "duckdb"} for sql in sql_queries]

# Compute embeddings
embeddings = model.encode(natural_language_texts, convert_to_numpy=True).tolist()

# Create collection and add data
col_sql = client.get_or_create_collection("duckdb_sql_queries")
col_sql.add(
    ids=ids,
    documents=natural_language_texts,
    embeddings=embeddings,
    metadatas=metadata
)

print(f"✅ Stored {len(sql_query_pairs)} SQL query pairs")

print(f"\n🎉 Done! SQL query vector database saved at {CHROMA_DIR}")
print("Collection 'duckdb_sql_queries' contains natural language questions with corresponding DuckDB SQL queries")
print("\nExample usage:")
print("- Query: 'Find warm ocean water'")
print("- Will return similar queries like 'Show me all ocean profiles with temperature above 20 degrees'")
print("- Along with the SQL: 'SELECT * FROM ocean_profiles WHERE temp > 20'")
