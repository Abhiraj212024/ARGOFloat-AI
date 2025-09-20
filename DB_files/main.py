import pandas as pd
import duckdb
import chromadb
from chromadb.utils import embedding_functions

# Connect to DuckDB and fetch data
con = duckdb.connect('data.duckdb')
df = con.execute("SELECT * FROM ocean_profiles").fetchdf()
con.close()

# Initialize ChromaDB client and collection
client = chromadb.Client()
collection = client.create_collection(name="ocean_profiles")

# Use a simple embedding function (for demonstration)
try:
    embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")
except Exception as e:
    print("Error: sentence_transformers package is not installed.")
    print("Please run: pip install sentence-transformers")
    exit(1)

# Prepare documents and metadata
documents = df.astype(str).apply(lambda row: " ".join(row), axis=1).tolist()
metadatas = df.to_dict(orient='records')
ids = [str(i) for i in range(len(documents))]

# Add documents to ChromaDB
collection.add(
    documents=documents,
    metadatas=metadatas,
    ids=ids,
    embedding_function=embedding_fn
)

print("Vector DB created with ChromaDB using ocean_profiles data.")
