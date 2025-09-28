import chromadb

def query_collections(client, collection_names, user_query, threshold=0.5, top_k=5):
    #cosine similarity < threshold
    matches = []

    # get list of actually existing collections
    existing = [c.name for c in client.list_collections()]

    for col_name in collection_names:
        if col_name not in existing:
            print(f"⚠️ Skipping missing collection: {col_name}")
            continue

        collection = client.get_collection(col_name)

        results = collection.query(
            query_texts=[user_query],
            n_results=top_k,
            include=["metadatas", "documents", "distances"]
        )

        documents = results["documents"][0]
        metadatas = results["metadatas"][0]
        distances = results["distances"][0]

        for doc, meta, dist in zip(documents, metadatas, distances):
            if dist <= threshold:
                matches.append({
                    "collection": col_name,
                    "nl_query": doc,
                    "sql_query": meta.get("sql") or meta.get("sql_query"),
                    "distance": dist
                })

    return sorted(matches, key=lambda x: x["distance"])



client = chromadb.PersistentClient(path="./sql_query_vectors", settings=chromadb.config.Settings(allow_reset=True))

COLLECTION_CANDIDATES = ["duckdb_sql_queries"]


user_query = "get annual salinity trends"

results = query_collections(client, COLLECTION_CANDIDATES, user_query, threshold=1, top_k=5)

if results:
    for r in results:
        print(f"[{r['collection']}] NL: {r['nl_query']} -> SQL: {r['sql_query']} (distance={r['distance']:.4f})")
else:
    print("No matches found above threshold.")
