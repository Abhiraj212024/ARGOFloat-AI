import os
import sys
import chromadb
from chromadb.config import Settings

# Path to the persistent Chroma vector database
CHROMA_DIR = "./sql_query_vectors"

# Possible collection names to look for
COLLECTION_CANDIDATES = ["duckdb_sql_queries", "sql_queries"]


def open_collection(client: chromadb.PersistentClient):
    # Prefer existing collections; avoid creating a new one inadvertently
    try:
        existing = [c.name for c in client.list_collections()]
    except Exception:
        existing = []

    for name in COLLECTION_CANDIDATES:
        if name in existing:
            try:
                return client.get_collection(name)
            except Exception:
                pass

    # Fallback: try to get by name in case list_collections is unavailable
    for name in COLLECTION_CANDIDATES:
        try:
            return client.get_collection(name)
        except Exception:
            continue

    return None


def main():
    # Ensure vector DB directory exists
    if not os.path.exists(CHROMA_DIR):
        print(f"❌ Vector DB directory not found: {CHROMA_DIR}")
        sys.exit(1)

    # Open Chroma persistent client
    client = chromadb.PersistentClient(path=CHROMA_DIR, settings=Settings(allow_reset=True))

    # Open existing collection
    col = open_collection(client)
    if col is None:
        print(f"❌ No expected collection found. Looked for: {', '.join(COLLECTION_CANDIDATES)}")
        sys.exit(1)

    # Fetch all items (documents and metadatas); IDs are always returned
    data = col.get(include=["documents", "metadatas"])
    ids = data.get("ids", []) or []
    docs = data.get("documents", []) or []
    metas = data.get("metadatas", []) or []

    if not ids:
        print(f"Collection '{col.name}' is empty.")
        return

    # Build list and sort by numeric suffix in the ID (e.g., query_42)
    items = []
    for i, _id in enumerate(ids):
        doc = docs[i] if i < len(docs) else None
        meta = metas[i] if i < len(metas) else {}
        try:
            numeric_idx = int(str(_id).split("_")[-1])
        except Exception:
            numeric_idx = i  # fallback
        items.append((numeric_idx, _id, doc, meta))

    items.sort(key=lambda x: x[0])  # ascending by numeric index
    last10 = items[-10:]

    print(f"Last {len(last10)} query pairs in collection '{col.name}':")
    for _, _id, doc, meta in last10:
        sql = meta.get("sql_query") if isinstance(meta, dict) else None
        print("-" * 80)
        print(f"ID: {_id}")
        print(f"Natural language: {doc}")
        if sql:
            print(f"SQL: {sql}")

    # Optional: show total count
    try:
        count = col.count()
        print(f"\nTotal items in collection: {count}")
    except Exception:
        pass

if __name__ == "__main__":
    main()
