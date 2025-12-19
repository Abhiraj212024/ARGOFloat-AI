# ============================
# RAG PIPELINE (V2 – FAST + ROBUST)
# ============================

import os
import logging
from functools import lru_cache
from typing import Dict, Any, Optional, List

import duckdb
import pandas as pd
import chromadb
from chromadb.config import Settings

from langchain_ollama import OllamaLLM
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent
# ----------------------------
# ENV + LOGGING
# ----------------------------
LANGCHAIN_API_KEY = os.getenv("LANGCHAIN_API_KEY")
if LANGCHAIN_API_KEY:
    os.environ["LANGCHAIN_API_KEY"] = LANGCHAIN_API_KEY
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

CHROMA_DIR = "./sql_query_vectors"

# ----------------------------
# VECTOR DB CLIENT (GLOBAL)
# ----------------------------
client = chromadb.PersistentClient(
    path=CHROMA_DIR,
    settings=Settings(allow_reset=False)
)

# ============================
# RAG PIPELINE
# ============================
class RAGPipeline:
    """
    Robust, optimized RAG pipeline for oceanographic data analysis.
    """

    def __init__(
        self,
        db_path: str = "./DB_files/data.duckdb",
        table_name: str = "ocean_profiles",
        collection_name: str = "duckdb_sql_queries",
    ):
        self.DB_PATH = str(BASE_DIR / "DB_files" / "data.duckdb")
        self.TABLE_NAME = table_name
        self.COLLECTION_NAME = collection_name

        # Persistent DB connection (READ ONLY)
        self.con = duckdb.connect(self.DB_PATH, read_only=True)

        # Models
        self.sql_llm = OllamaLLM(model="llama3:8b-instruct-q4_K_M")
        self.summary_llm = OllamaLLM(model="gemma:2b")

        # Schema + columns
        self.schema_text, self.allowed_columns = self._load_schema()

        # Vector collection
        self.collection = self._load_collection()

        logger.info("RAGPipeline initialized successfully")
        logger.info(f"Connected to DuckDB at {self.DB_PATH}")

    # ----------------------------
    # SCHEMA
    # ----------------------------
    def _load_schema(self):
        df = self.con.execute(f"DESCRIBE {self.TABLE_NAME}").fetchdf()
        schema_lines = []
        cols = set()

        for r in df.itertuples():
            schema_lines.append(f"- {r.column_name} ({r.column_type})")
            cols.add(r.column_name)

        return "Table schema:\n" + "\n".join(schema_lines), cols

    # ----------------------------
    # VECTOR COLLECTION
    # ----------------------------
    def _load_collection(self):
        existing = [c.name for c in client.list_collections()]
        if self.COLLECTION_NAME not in existing:
            logger.warning("Chroma collection not found")
            return None
        return client.get_collection(self.COLLECTION_NAME)

    # ----------------------------
    # VECTOR RETRIEVAL (CACHED)
    # ----------------------------
    @lru_cache(maxsize=128)
    def retrieve_examples(self, user_query: str) -> str:
        if not self.collection:
            return ""

        results = self.collection.query(
            query_texts=[user_query],
            n_results=3,
            include=["documents", "metadatas"]
        )

        examples = []
        for doc, meta in zip(results["documents"][0], results["metadatas"][0]):
            sql = meta.get("sql") or meta.get("sql_query")
            if sql:
                examples.append(f"Q: {doc}\nSQL: {sql}")

        return "\n\n".join(examples)

    # ----------------------------
    # SQL GENERATION (CACHED)
    # ----------------------------
    @lru_cache(maxsize=128)
    def generate_sql(self, user_query: str) -> str:
        examples = self.retrieve_examples(user_query)

        system_prompt = f"""
        You are an expert DuckDB SQL assistant for oceanographic analysis.

        {self.schema_text}

        Rules:
        - Use ONLY columns listed above
        - Always include: n_prof, n_levels, pres, temp, psal, latitude, longitude, time
        - Do NOT add latitude/longitude filters unless explicitly requested
        - Return ONLY valid DuckDB SQL (no markdown)

        Examples:
        {examples}
        """

        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", "{question}")
        ])

        chain = prompt | self.sql_llm | StrOutputParser()
        sql = chain.invoke({"question": user_query})
        return self._clean_sql(sql)

    # ----------------------------
    # SQL CLEAN + VALIDATE
    # ----------------------------
    def _clean_sql(self, sql: str) -> str:
        sql = sql.replace("```sql", "").replace("```", "").strip()
        return sql if sql.endswith(";") else sql + ";"

    def _validate_sql(self, sql: str) -> Optional[str]:
        lower = sql.lower()
        if any(x in lower for x in ["drop", "delete", "update", "insert"]):
            return "Destructive SQL operations are not allowed"
        return None

    # ----------------------------
    # SQL AUTO-REPAIR
    # ----------------------------
    def _repair_sql(self, user_query: str, bad_sql: str, error: str) -> str:
        system_prompt = f"""
        You are an expert DuckDB SQL assistant.

        Schema:
        {self.schema_text}

        User question:
        {user_query}

        SQL that failed:
        {bad_sql}

        Error:
        {error}

        Fix the SQL. Return ONLY corrected SQL.
        """
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", "{question}")
        ])

        chain = prompt | self.sql_llm | StrOutputParser()
        return self._clean_sql(chain.invoke({"question": user_query}))

    # ----------------------------
    # SQL EXECUTION
    # ----------------------------
    def run_sql(self, sql: str, user_query: str) -> pd.DataFrame:
        validation_error = self._validate_sql(sql)
        if validation_error:
            raise ValueError(validation_error)

        try:
            return self.con.execute(sql).fetchdf().dropna()
        except Exception as e:
            logger.warning(f"SQL failed, attempting repair: {e}")
            repaired = self._repair_sql(user_query, sql, str(e))
            return self.con.execute(repaired).fetchdf().dropna()

    # ----------------------------
    # FAST SUMMARY
    # ----------------------------
    def _fast_summary(self, user_query: str, df: pd.DataFrame) -> Optional[str]:
        q = user_query.lower()
        if df.empty:
            return "No data found for your query."

        if "average" in q or "avg" in q:
            return f"The average value is approximately {df.iloc[0,0]:.2f}."

        if "how many" in q or "count" in q:
            return f"There are {len(df)} records matching your query."

        return None

    # ----------------------------
    # LLM SUMMARY
    # ----------------------------
    def summarize_query(self, user_query: str, df: pd.DataFrame) -> str:
        fast = self._fast_summary(user_query, df)
        if fast:
            return fast

        context = df.head(20).to_markdown()
        prompt = f"""
        You are an oceanography assistant.

        User query:
        {user_query}

        Data:
        {context}

        Provide a concise factual summary using ONLY the data shown.
        """
        return self.summary_llm.invoke(prompt)

    # ----------------------------
    # PUBLIC API (DASHBOARD USES THESE)
    # ----------------------------
    def process_query(self, user_query: str) -> Dict[str, Any]:
        try:
            sql = self.generate_sql(user_query)
            data = self.run_sql(sql, user_query)
            answer = self.summarize_query(user_query, data)

            return {
                "success": True,
                "sql": sql,
                "data": data,
                "answer": answer,
                "error": None
            }
        except Exception as e:
            logger.error(f"Pipeline error: {e}")
            return {
                "success": False,
                "sql": "",
                "data": pd.DataFrame(),
                "answer": f"Error processing query: {e}",
                "error": str(e)
            }

    def test_connection(self) -> bool:
        try:
            self.con.execute("SELECT 1;")
            self.sql_llm.invoke("Hello")
            return True
        except Exception:
            return False

    def get_available_columns(self) -> List[str]:
        return list(self.allowed_columns)

    def get_sample_data(self, n: int = 5) -> pd.DataFrame:
        try:
            return self.con.execute(
                f"SELECT * FROM {self.TABLE_NAME} LIMIT {n}"
            ).fetchdf()
        except Exception as e:
            return pd.DataFrame({"Error": [str(e)]})

    def analyze_time_column_format(self) -> Dict[str, Any]:
        try:
            df = self.con.execute(
                f"SELECT time FROM {self.TABLE_NAME} LIMIT 10"
            ).fetchdf()

            samples = df["time"].astype(str).tolist()
            return {
                "samples": samples,
                "suggestions": [
                    "Try filtering by year (e.g., year 2015)",
                    "Use string matching on time column"
                ]
            }
        except Exception as e:
            return {"samples": [], "suggestions": [str(e)]}

# ----------------------------
# FACTORY
# ----------------------------
def create_rag_pipeline(
    db_path: str = "./DB_files/data.duckdb",
    table_name: str = "ocean_profiles"
) -> RAGPipeline:
    return RAGPipeline(db_path=db_path, table_name=table_name)
