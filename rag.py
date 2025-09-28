#Importing necessary libraries and defining environment variables for LangChain, API keys, and endpoints.

import os
os.environ['LANGCHAIN_TRACING_V2'] = 'true'
os.environ['LANGCHAIN_ENDPOINT'] = 'https://api.smith.langchain.com'
LANGCHAIN_API_KEY = os.getenv('LANGCHAIN_API_KEY')

import duckdb
import pandas as pd
from langchain_ollama import OllamaLLM
from langchain.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda
from langchain_core.output_parsers import StrOutputParser
import json
import chromadb
from typing import Dict, Any, Optional
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RAGPipeline:
    """
    Retrieval-Augmented Generation pipeline for oceanographic data queries.
    Handles natural language to SQL conversion and result summarization.
    """
    
    def __init__(self, db_path: str = "./DB_files/data.duckdb", table_name: str = "ocean_profiles", vector_db_path: str = "./sql_query_vectors"):
        """
        Initialize the RAG pipeline.
        
        Args:
            db_path: Path to the DuckDB database file
            table_name: Name of the main table to query
        """
        self.DB_PATH = db_path
        self.TABLE_NAME = table_name
        self.VECTOR_DB_PATH = vector_db_path
        self.llm = None
        self.SCHEMA_TEXT = ""
        
        # Initialize components
        self._initialize_llm()
        self._load_schema()
    
    def _initialize_llm(self):
        """Initialize the language model."""
        try:
            self.llm = OllamaLLM(model="llama3:8b-instruct-q4_K_M")
            logger.info("LLM initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize LLM: {e}")
            raise
    
    def _load_schema(self):
        """Load and format database schema."""
        try:
            self.SCHEMA_TEXT = self.get_schema_text(self.DB_PATH, self.TABLE_NAME)
            logger.info("Database schema loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load schema: {e}")
            self.SCHEMA_TEXT = "Schema unavailable"
    
    def get_schema_text(self, db_path: str, table_name: str) -> str:
        """
        Extract and format database schema information.
        
        Args:
            db_path: Path to database
            table_name: Table name to describe
            
        Returns:
            Formatted schema text
        """
        try:
            con = duckdb.connect(db_path)
            columns = con.execute(f"DESCRIBE {table_name};").fetchdf()
            con.close()
            
            schema_text = f"Table: {table_name}\n"
            for row in columns.itertuples():
                schema_text += f"  - {row.column_name} ({row.column_type})\n"
            return schema_text
        except Exception as e:
            logger.error(f"Error getting schema: {e}")
            return f"Error accessing database schema: {str(e)}"

    def clean_sql(self, sql: str) -> str:
        """
        Clean SQL query by removing markdown formatting.
        
        Args:
            sql: Raw SQL string from LLM
            
        Returns:
            Cleaned SQL query
        """
        sql = sql.replace("```sql", "").replace("```", "").strip()
        # Remove any leading/trailing whitespace and ensure it ends properly
        sql = sql.rstrip(';') + ';' if sql and not sql.endswith(';') else sql #to be tested
        return sql
    
    def query_similarity_search(self, client : chromadb.PersistentClient, collection_names : list[str], user_query : str, threshold : int=0.6, top_k : int =5) -> str:
        """
        Returns similar queries above with cosine similarity > threshold upto limit top_k.
        
        Args:
            client : ChromaDB client
            collection_names: name of the collections in chromaDB
            user_query: input by the user
            threshold: only queries with cosine_similarity > threshold are given
            top_k: returns atmost too_k many examples
        
        Returns:
            Most similar queries in String format
        """
        #cosine similarity < threshold
        matches = []

        #get list of actually existing collections
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
        
        if matches:
            context = "\n".join([f"Natural Language: {r['nl_query']} -> SQL: {r['sql_query']} (distance={r['distance']:.4f})" for r in matches])
            return context
        else:
            return " "
    
    def generate_sql(self, user_query: str) -> str:
        """
        Generate SQL query from natural language input.
        
        Args:
            user_query: User's natural language question
            
        Returns:
            Generated SQL query
        """
        if not self.llm:
            raise Exception("LLM not initialized")
        
        system_prompt = f"""
        You are an expert DuckDB SQL assistant specializing in oceanographic data analysis.

        The database has a single table with the following schema:
        {self.SCHEMA_TEXT}

        Instructions:
        - Always include all the corresponding fields (n_prof, n_levels, pres, temp, psal, latitude, longitude, time).
        - If not specified, do not add constraints on longitude and latitude
        - Generate valid DuckDB SQL for the user question.
        - Only use columns that exist in the schema above.
        - Return ONLY the DuckDB SQL query, no explanations or markdown formatting.
        - Use appropriate WHERE clauses to filter data based on the question.
        - Use aggregation functions (AVG, COUNT, MAX, MIN, SUM) when appropriate.
        - For date/time queries, assume date columns are in standard formats.
        - Limit results to reasonable numbers (e.g., LIMIT 1000) unless specifically asked for all data.
        """
        
        try:
            prompt = ChatPromptTemplate.from_messages([
                ("system", system_prompt),
                ("human", "{question}")
            ])
            
            chain = prompt | self.llm | StrOutputParser()
            sql_query = chain.invoke({"question": user_query})
            cleaned_sql = self.clean_sql(sql_query)
            
            logger.info(f"Generated SQL: {cleaned_sql}")
            return cleaned_sql
            
        except Exception as e:
            logger.error(f"Error generating SQL: {e}")
            raise Exception(f"Failed to generate SQL query: {str(e)}")
        
    def code_checker(self, user_query : str, last_sql : str, error_msg : str)->str:
        """
            Ask the LLM to fix the last SQL query based on the error message
        """

        if not self.llm:
            raise Exception("LLM not initialized")
        
        system_prompt = f"""
            You are an expert DuckDB SQL assistant.
            The database schema is:
            {self.SCHEMA_TEXT}

            The user asked: {user_query}

            The last generated SQL was:
            {last_sql}

            This SQL produced the following error when executed:
            {error_msg}

            Fix the SQL query so it runs correctly in DuckDB and satisfies the user's request.
            Return ONLY the corrected SQL query, with no explanation.
        """

        try:
            prompt = ChatPromptTemplate.from_messages([
                ("system", system_prompt)
                ("human", "{question}")
            ])
            
            chain = prompt | self.llm | StrOutputParser()
            fixed_sql = chain.invoke({"question" : user_query})
            return self.clean_sql(fixed_sql)
        
        except Exception as e:
            logger.error(f"Error in code_checker: {e}")
            raise Exception(f"Failed to fix SQL: {str(e)}")

    def run_sql(self, sql_query: str, user_query : str) -> pd.DataFrame:
        """
        Execute SQL query against the database.
        
        Args:
            sql_query: SQL query to execute
            
        Returns:
            Query results as DataFrame
        """
        try:
            con = duckdb.connect(self.DB_PATH)
            result_df = con.execute(sql_query).fetchdf()
            con.close()
            
            logger.info(f"Query executed successfully, returned {len(result_df)} rows")
            return result_df
            
        except Exception as e:
            logger.warning(f"SQL execution failed: {e}")
            logger.info("Attempting to fix SQL with code checker...")

            try:
                fixed_sql = self.code_checker(user_query=user_query, last_sql= sql_query, error_msg=str(e))
                logger.info(f"Fixed SQL : {fixed_sql}")
                con = duckdb.connect(self.DB_PATH)
                result_df = con.execute(fixed_sql).fetch_df()
                con.close()
                return result_df
            except Exception as fix_e:
                logger.error(f"Fix attempt failed: {fix_e}")
                return pd.DataFrame({"Error" : [str(fix_e)]})

    def summarize_query(self, user_query: str, df: pd.DataFrame) -> str:
        """
        Generate natural language summary of query results.
        
        Args:
            user_query: Original user question
            df: Query results DataFrame
            
        Returns:
            Natural language summary
        """
        if df.empty:
            return "No data found for your query. Please try rephrasing your question or check if the data exists in the database."
        
        if "Error" in df.columns:
            return f"I encountered an error while processing your query: {df.iloc[0]['Error']}"
        
        try:
            # Limit context size for large datasets
            if len(df) > 50:
                context = df.head(20).to_markdown() + f"\n\n... (showing first 20 of {len(df)} total rows)"
                summary_note = f"Note: This dataset contains {len(df)} total rows."
            else:
                context = df.to_markdown()
                summary_note = ""
            
            prompt_text = f"""
            You are an expert oceanographer analyzing data.

            User query: {user_query}

            Data results:
            {context}

            {summary_note}

            Instructions:
            - Provide a clear, concise answer based ONLY on the data shown.
            - Include specific numbers and statistics when relevant.
            - If the data shows trends or patterns, mention them.
            - Keep the response conversational and informative.
            - Don't make assumptions beyond what the data shows.
            """
            
            summary = self.llm.invoke(prompt_text)
            logger.info("Summary generated successfully")
            return summary
            
        except Exception as e:
            logger.error(f"Error generating summary: {e}")
            return f"I found {len(df)} records matching your query, but encountered an error generating the summary. You can view the raw data for details."

    def process_query(self, user_query: str) -> Dict[str, Any]:
        """
        Complete pipeline: process user query and return comprehensive results.
        
        Args:
            user_query: User's natural language question
            
        Returns:
            Dictionary containing SQL, data, summary, and status
        """
        try:
            # Generate SQL
            sql_query = self.generate_sql(user_query)
            
            # Execute query
            data = self.run_sql(sql_query, user_query)
            
            # Generate summary
            answer = self.summarize_query(user_query, data)
            
            return {
                "sql": sql_query,
                "data": data,
                "answer": answer,
                "success": True,
                "error": None
            }
            
        except Exception as e:
            logger.error(f"Error in process_query: {e}")
            return {
                "sql": "",
                "data": pd.DataFrame(),
                "answer": f"I apologize, but I encountered an error processing your query: {str(e)}",
                "success": False,
                "error": str(e)
            }
    
    def test_connection(self) -> bool:
        """
        Test database connection and LLM availability.
        
        Returns:
            True if both are working, False otherwise
        """
        try:
            # Test database
            con = duckdb.connect(self.DB_PATH)
            con.execute("SELECT 1;")
            con.close()
            
            # Test LLM
            if self.llm:
                test_response = self.llm.invoke("Hello")
                
            logger.info("Connection test successful")
            return True
            
        except Exception as e:
            logger.error(f"Connection test failed: {e}")
            return False
    
    def get_available_columns(self) -> list:
        """
        Get list of available columns in the database.
        
        Returns:
            List of column names
        """
        try:
            con = duckdb.connect(self.DB_PATH)
            columns = con.execute(f"DESCRIBE {self.TABLE_NAME};").fetchdf()
            con.close()
            return columns['column_name'].tolist()
        except Exception as e:
            logger.error(f"Error getting columns: {e}")
            return []

# Factory function for easy instantiation
def create_rag_pipeline(db_path: str = "./DB_files/data.duckdb", 
                       table_name: str = "ocean_profiles") -> RAGPipeline:
    """
    Factory function to create and initialize RAG pipeline.
    
    Args:
        db_path: Path to database file
        table_name: Name of the main table
        
    Returns:
        Initialized RAGPipeline instance
    """
    return RAGPipeline(db_path, table_name)

# ----------------------------
# Example Usage and Testing
# ----------------------------
if __name__ == "__main__":
    # Initialize pipeline
    rag = create_rag_pipeline()
    
    # Test connection
    if not rag.test_connection():
        print("Failed to connect to database or LLM")
        exit(1)
    
    # Example queries
    test_queries = [
        "What's the average temperature at 500m depth?",
        "Show me salinity data from 2012 onwards",
        "Find the deepest measurements in the Bay of Bengal",
        "How many temperature readings do we have?"
    ]
    
    print("Testing RAG Pipeline...")
    print("=" * 50)
    
    for query in test_queries:
        print(f"\nQuery: {query}")
        print("-" * 30)
        
        result = rag.process_query(query)
        
        if result["success"]:
            print(f"SQL: {result['sql']}")
            print(f"Rows returned: {len(result['data'])}")
            print(f"Answer: {result['answer']}")
        else:
            print(f"Error: {result['error']}")
        
        print()