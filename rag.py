#Importing necessary libraries and defining environment variables for LangChain, API keys, and endpoints.

import os
os.environ['LANGCHAIN_TRACING_V2'] = 'true'
os.environ['LANGCHAIN_ENDPOINT'] = 'https://api.smith.langchain.com'
LANGCHAIN_API_KEY = os.getenv('LANGCHAIN_API_KEY')

import duckdb
from langchain_community.llms import Ollama
from langchain.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda
from langchain_core.output_parsers import StrOutputParser

# ----------------------------
# 1. Setup LLaMA 3
# ----------------------------
llm = Ollama(model="llama3:8b-instruct-q4_K_M")

# ----------------------------
# 2. Hardcode or Introspect Schema
# ----------------------------
DB_PATH = "argo.db"

def get_schema_text(db_path: str) -> str:
    con = duckdb.connect(db_path)
    columns = con.execute("DESCRIBE argo;").fetchdf()
    schema_text = "Table: argo\n"
    for row in columns.itertuples():
        schema_text += f"  - {row.column_name} ({row.column_type})\n"
    return schema_text

SCHEMA_TEXT = get_schema_text(DB_PATH)

# ----------------------------
# 3. SQL Generation Function
# ----------------------------
def generate_sql(user_query: str) -> str:
    system_prompt = f"""
    You are an expert SQL assistant.

    The database has a single table with the following schema:
    {SCHEMA_TEXT}

    Instructions:
    - Generate valid SQL in DuckDB syntax for the user question.
    - Only use columns listed above.
    - Return ONLY the SQL query, no explanations.
    """
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{question}")
    ])
    
    def query_lambda(question: str) -> str:
        return (prompt | llm).invoke({"question": question})
    
    chain = RunnableLambda(query_lambda) | StrOutputParser
    return chain.invoke(user_query)

# ----------------------------
# 4. Execute SQL
# ----------------------------
def run_sql(sql_query: str):
    con = duckdb.connect(DB_PATH)
    return con.execute(sql_query).fetchdf()

# ----------------------------
# 5. Summarize Results
# ----------------------------
def summarize_query(user_query: str, df) -> str:
    context = df.to_markdown()
    prompt_text = f"""
    You are an expert oceanographer.

    User query: {user_query}

    Data:
    {context}

    Provide a concise, accurate answer based ONLY on the data.
    """
    return llm(prompt_text)

# ----------------------------
# 6. Example Usage
# ----------------------------
if __name__ == "__main__":
    user_question = "Salinity data from 2015 onwards in the Bay of Bengal at 100m depth"
    
    sql_query = generate_sql(user_question)
    print("Generated SQL:\n", sql_query)
    
    data = run_sql(sql_query)
    print("Query Results:\n", data)
    
    answer = summarize_query(user_question, data)
    print("Answer:\n", answer)
