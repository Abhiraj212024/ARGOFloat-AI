import duckdb
import json
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import OllamaLLM

# ----------------------------
# 1. Setup LLaMA 3
# ----------------------------
llm = OllamaLLM(model="llama3:8b-instruct-q4_K_M")

# ----------------------------
# 2. Hardcode or Introspect Schema
# ----------------------------
DB_PATH = "./DB_files/data.duckdb"
TABLE_NAME = "ocean_profiles"

#-----------------------------
# Handle constraints for error handling
#-----------------------------

def get_min_max_constraints(db_path: str, table_name: str) -> str:
    con = duckdb.connect(db_path)
    
    # Get all columns
    columns = con.execute(f"PRAGMA table_info('{table_name}');").fetchdf()
    col_names = columns['name'].tolist()
    
    constraints = {}
    for col in col_names:
        try:
            result = con.execute(f"SELECT MIN({col}) AS min_val, MAX({col}) AS max_val FROM {table_name}").fetchone()
            constraints[col] = {"min": result[0], "max": result[1]}
        except Exception:
            # Skip columns that cannot be aggregated (e.g., strings without order)
            continue
    
    return json.dumps(constraints, indent=4)


def valid_query(user_query: str) -> bool:
    constraints = json.loads(get_min_max_constraints(DB_PATH, TABLE_NAME))
    system_prompt = f"""
        You are an expert constraint checker.

        The database has the following constraints:
        {constraints}

        Instructions:
        - Check if the user query violates any constraints.
        - Return ONLY "True" if user_query is valid and "False" if it violates any constraints.
        -If you return "False", also provide a brief explanation of which constraint is violated.
    """

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            ("human", "{question}"),
        ]
    )

    chain = prompt | llm | StrOutputParser()
    result = chain.invoke({"question": user_query})

    return result


print(valid_query("Show me all profiles with temperature between 0 and 15 degrees and salinity above 30."))