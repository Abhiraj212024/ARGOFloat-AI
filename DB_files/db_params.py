import duckdb 
import json
import os

db_path = './DB_files/data.duckdb'
table_name = "ocean_profiles"
# def min_max_query(db_path: str, table_name: str):
#     con = duckdb.connect(db_path)
#     # Get all columns from schema
#     columns = con.execute(f"DESCRIBE {table_name};").fetchdf()
    
#     # Build MIN/MAX select expressions
#     select_exprs = []
#     for col in columns['column_name']:
#         select_exprs.append(f"MIN({col}) AS min_{col}")
#         select_exprs.append(f"MAX({col}) AS max_{col}")
    
#     query = f"SELECT {', '.join(select_exprs)} FROM {table_name};"
#     return query

# Example usage:
# con = duckdb.connect(db_path)
# sql = min_max_query(db_path, table_name)
# print("Generated SQL:\n", sql)

# df = con.execute(sql).fetchdf()
# for col in df.columns:
#     print(f"{col}: {df[col].min()} : {df[col].max()}")

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
    
    return json.dumps(constraints)

CONSTRAINTS = get_min_max_constraints(db_path, table_name)
print(CONSTRAINTS)
