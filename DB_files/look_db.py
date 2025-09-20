import duckdb
import os

con = duckdb.connect('data.duckdb')

# Print DB file size
db_path = 'data.duckdb'
db_size = os.path.getsize(db_path)
print(f"Database size: {db_size} bytes")

# Number of columns
columns = con.execute("PRAGMA table_info('ocean_profiles')").fetchdf()['name'].tolist()
print(f"Number of columns: {len(columns)}")

# Number of rows
num_rows = con.execute("SELECT COUNT(*) FROM ocean_profiles").fetchone()[0]
print(f"Number of rows: {num_rows}")

all_rows = con.execute("SELECT * FROM ocean_profiles").fetchdf()
print(all_rows)
con.close()
