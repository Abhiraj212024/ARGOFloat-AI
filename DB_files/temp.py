import duckdb

con = duckdb.connect("/Users/abhirajraje/Desktop/ARGOFloat-AI/DB_files/data.duckdb")
print(con.execute("DESCRIBE ocean_profiles").fetchdf())
