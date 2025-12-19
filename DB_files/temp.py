import duckdb

con = duckdb.connect("./DB_files/data.duckdb")
print(con.execute("SHOW TABLES").fetchdf())

print(con.execute("SELECT * FROM ocean_profiles LIMIT 5").fetchdf())
con.close()