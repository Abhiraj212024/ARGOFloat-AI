import duckdb

conn = duckdb.connect("./DB_files/data.duckdb")

result = conn.execute("SELECT ROUND(latitude/10)*10 AS lat_band, AVG(temp) FROM ocean_profiles GROUP BY lat_band").fetchdf()
print(result.head())
# # First, drop the original time column
# conn.execute("ALTER TABLE ocean_profiles DROP COLUMN time")

# # Then rename the time_timestamp column to time
# conn.execute("ALTER TABLE ocean_profiles RENAME COLUMN time_timestamp TO time")

# # Verify the changes
# result = conn.execute("DESCRIBE ocean_profiles").fetchall()
# print("Table structure after changes:")
# for row in result:
#     print(f"{row[0]}: {row[1]}")

conn.close()