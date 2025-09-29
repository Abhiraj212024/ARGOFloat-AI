import duckdb

conn = duckdb.connect("./DB_files/data.duckdb")

result = conn.execute("PRAGMA table_info(ocean_profiles)").fetchall()
print("\nColumn types using PRAGMA table_info():")
for row in result:
    print(f"{row[1]}: {row[2]}")

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