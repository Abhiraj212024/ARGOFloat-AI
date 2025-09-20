import pandas as pd
import duckdb
import os

parquet_dir = 'parquet_files'
parquet_files = [f for f in os.listdir(parquet_dir) if f.endswith('.parquet')]

con = duckdb.connect('data.duckdb')

table_created = False

for file in parquet_files:
    df = pd.read_parquet(os.path.join(parquet_dir, file))
    df = df.drop(['temp', 'pres', 'psal'], axis=1)
    df = df.rename(columns={
        'pres_adjusted': 'pres',
        'temp_adjusted': 'temp',
        'psal_adjusted': 'psal'
    })
    if 'time' in df.columns and pd.api.types.is_datetime64_any_dtype(df['time']):
        df['time'] = df['time'].dt.strftime('%Y-%m-%d %H:%M:%S').str[:19]
    con.register('df_view', df)
    if not table_created:
        con.execute("CREATE TABLE ocean_profiles AS SELECT * FROM df_view")
        table_created = True
    else:
        con.execute("INSERT INTO ocean_profiles SELECT * FROM df_view")
    con.unregister('df_view')

con.close()
