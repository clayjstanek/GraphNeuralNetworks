import os
import duckdb
import pandas as pd

def convert_to_markdown_table(df):
    markdown_table = df.to_markdown(index=False)
    return markdown_table

def get_all_tables(duckdb_path):
    with duckdb.connect(duckdb_path) as db:
        tables = db.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
        table_names = [table[0] for table in tables]
    return table_names

def main(duckdb_path, output_file):
    table_names = get_all_tables(duckdb_path)
    
    with open(output_file, 'w') as f:
        for table_name in table_names:
            # Connect to DuckDB and fetch data
            with duckdb.connect(duckdb_path) as db:
                data = db.execute(f"SELECT * FROM {table_name} ORDER BY rowid DESC LIMIT 10").fetchdf()

            # Convert data to a Markdown table
            markdown_table = convert_to_markdown_table(data)

            # Write table name and content to the output file
            f.write(f"## {table_name}\n\n")
            f.write(markdown_table)
            f.write("\n\n")

if __name__ == "__main__":
    is_prod = False  # Set to True for production, False for development

    # Config Parameters - must set DUCKDB_PROD_PATH and DUCKDB_DEV_PATH variables in ~/.bashrc
    if is_prod:
        duckdb_path = os.environ['DUCKDB_PROD_PATH']
        print("Using Production Database")
    else:
        duckdb_path = os.environ['DUCKDB_DEV_PATH']
        print("Using Development Database")
    print(f"Database path: {duckdb_path}")

    output_file = "tabledump.md"
    
    main(duckdb_path, output_file)
