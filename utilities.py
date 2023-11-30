import duckdb
import os

def query(sql, path = 'tmp.db', to_df = True):
    """
    Runs a sql query on duckdb database, returns results as a pandas dataframe
    Args:
        sql (str): A sql query using duckdb specific syntax, see https://duckdb.org/docs/sql/introduction.html
        path (str):  Path to duckdb database file.  Defaults to local directory with name 'tmp.db'
        to_df (bool): Returns results as a pandas dataframe if True, duckdb relation if False.  Defaults to True.
    Returns: 
        Query results as a pandas dataframe or ducdb relation depending on to_df argument.

    """
    with duckdb.connect(path) as db:
        result = db.sql(sql)
         
        if result is None: #Hand queries that return nothing (e.g DROP, INSERT, CREATE)
            return None
    
        elif to_df:
            return result.df()
        else:
            return result