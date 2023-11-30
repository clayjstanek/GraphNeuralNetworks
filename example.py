#%%
from neo4j import GraphDatabase
import pandas as pd
from neo4j_utilities import Connection
import logging
logging.getLogger("utilities.Connection").setLevel(logging.ERROR)


#%%
##NEO4J CONNECTION
URI = 'neo4j://10.236.61.21:7687'
AUTH = ("neo4j","neo4jneo4j")

conn = Connection(URI, AUTH)

#%%
events = f"""
    match (a: event)
    return count(a) as n
    """  
records = conn.query(events, dtype = 'df')
records.to_csv('test.csv')
#print(records)

# %%
