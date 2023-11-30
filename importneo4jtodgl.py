#%%
import networkx as nx
from neo4j import GraphDatabase

uri = "neo4j://10.236.61.21:7687"  # Update with your Neo4j connection details
username = "neo4j"
password = "neo4jneo4j"

driver = GraphDatabase.driver(uri, auth=(username, password))

query = "MATCH (n)-[r]->(m) RETURN n, r, m"  # Replace with your actual query

with driver.session() as session:
    result = session.run(query)
    data = result.data()

nx_graph = nx.Graph()

for row in data:
    # Extract the nodes and relationships from the query result
    node1 = row['n']
    rel = row['r']
    node2 = row['m']
    
    # Add nodes to the graph
    nx_graph.add_node(node1['name'], label=node1['label'])
    nx_graph.add_node(node2['name'], label=node2['label'])
    
    # Add edges to the graph
    nx_graph.add_edge(node1['name'], node2['name'], relationship=rel['type'])

my_dgl_graph = dgl.from_networkx(nx_graph)

# Use my_dgl_graph ...
# %%
