# %%

import networkx as nx
import dgl
from neo4j import GraphDatabase
import pandas as pd
import neo4j

uri = "neo4j://10.236.61.21:7687"
username = "neo4j"
password = "neo4jneo4j"

driver = GraphDatabase.driver(uri, auth=(username, password))
# Query Nodes with Relationships
query = """MATCH (n)-[r]->(m)
RETURN
    coalesce(n.name, elementid(n)) AS n_name,
    labels(n) AS n_labels,
    n as n_properties,
    coalesce(m.name, elementid(m)) AS m_name,
    labels(m) AS m_labels,
    m as m_properties,
    type(r) AS r_type
"""

with driver.session() as session:
    result = session.run(query)

    data = result.data()

# %%

nx_graph = nx.Graph()

for row in data:  # Extract the nodes and relationships from the query result

    # CONVERT neo4j dates to python dtypes for n_properties

    for key, val in row['n_properties'].items():
         if isinstance(val, neo4j.time.Date):
            row['n_properties'][key] = pd.to_datetime(str(val))
        elif isinstance(val, neo4j.time.DateTime):
            row[key] = pd.to_datetime(str(val))

    # CONVERT neo4j dates to python dtypes for m_properties

    for key, val in row['m_properties'].items():
        if isinstance(val, neo4j.time.Date):
            row['m_properties'][key] = pd.to_datetime(str(val))
        elif isinstance(val, neo4j.time.DateTime):
            row['m_properties'][key] = pd.to_datetime(str(val))

    # add source node
    nx_graph.add_node(row['n_name'], label=row['n_labels'][0], **row['n_properties'])
    # Add dest node
    nx_graph.add_node(row['m_name'], label=row['m_labels'][0], **row['m_properties'])
    # Add edge
    nx_graph.add_edge(row['n_name'], row['m_name'], relationship=row['r_type'])

##Query nodes without relationships
query2 = """MATCH (n)
WHERE NOT (n)-[]-()
RETURN
    coalesce(n.name, elementid(n)) AS n_name,
    labels(n) AS n_labels,
    n as n_properties
"""

with driver.session() as session:
    result2 = session.run(query2)

    data2 = result2.data()

for row in data2:  # Extract the nodes and relationships from the query result

    # CONVERT neo4j dates to python dtypes for n_properties
    for key, val in row['n_properties'].items():
        if isinstance(val, neo4j.time.Date):
            row['n_properties'][key] = pd.to_datetime(str(val))
        elif isinstance(val, neo4j.time.DateTime):
            row[key] = pd.to_datetime(str(val))

    # Ensure the properties exist before accessing them

    # add source node
    nx_graph.add_node(row['n_name'], label=row['n_labels'][0], **row['n_properties'])

# %%
my_dgl_graph = dgl.from_networkx(nx_graph)
dgl.save_graphs('EUAIOpsDGLGraph091423.bin', [my_dgl_graph])

