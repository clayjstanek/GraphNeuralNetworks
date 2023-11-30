#%%
import networkx as nx
import dgl

from neo4j import GraphDatabase

uri = "neo4j://10.236.61.21:7687"
username = "neo4j"
password = "neo4jneo4j"
driver = GraphDatabase.driver(uri, auth=(username, password))

#Query Nodes with Relationships
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

#%%

nx_graph = nx.Graph()

for row in data:     # Extract the nodes and relationships from the query result      
    #add source nod
    nx_graph.add_node(row['n_name'], label=row['n_labels'][0], **row['n_properties'])
    #Add dest node
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


for row in data2:     # Extract the nodes and relationships 
    #from the query result      
    #Ensure the properties exist before accessing them  
    #add source node
    nx_graph.add_node(row['n_name'], label=row['n_labels'][0], **row['n_properties'])

 
#%%
my_dgl_graph = dgl.from_networkx(nx_graph)
dgl.save_graphs('EUAIOpsDGLGraph.bin', [my_dgl_graph])

# %%
