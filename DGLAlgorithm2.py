# -*- coding: utf-8 -*-
"""
Created on Wed Sep  6 22:02:58 2023
## use the .env python environment
## on home machine, use the gnnv1 environment

Event correlation is a common task in cybersecurity and IT operations where you want 
to analyze log data to identify patterns or anomalies. PyTorch Geometric is a library 
primarily used for deep learning on graphs and structured data. While it's not a typical 
choice for log analysis, you can still use it to process and analyze log data for event 
correlation. However, you would likely need to preprocess the log data into a suitable 
format for graph-based analysis.

Here's a high-level overview of how you might approach using PyTorch Geometric for 
event correlation from sys log files and domain controller log files:

Data Preprocessing:

Read and parse the log files to extract relevant information.
Transform the log data into a graph or tensor format that PyTorch Geometric can 
work with. Each log entry can become a node in the graph, and the relationships 
between log entries can be represented as edges.
Constructing Graphs:

Use PyTorch Geometric's Data class to represent your graphs.
Create nodes and edges based on your log data.
Feature Engineering:

For each node in the graph, you may need to create feature vectors representing the 
log entries. These features could include information such as timestamps, log types, source IP addresses, etc.
You might also consider using techniques like word embeddings for text data within 
the log entries. Graph Neural Network (GNN):

Define a GNN architecture using PyTorch Geometric's classes like GCNConv, GATConv, 
or other graph convolutional layers. Train the GNN on your log data to learn patterns or correlations between events.

Event Correlation:

After training, you can use the GNN to make predictions or identify correlations between log events.
For example, you might look for patterns that indicate a potential security incident or system issue.
Evaluation and Analysis:

Evaluate the performance of your event correlation model using appropriate metrics.
Continuously monitor and update your model as new log data becomes available

@author: cstan
"""
#%%
import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data
import dgl
from dgl.nn import GraphConv
import networkx as nx

#%%
def parse_logs_to_graph(sys_logs, domain_logs):
    nodes = []
    edges = []
    IP_to_node = {}
    user_to_node = {}
    
    # Parse sys logs
    for log in sys_logs:
        timestamp, IP, event, _ = log.split(', ')
        node_id = len(nodes)
        nodes.append((timestamp, IP, event))
        if IP in IP_to_node:
            edges.append((IP_to_node[IP], node_id))
        IP_to_node[IP] = node_id

    # Parse domain logs
    for log in domain_logs:
        timestamp, user, event, domain = log.split(', ')
        node_id = len(nodes)
        nodes.append((timestamp, user, event, domain))
        if user in user_to_node:
            edges.append((user_to_node[user], node_id))
        user_to_node[user] = node_id

    return nodes, edges
#%%
#nodes, edges = parse_logs_to_graph(sys_logs, domain_logs)
loaded_graphs, labels = dgl.load_graphs('C:/Users/cstan/Documents/CooperStandard/tutorials/EUAIOps/sample_data_directed_edges.bin')
graph = loaded_graphs[0]
keys = graph.ndata.keys()
ekeys = graph.edata.keys()
"""
PICKLE=True
import pickle
if PICKLE:
    # Load from pickle
    with open("graph091523.pkl", "rb") as f:
        G_nx = pickle.load(f)
else:
    # Load graph from GraphML
    G_nx = nx.read_graphml("graphforNetworkx091523.graphml")
graph = dgl.from_networkx(G_nx)
#Now, nodes contains the features for each log entry, and edges contains the 
#correlations between them. You can then convert these into tensors and use them 
#as input to the GNN as demonstrated in the previous code.

#This is a basic example, and in a real-world scenario, the logs would be more complex, 
#and the correlation criteria might need to be more sophisticated.
#%%
# Sample log data (you would typically parse and preprocess your logs to get this)
# Each event is represented as a node with some features
"""
# Node data
node_data = graph.ndata
for key, value in node_data.items():
    print(f"Node feature '{key}': {value}")

# Edge data
edge_data = graph.edata
for key, value in edge_data.items():
    print(f"Edge feature '{key}': {value}")

#%%
# Define the GNN model for DGL
class GNN(torch.nn.Module):
    def __init__(self, in_feats, hidden_size, num_classes):
        super(GNN, self).__init__()
        self.conv1 = GraphConv(in_feats, hidden_size)
        self.conv2 = GraphConv(hidden_size, num_classes)

    def forward(self, g, inputs):
        h = self.conv1(g, inputs)
        h = F.relu(h)
        h = F.dropout(h, training=self.training)
        h = self.conv2(g, h)
        return F.log_softmax(h, dim=1)

# Assuming 'feat' is the name of the node feature tensor in the graph's ndata
in_features = graph.ndata['feat'].shape[1]
hidden_size = 16
num_classes = 2  # Assuming binary classification

model = GNN(in_features, hidden_size, num_classes)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
criterion = torch.nn.NLLLoss()

# Assuming labels is a tensor that provides labels for each node in the graph
for epoch in range(100):
    model.train()
    optimizer.zero_grad()
    out = model(graph, graph.ndata['feat'])
    loss = criterion(out, labels)
    loss.backward()
    optimizer.step()

# Use the trained model for prediction or anomaly detection
model.eval()
predictions = model(graph, graph.ndata['feat'])