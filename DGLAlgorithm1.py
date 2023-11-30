# -*- coding: utf-8 -*-
"""
Created on Wed Sep  6 22:02:58 2023
## use the .env python environment

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
loaded_graphs, labels = dgl.load_graphs('/theranch/clay/user_score/main/EUAIOpsDGLGraph.bin')
graph = loaded_graphs[0]
#Now, nodes contains the features for each log entry, and edges contains the 
#correlations between them. You can then convert these into tensors and use them 
#as input to the GNN as demonstrated in the previous code.

#This is a basic example, and in a real-world scenario, the logs would be more complex, 
#and the correlation criteria might need to be more sophisticated.
#%%
# Sample log data (you would typically parse and preprocess your logs to get this)
# Each event is represented as a node with some features
# Node data
node_data = graph.ndata
for key, value in node_data.items():
    print(f"Node feature '{key}': {value}")

# Edge data
edge_data = graph.edata
for key, value in edge_data.items():
    print(f"Edge feature '{key}': {value}")

#%%

# Define the GNN model
class GNN(torch.nn.Module):
    def __init__(self):
        super(GNN, self).__init__()
        self.conv1 = GCNConv(nodes_data.size(1), 16)
        self.conv2 = GCNConv(16, 2)  # Assuming binary classification (normal vs. anomalous)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)

# Create a graph data object
data = Data(x=nodes_data, edge_index=edges.t().contiguous())

# Initialize and train the model (assuming you have labeled data)
model = GNN()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
criterion = torch.nn.NLLLoss()

for epoch in range(100):
    model.train()
    optimizer.zero_grad()
    out = model(data)
    loss = criterion(out, labels)  # Assuming you have a 'labels' tensor for supervised training
    loss.backward()
    optimizer.step()

# Use the trained model for prediction or anomaly detection
model.eval()
predictions = model(data)