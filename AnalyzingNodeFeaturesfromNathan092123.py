# -*- coding: utf-8 -*-
"""
Created on Thu Sep 21 07:10:16 2023

@author: cstan
"""

# Import the load function from joblib
from joblib import load
import dgl
import networkx
import numpy as np
import pandas as pd

# Load the .joblib file
nx_full_features_with_nans = load('nx_full_20230920.joblib')
nx_features_no_nans = load('nx_sparse_20230920.joblib')
dgl_full_features, labels = dgl.load_graphs('dgl_20230920.bin')
graphDGL = dgl_full_features[0]

feature_keys = list(graphDGL.ndata.keys())
#create a dictionary for features
all_node_features = {}
# Iterate over each node feature key to extract the data
for key in feature_keys:
    all_node_features[key] = graphDGL.ndata[key].numpy()  # Convert to numpy array for easier handling

nodes_feature_list = []
# Iterate over each node
for node in range(graphDGL.number_of_nodes()):
    node_data = {'node_name': node}
    
    # Extract feature values for this node
    for key in feature_keys:
        node_data[key] = graphDGL.ndata[key][node]  # Assuming features are scalar values
    
    nodes_feature_list.append(node_data)

# Data from the sparse Networkx graph.  That is, if feature doesn't exist, not included as feature with that node
# Extracting nodes
nodes = list(nx_features_no_nans.nodes())

# Extracting edges
edges = list(nx_features_no_nans.edges())

# Extracting node features (attributes)
node_features = list(nx_features_no_nans.nodes(data=True))

# Extracting edge features (attributes)
edge_features = list(nx_features_no_nans.edges(data=True))

# Let's display the count of nodes, edges, and a few samples of node and edge features for a quick overview
len_nodes = len(nodes)
len_edges = len(edges)
sample_node_features = node_features[:5]
sample_edge_features = edge_features[:5]

print(f"len_nodes {len_nodes}, Len_edges: {len_edges}, sample_node_features")
print(f"Sample Node Features: {sample_node_features}, Sample_Edge_Features: {sample_edge_features}")


