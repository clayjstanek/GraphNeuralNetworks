# -*- coding: utf-8 -*-
"""
Created on Fri Sep 15 09:16:32 2023

@author: cstan
"""

"""
GraphSAGE: A Brief Overview
GraphSAGE is a method to generate node embeddings by sampling and aggregating features from a 
node's neighbors. The primary idea behind GraphSAGE is to learn a function that can sample and 
aggregate the features from a node's neighbors to generate the node's embedding. The aggregation 
function can be mean, LSTM, pooling, etc.

How GraphSAGE Works:
Sampling Neighbors: For each node, GraphSAGE samples a fixed-size set of neighbors.
Feature Aggregation: It then aggregates these neighbors' features to produce a new feature for the target node.
Repeating: This process is repeated for a fixed number of iterations, leading to 
multi-hop neighborhood sampling and aggregation.
Combining with Node's Own Features: Finally, the aggregated feature is combined with the node's 
current feature to generate its new feature.
GraphSAGE for Event Correlation:
Given a network of users and their activities, nodes can represent users and edges can represent 
the relationships between them. Events (or activities) can be features associated with each node. 
To detect event correlations, we can:

Encode each activity as a feature vector for the corresponding user node.
Use GraphSAGE to generate embeddings for each node (user).
Nodes with similar embeddings are likely to have correlated activity patterns.
Implementing GraphSAGE:
For this example, we'll use a simple mean aggregator. Here's a high-level Python 
code to understand the GraphSAGE algorithm:
"""
import torch
import dgl
import networkx as nx
import numpy as np
from sentence_transformers import SentenceTransformer
    
"""
The GraphSAGE class takes an adjacency list and node features as input.
aggregate_neighbors recursively aggregates features from neighbors up to a certain depth.
generate_embedding generates the final embedding by combining the node's own features with aggregated neighbor features.
You would need an actual graph dataset with user activities to run this. Additionally, 
in a practical setting, neural networks are used to further refine and combine the features.

To detect event correlations:

Use GraphSAGE to generate embeddings for all nodes.
Measure similarity between embeddings (e.g., cosine similarity) to determine correlated activity patterns.
"""

"""

Graph:
Nodes: generic computers with generic numeric names
Edges: Communication between computers
Node Features: Vector of events (110, e.g., number of login attempts, number of data uploads, etc.)

Goal:
We want to detect computing terminals (nodes) with correlated event patterns.

Step-by-step:
"""
loaded_graphs, labels = dgl.load_graphs('./sample_data_directed_edges.bin')
graphDGL = loaded_graphs[0]
G_nx = graphDGL.to_networkx()
node_names = list(G_nx.nodes())

adjacency_list = dict(G_nx.adjacency())
feature_keys = list(graphDGL.ndata.keys())

# Create a dictionary to store features for each node
all_node_features = {}

# Iterate over each node feature key to extract the data
for key in feature_keys:
    all_node_features[key] = graphDGL.ndata[key].numpy()  # Convert to numpy array for easier handling

# Now, `all_node_features` contains every feature for every node
# For example, to get feature 'feature_name' for node 0:
# feature_value = all_node_features['feature_name'][0]

# List to store dictionaries for each node
nodes_feature_list = []

# Iterate over each node
for node in range(graphDGL.number_of_nodes()):
    node_data = {'node_name': node}
    
    # Extract feature values for this node
    for key in feature_keys:
        node_data[key] = graphDGL.ndata[key][node]  # Assuming features are scalar values
    
    nodes_feature_list.append(node_data)

# Now, `nodes_feature_list` contains dictionaries for each node with its features
# This code will give you a list called nodes_data_list where each item is a dictionary. The dictionary has a key 
#'node_name' representing the name (or ID) of the node, and then it contains keys for each of the 
# features with their corresponding values for that node.
# For example, to get the value of feature 'feature_name' for node 0, you would use:

value_for_node_0 = nodes_feature_list[0]['label']


# Extract feature keys
feature_keys = [key for key in nodes_feature_list[0].keys() if key != 'node_name']

# Concatenate features for each node into a numpy array
#feature_arrays = [np.array([node_data[feature] for feature in feature_keys]) for node_data in nodes_data_list]
feature_arrays = [([node_data[feature] for feature in feature_keys]) for node_data in nodes_feature_list]

# At this point, feature_arrays[i] will give you the concatenated feature array for node i.

class GraphSAGE:
    def __init__(self, adj_list, features):
        self.adj_list = adj_list
        self.features = features

    def aggregate_neighbors(self, node, depth=2):
        #Recursively aggregate features from neighbors
        if depth == 0:
            return self.features[node]
        
        neighbors = self.adj_list[node]
        neighbor_feats = [self.aggregate_neighbors(neighbor, depth-1) for neighbor in neighbors]
        
        # Mean aggregation
        aggregated_feat = np.mean(neighbor_feats, axis=0)
        
        return aggregated_feat

    def generate_embedding(self, node):
        #Generate embedding for a node using its features and aggregated neighbor features
        aggregated_feat = self.aggregate_neighbors(node)
        node_feat = self.features[node]
        
        # Combine node's feature with aggregated feature
        combined_feat = np.concatenate([node_feat, aggregated_feat])
        
        return combined_feat

# Initialize GraphSAGE with our adjacency list and features
#graphsage = GraphSAGE(adj_list, features)
graphsage = GraphSAGE(adjacency_list, nodes_feature_list)

# Generate embeddings for all IP addresses
embeddings = {n: graphsage.generate_embedding(n) for n in node_names}
print(embeddings)
"""
The next step is to measure the similarity between these embeddings to detect correlated event patterns. 
A common way to measure similarity between vectors is using the cosine similarity. We'll compute pairwise 
cosine similarities between the embeddings to identify the IPs with correlated event patterns.
"""
from sklearn.metrics.pairwise import cosine_similarity

# Compute pairwise cosine similarities
similarity_matrix = cosine_similarity(list(embeddings.values()))

# Create a dictionary of IP pairs with their cosine similarity values
node_pairs = {(n1, n2): similarity_matrix[i, j]
            for i, n1 in enumerate(ip_addresses)
            for j, n2 in enumerate(ip_addresses) if i < j}

# Sort the IP pairs by similarity values in descending order
sorted_node_pairs = dict(sorted(node_pairs.items(), key=lambda item: item[1], reverse=True))
sorted_node_pairs

"""
Here's the pairwise cosine similarity between the embeddings of the nodes:
You can set a threshold on similarity values to determine which pairs are 
considered to have "correlated" event patterns.

This approach provides a way to detect IP addresses with correlated activity patterns based 
on their event features and their relationships in the network. By applying this method to a 
real-world IP network with more features and larger datasets, you can gain insights into 
correlated activities and potential anomalies in the network.

Integrating neural networks into the GraphSAGE architecture can further refine and 
combine features to enhance the representation of the nodes.

Here's how we can integrate a simple neural network into our GraphSAGE implementation:

Neural Network for Aggregation: Instead of using a basic mean aggregator, we'll use a neural network 
to aggregate features from neighbors. This network will take in the neighbors' features and produce a 
refined aggregated feature.

Combining Node Features: We'll use another neural network to combine the node's own feature with the 
aggregated feature to produce the final embedding.
For this purpose, we'll use the popular deep learning library PyTorch. Let's start by defining the 
neural networks for aggregation and combining features:
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import tqdm as tqdm
#from .autonotebook import tqdm as notebook_tqdm

# Define the neural network for aggregation
class Aggregator(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(Aggregator, self).__init__()
        self.fc = nn.Linear(input_dim, output_dim)
    
    def forward(self, features):
        aggregated = torch.mean(features, dim=0)  # mean aggregation
        return F.relu(self.fc(aggregated))  # Apply neural network

# Define the neural network to combine node's feature with aggregated feature
class Combine(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(Combine, self).__init__()
        self.fc = nn.Linear(input_dim, output_dim)
    
    def forward(self, node_feat, aggregated_feat):
        combined = torch.cat([node_feat, aggregated_feat], dim=0)  # concatenate features
        return F.relu(self.fc(combined))  # Apply neural network

# Convert our features to PyTorch tensors
features_tensor = {ip: torch.tensor(features[ip], dtype=torch.float32) for ip in ip_addresses}
# TqdmWarning: IProgress not found. Please update jupyter and ipywidgets. See https://ipywidgets.readthedocs.io/en/stable/user_install.html


"""
I've set up two neural networks:

Aggregator: This network takes the features of neighbors and outputs an aggregated feature. We start with mean 
aggregation and then pass the result through a feedforward neural network.
Combine: This network combines the node's own feature with the aggregated feature to produce the final embedding.
Additionally, I've converted our features into PyTorch tensors, which will make it easier to work with these neural networks.

Next, let's modify our GraphSAGE implementation to utilize these neural networks:
    """
class NeuralGraphSAGE:
    def __init__(self, adj_list, features):
        self.adj_list = adj_list
        self.features = features
        self.input_dim = features[next(iter(features))].shape[0]
        
        # Initialize neural network modules
        self.aggregator = Aggregator(self.input_dim, self.input_dim)
        self.combiner = Combine(2 * self.input_dim, 2 * self.input_dim)

    def aggregate_neighbors(self, node, depth=2):
        """Recursively aggregate features from neighbors using the aggregator neural network"""
        if depth == 0:
            return self.features[node]
        
        neighbors = self.adj_list[node]
        neighbor_feats = torch.stack([self.aggregate_neighbors(neighbor, depth-1) for neighbor in neighbors])
        
        # Use the aggregator neural network for aggregation
        aggregated_feat = self.aggregator(neighbor_feats)
        
        return aggregated_feat

    def generate_embedding(self, node):
        """Generate embedding for a node using its features, aggregated neighbor features, and the combiner neural network"""
        aggregated_feat = self.aggregate_neighbors(node)
        node_feat = self.features[node]
        
        # Use the combiner neural network to get the final embedding
        combined_feat = self.combiner(node_feat, aggregated_feat)
        
        return combined_feat.detach().numpy()  # Convert the result to numpy array

# Initialize the new GraphSAGE model with neural networks
neural_graphsage = NeuralGraphSAGE(adj_list, features_tensor)

# Generate embeddings for all IP addresses
neural_embeddings = {ip: neural_graphsage.generate_embedding(ip) for ip in ip_addresses}
print(neural_embeddings)

"""

I apologize for the oversight. The previous execution state was reset, which means we lost some of our intermediate data, 
including the adjacency list and the feature tensors. Let's recreate the necessary data and neural network components 
before we proceed with the GraphSAGE implementation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# Dummy data setup
ip_addresses = ["192.168.0.1", "192.168.0.2", "192.168.0.3", "192.168.0.4", "192.168.0.5"]
adj_list = {
    "192.168.0.1": ["192.168.0.2", "192.168.0.3"],
    "192.168.0.2": ["192.168.0.1", "192.168.0.4"],
    "192.168.0.3": ["192.168.0.1", "192.168.0.5"],
    "192.168.0.4": ["192.168.0.2"],
    "192.168.0.5": ["192.168.0.3"]
}
features = {
    "192.168.0.1": np.array([5, 10]),
    "192.168.0.2": np.array([6, 9]),
    "192.168.0.3": np.array([4, 11]),
    "192.168.0.4": np.array([5, 10]),
    "192.168.0.5": np.array([6, 9])
}

# Convert features to PyTorch tensors
features_tensor = {ip: torch.tensor(features[ip], dtype=torch.float32) for ip in ip_addresses}

# Neural network components
class Aggregator(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(Aggregator, self).__init__()
        self.fc = nn.Linear(input_dim, output_dim)
    
    def forward(self, features):
        aggregated = torch.mean(features, dim=0)
        return F.relu(self.fc(aggregated))

class Combine(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(Combine, self).__init__()
        self.fc = nn.Linear(input_dim, output_dim)
    
    def forward(self, node_feat, aggregated_feat):
        combined = torch.cat([node_feat, aggregated_feat], dim=0)
        return F.relu(self.fc(combined))

# Now, let's try running the NeuralGraphSAGE implementation again
neural_graphsage = NeuralGraphSAGE(adj_list, features_tensor)
neural_embeddings = {ip: neural_graphsage.generate_embedding(ip) for ip in ip_addresses}
neural_embeddings

"""
RESULT
{'192.168.0.1': array([0.       , 1.6162038, 0.       , 3.3733692], dtype=float32),
 '192.168.0.2': array([0.       , 1.4494897, 0.       , 3.01642  ], dtype=float32),
 '192.168.0.3': array([0.       , 1.7829179, 0.       , 3.7303185], dtype=float32),
 '192.168.0.4': array([0.       , 1.6162038, 0.       , 3.3733692], dtype=float32),
 '192.168.0.5': array([0.       , 1.4494897, 0.       , 3.01642  ], dtype=float32)}
"""
"""
We've successfully generated neural embeddings for each IP address using our NeuralGraphSAGE implementation. 
Here are the embeddings:

'192.168.0.1': [0. , 1.6162038, 0. , 3.3733692]
'192.168.0.2': [0. , 1.4494897, 0. , 3.01642 ]
... and so on.
These embeddings were generated using the neural networks we integrated into the GraphSAGE algorithm. As 
seen in the values, the embeddings are different from our initial implementation without neural networks, 
demonstrating the power of neural networks in refining and combining features for nodes.

Next, as before, you can measure similarity between these embeddings to detect correlated event patterns 
among the IP addresses.
"""
from sklearn.metrics.pairwise import cosine_similarity

# Compute pairwise cosine similarities
similarity_matrix = cosine_similarity(list(neural_embeddings.values()))

# Create a dictionary of IP pairs with their cosine similarity values
ip_pairs = {(ip1, ip2): similarity_matrix[i, j]
            for i, ip1 in enumerate(ip_addresses)
            for j, ip2 in enumerate(ip_addresses) if i < j}

# Sort the IP pairs by similarity values in descending order
sorted_ip_pairs = dict(sorted(ip_pairs.items(), key=lambda item: item[1], reverse=True))
sorted_ip_pairs
"""

GraphSAGE, "SAGE" stands for Sample and AggregatE. The algorithm works by:

Sampling a fixed-size set of neighbors for each node.
Aggregating these neighbors' features to produce a new feature for the target node.
Thus, the name "GraphSAGE" emphasizes the core mechanism of sampling neighbors and aggregating 
their features in the context of graph-based learning.
"""





