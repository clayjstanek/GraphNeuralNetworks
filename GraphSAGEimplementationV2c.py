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
        nan_mask = np.bool_(np.isnan(node_data[key]))
        try:
            node_data[key][nan_mask]=0
        except TypeError:
            print(f"Trying to mask a 0-d tensor.")
        except IndexError:
            print(f"IndexError too many indices for tensor of dim 0: key-{key} value- {node_data[key]}")
            if np.isnan(node_data[key])== 1:
                node_data[key] = 0
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
import torch.optim as optim
import torch.nn as nn

def dict_to_tensor(data_dict):
    tensors = []

    for key, value in data_dict.items():
        if isinstance(value, int):  # Convert integers to tensor with shape (1,)
            tensors.append(torch.tensor([value]))
        elif isinstance(value, torch.Tensor):
            if value.dim() == 0:  # If tensor is a scalar, reshape it
                value = value.reshape(1)
            tensors.append(value)
        else:
            raise ValueError(f"Unexpected data type for key {key}: {type(value)}")

    return torch.cat(tensors)

def CalculateFeaturesLength(feature_list, feature_keys):
    length = 0
    for key in enumerate(feature_keys):
        try:
            featurelen=len(feature_list[0][key[1]])
            length = length + featurelen
        except TypeError:
            print(f"TypeError: {key[1]} trying to take length of 0-d tensor")
    return length

class EmbeddingReducer(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(EmbeddingReducer, self).__init__()
        self.fc = nn.Linear(input_dim, output_dim)
        # Optionally, add more layers or nonlinearities if needed

    def forward(self, x):
        return self.fc(x)

class GraphSAGE:
    def __init__(self, adj_list, features, feature_keys):
        self.adj_list = adj_list
        self.features = features
        self.feature_keys= feature_keys
        
        # Define the neural network for dimensionality reduction
        #caclulate dimensionality of features together
        concatenated_dim = 2 * CalculateFeaturesLength(self.features, self.feature_keys) # Assuming all nodes have the same feature size
        reduced_dim = 384  # Or any other desired size
        self.reducer = EmbeddingReducer(concatenated_dim, reduced_dim)
        self.optimizer = optim.Adam(self.reducer.parameters(), lr=0.01)
        self.loss_fn = nn.MSELoss()


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
        # ... [Original aggregation code remains unchanged]
        #Generate embedding for a node using its features and aggregated neighbor features
#        del node['node_name']

        concatenated_embedding = []
        aggregated_feat = self.aggregate_neighbors(node['node_name'])
        if np.isnan(aggregated_feat):
            aggregated_feat = {}
        node_feat = self.features[node['node_name']]
        
        # Combine node's feature with aggregated feature
        if aggregated_feat != {}:
            concatenated_embedding = np.concatenate([node_feat, aggregated_feat])
        else:
            concatenated_embedding = node_feat
        # Convert concatenated embedding to torch tensor
        try:
            embedding_tensor = dict_to_tensor(concatenated_embedding).clone().detach()
        except ValueError:
#            embedding_tensor = torch.tensor(dict_to_tensor(node_feat), dtype=torch.float32).clone().detach()
            embedding_tensor = dict_to_tensor(node_feat).clone().detach()
            print(f"ValueError:  zero-dimensional arrays can't be concatenated (no adjacencies)")
        # Get reduced embedding
        reduced_embedding = self.reducer(embedding_tensor)
        
        return reduced_embedding.detach().numpy()

    def train_reducer(self, num_epochs=10):
        for epoch in range(num_epochs):
            total_loss = 0
            for node, feature in enumerate(self.features):
                concatenated_embedding = self.generate_embedding(node, self.adj_list, self.features)
                embedding_tensor = torch.tensor(concatenated_embedding, dtype=torch.float32)
                
                # Zero the gradients
                self.optimizer.zero_grad()
                
                # Forward pass
                reduced_embedding = self.reducer(embedding_tensor)
                
                # Compute loss
                loss = self.loss_fn(reduced_embedding, embedding_tensor)
                total_loss += loss.item()
                
                # Backward pass and optimization
                loss.backward()
                self.optimizer.step()
            
            print(f"Epoch {epoch+1}/{num_epochs}, Loss: {total_loss/len(self.features)}")
            
"""
This code introduces a simple one-layer network for dimensionality reduction. After creating an 
instance of GraphSAGE, you can call the train_reducer method to train the neural network. Once 
trained, the generate_embedding method will produce the reduced-dimensionality embeddings.

Note: This is a basic approach, and there are many possible improvements and nuances to consider, 
such as more complex architectures, different loss functions, and training strategies
"""
# Initialize GraphSAGE with our adjacency list and features
#graphsage = GraphSAGE(adj_list, features)
graphsage = GraphSAGE(adjacency_list, nodes_feature_list, feature_keys)

# Generate embeddings for all nodes
embeddings = {n: graphsage.generate_embedding(n) for n in nodes_feature_list}
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
            for i, n1 in enumerate(node_names)
            for j, n2 in enumerate(node_names) if i < j}

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
