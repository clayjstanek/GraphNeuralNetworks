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


from sklearn.metrics.pairwise import cosine_similarity
import torch.nn as nn
import torch.optim as optim
import torch
import dgl
import networkx as nx
import numpy as np
import logging
import sys
from sentence_transformers import SentenceTransformer
import pathlib as Path
from   tqdm import tqdm
import joblib
import json

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

initial_file_handler = logging.FileHandler('C://Users/cstan/Documents/CooperStandard/tutorials/EUAIops/logs/GraphSAGELog092623.log')
logger.addHandler(initial_file_handler)

formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
initial_file_handler.setFormatter(formatter)

logger.debug("This will be logged to 'GraphSAGELog092623.log'")
"""
logging.basicConfig(
    # Set the desired log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    level=logging.DEBUG,
    format="%(asctime)s [%(levelname)s]: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    # Specify the filename for the log file
    filename="C://Users/cstan/Documents/CooperStandard/tutorials/EUAIops/logs/GraphSAGELog092623.log",
    filemode="w",  # "w" to overwrite the file on each run, "a" to append
)
"""
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
logger.debug(f"Feature Keys: {feature_keys}")
# Create a dictionary to store features for each node
all_node_features = {}

# Iterate over each node feature key to extract the data
for key in feature_keys:
    # Convert to numpy array for easier handling
    all_node_features[key] = graphDGL.ndata[key].numpy()
logger.debug(f"All_node_features: {all_node_features}")
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
        # Assuming features are scalar values
        node_data[key] = graphDGL.ndata[key][node]
        nan_mask = np.bool_(np.isnan(node_data[key]))
        try:
            node_data[key][nan_mask] = 0
        except TypeError:
            print(f"Trying to mask a 0-d tensor.")
            logger.debug(f"Trying to mask a 0-d tensor.")
        except IndexError:
            print(
                f"IndexError too many indices for tensor of dim 0: key-{key} value- {node_data[key]}")
            logger.debug(
                f"IndexError too many indices for tensor of dim 0: key-{key} value- {node_data[key]}")
            if np.isnan(node_data[key]) == 1:
                node_data[key] = torch.tensor(0.0, dtype=torch.float32)
    nodes_feature_list.append(node_data)

# Now, `nodes_feature_list` contains dictionaries for each node with its features
# This code will give you a list called nodes_data_list where each item is a dictionary. The dictionary has a key
# 'node_name' representing the name (or ID) of the node, and then it contains keys for each of the
# features with their corresponding values for that node.
# For example, to get the value of feature 'feature_name' for node 0, you would use:

value_for_node_0 = nodes_feature_list[0]['label']
#logging.debug(f"nodes_feature_list: {nodes_feature_list}")

# Extract feature keys
feature_keys = [key for key in nodes_feature_list[0].keys()
                if key != 'node_name']

# Concatenate features for each node into a numpy array
#feature_arrays = [np.array([node_data[feature] for feature in feature_keys]) for node_data in nodes_data_list]
feature_arrays = [([node_data[feature] for feature in feature_keys])
                  for node_data in nodes_feature_list]

# At this point, feature_arrays[i] will give you the concatenated feature array for node i.

def Cosine_Similarity(values):
    similarity_matrix = cosine_similarity(values)
    for ii in range(50):
        for jj in range(100):
            similarity_matrix[ii][jj] = 0.0
    for ii in range(50):
        similarity_matrix[ii][ii] = 1
    return similarity_matrix

def dict_to_tensor(data_dict):
    tensors = []
#    if data_dict.items>0:
#        print("The dictionary has items.")
#    else:
#        print("The dictionary is empty.")
#        return torch.tensor(data_dict, dtype=torch.float32)
    try:
        for key, value in data_dict.items():
            if isinstance(value, int):  # Convert integers to tensor with shape (1,)
                tensors.append(torch.tensor([value]))
            elif isinstance(value, torch.Tensor):
                if value.dim() == 0:  # If tensor is a scalar, reshape it
                    value = value.reshape(1)
                tensors.append(value)
            else:
                raise ValueError(
                    f"Unexpected data type for key {key}: {type(value)}")
                logger.warning(
                    f"Unexpected data type for key {key}: {type(value)}")
    except AttributeError:
        logger.critical(f"Could not find items in data_dict term")
    except Exception as e:
        logger.critical(f"dict_to_tensor exception")
    try:
        return torch.cat(tensors)
    except Exception as e:
        logger.critical("Dict_to_Tensor return torch.cat(tensors) failed")
        try:
            if data_dict.dtype == torch.float32:
                tensors = data_dict
                return tensors
        except Exception as e:
            return torch.zeros(35730, dtype=torch.float32)

def CalculateFeaturesLength(feature_list, feature_keys):
    length = 1
    for key in enumerate(feature_keys):
        try:
            featurelen = len(feature_list[0][key[1]])
            length = length + featurelen
        except TypeError:
            print(f"TypeError: {key[1]} trying to take length of 0-d tensor")
            logger.warning(
                f"TypeError: {key[1]} trying to take length of 0-d tensor")
            length = length + 1
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
        self.feature_keys = feature_keys

        # Define the neural network for dimensionality reduction
        # caclulate dimensionality of features together
        # Assuming all nodes have the same feature size
        concatenated_dim = 2*CalculateFeaturesLength(
            self.features, self.feature_keys)
        reduced_dim = 384  # Or any other desired size
        self.reducer = EmbeddingReducer(concatenated_dim, reduced_dim)
        self.optimizer = optim.Adam(self.reducer.parameters(), lr=0.01)
        self.loss_fn = nn.MSELoss()

    def aggregate_neighbors(self, node, depth=2):
        # Recursively aggregate features from neighbors
        nan_value = np.nan
        aggregated_features = []
        neighborhood_tensors = []
        neighborhood_tensors_list = []
        ii = 0
        if depth == 0:
            return self.features[node]

        neighbors = self.adj_list[node]
        neighbor_feats = [self.aggregate_neighbors(
            neighbor, depth-1) for neighbor in neighbors]

        features_dict = []  # lilst of dictionaries
        for nfeats in (neighbor_feats):
            features_list = [nfeats]  # This is how your data looks like
            # Extracting the dictionary from the list
            features_dict.append(features_list[0])
            ii = ii+1
        # Mean aggregation
        if features_dict == []:
            aggregated_feat = nan_value
            return aggregated_feat

        for jj in range(ii):
            neighborhood_tensors_list.append(dict_to_tensor(features_dict[jj]))

        aggregated_feat = torch.mean(torch.stack(
            neighborhood_tensors_list, dim=0), dim=0)
        aggregated_feat[0] = 0
        return aggregated_feat

    def generate_embedding(self, node, TRAIN=False):
        # Generate embedding for a node using its features and aggregated neighbor features
        #        del node['node_name']
        concatenated_embedding = []
        aggregated_feat  = []
        if isinstance(node, tuple):
            node = node[1]
        node_feat = dict_to_tensor(self.features[node['node_name']])
        aggregated_feat = np.float32(self.aggregate_neighbors(node['node_name']))
        dummy_node = np.zeros((35730,), dtype=np.float32)

        try:
            if (np.any(aggregated_feat)) & (aggregated_feat.size==1):
            # aggregated_feat has a nan and is length 1 so don't concatenate               
                embedding_tensor = node_feat
        except AttributeError:
            logger.critical(f"aggregated_feat is length 1, type float, value NAN")
            print(f"aggregated_feat is length 1, type float, value NAN")
            aggregated_feat = {}
            embedding_tensor = node_feat
        if isinstance(aggregated_feat, dict):
            if(aggregated_feat):
                values_array = np.array(list(aggregated_feat.values()), dtype=np.float32)
                #concatenate node_feat with aggregated_feat (but it's a dictionary)
                concatenated_embedding = np.concatenate(node_feat, values_array)
                embedding_tensor = torch.tensor(concatenated_embedding)
       # Combine node's feature with aggregated feature
        else: # aggregated_feat is array
            if((aggregated_feat.size>1) & (len(np.nonzero(np.isnan(aggregated_feat))[0])==0)):
                concatenated_embedding = np.concatenate([node_feat.numpy(), aggregated_feat]) 
                embedding_tensor = torch.tensor(concatenated_embedding)  #arrays to tensor
            else:
                concatenated_embedding = np.concatenate([node_feat.numpy(), dummy_node])
                embedding_tensor = torch.tensor(concatenated_embedding)

        embedding_tensor[0] = 0.0
        # Get reduced embedding
#        reduced_embedding = self.reducer(embedding_tensor)
        if (TRAIN):
            reduced_embedding = embedding_tensor
        else:
            reduced_embedding = self.reducer(embedding_tensor)
            
        return reduced_embedding.detach().numpy()

    def train_reducer(self, num_epochs=10):
        for epoch in range(num_epochs):
            total_loss = 0
            for node in enumerate(self.features):
                concatenated_embedding = self.generate_embedding(node, True)
                embedding_tensor = torch.tensor(
                    concatenated_embedding, dtype=torch.float32)

                # Zero the gradients
                self.optimizer.zero_grad()

                # Forward pass
                reduced_embedding = self.reducer(embedding_tensor)
                embedding_tensor = embedding_tensor.unsqueeze(1)
                reduced_embedding = reduced_embedding.unsqueeze(1)

                # Compute loss
                loss = self.loss_fn( reduced_embedding, embedding_tensor )
                total_loss += loss.item()

                # Backward pass and optimization
                loss.backward()
                self.optimizer.step()

            print(
                f"Epoch {epoch+1}/{num_epochs}, Loss: {total_loss/len(self.features)}")
            logger.CRITICAL(
                f"Epoch {epoch+1}/{num_epochs}, Loss: {total_loss/len(self.features)}")


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
#graphsage.train_reducer(num_epochs=10)
# Generate embeddings for all nodes
embeddings = {index: graphsage.generate_embedding(node_features,True) for index, node_features in enumerate(nodes_feature_list)}
#print(embeddings)
#logging.debug(f"Embeddings: {embeddings}")
"""
The next step is to measure the similarity between these embeddings to detect correlated event patterns. 
A common way to measure similarity between vectors is using the cosine similarity. We'll compute pairwise 
cosine similarities between the embeddings to identify the IPs with correlated event patterns.
"""

# Compute pairwise cosine similarities
#similarity_matrix = cosine_similarity(list(embeddings.values()))
similarity_matrix = Cosine_Similarity(list(embeddings.values()))
logger.critical(f"{similarity_matrix}")

# Create a dictionary of IP pairs with their cosine similarity values
node_pairs = {(n1, n2): similarity_matrix[i, j]
              for i, n1 in enumerate(node_names)
              for j, n2 in enumerate(node_names) if i < j}

# Sort the IP pairs by similarity values in descending order
sorted_node_pairs = dict(
    sorted(node_pairs.items(), key=lambda item: item[1], reverse=True))
logger.debug(f"sorted_node_pairs:  {sorted_node_pairs.items()}")

from joblib import dump
dump(sorted_node_pairs, './SortedNodePairs_directededges092623.joblib')
"""
Here's the pairwise cosine similarity between the embeddings of the nodes:
You can set a threshold on similarity values to determine which pairs are 
considered to have "correlated" event patterns.

This approach provides a way to detect IP addresses with correlated activity patterns based 
on their event features and their relationships in the network. By applying this method to a 
real-world IP network with more features and larger datasets, you can gain insights into 
correlated activities and potential anomalies in the network.
"""
def convert_to_json_serializable(item):
    if isinstance(item, dict):
        return {convert_to_json_serializable(key): convert_to_json_serializable(value) for key, value in item.items()}
    elif isinstance(item, tuple):
        return str(item)
    elif isinstance(item, np.float32):
        return float(item)
    else:
        return item

# Load data from joblib file
joblib_file = './SortedNodePairs_directededges092623.joblib'
data = joblib.load(joblib_file)

# Convert data to be JSON serializable
json_serializable_data = convert_to_json_serializable(data)

# Save data to JSON file
json_file = 'SortedNodePairs_directededges092623.json'
with open(json_file, 'w') as f:
    json.dump(json_serializable_data, f)

print(f"Data saved to {json_file}")


logging.shutdown()
sys.exit(0)
"""
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
