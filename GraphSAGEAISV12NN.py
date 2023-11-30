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
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset


import dgl
import numpy as np
import logging
import sys
import itertools

import json
import joblib
from joblib import dump
from tqdm import tqdm

BATCHSIZE = 32
NUM_EPOCHS = 100
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

initial_file_handler = logging.FileHandler('./logs/GraphSAGEAISLog112923V1.log')
logger.addHandler(initial_file_handler)

formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
initial_file_handler.setFormatter(formatter)

logger.debug("This will be logged to 'GraphSAGEAISLog112823.log'")
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

# Now, `nodes_feature_list` contains dictionaries for each node with its features
# This code will give you a list called nodes_data_list where each item is a dictionary. The dictionary has a key
# 'node_name' representing the name (or ID) of the node, and then it contains keys for each of the
# features with their corresponding values for that node.
# For example, to get the value of feature 'feature_name' for node 0, you would use:
"""
NODELEN = 10
#N = 34883

def Cosine_Similarity(values):
    similarity_matrix = cosine_similarity(values)
    return similarity_matrix

def dict_to_tensor(data_dict):
    tensors = []
    try:
        for key, value in data_dict.items():
            if isinstance(value, int):  # Convert integers to tensor with shape (1,)
                tensors.append(torch.tensor([value]))
            elif isinstance(value, torch.Tensor):
                if value.dim() == 0:  # If tensor is a scalar, reshape it
                    value = value.reshape(1)
                tensors.append(value)
            else:
                raise ValueError(logger.warning(f"Unexpected data type for key {key}: {type(value)}"))
    except AttributeError:
        logger.critical("Could not find items in data_dict term")
    except Exception as e:
        logger.critical(f"dict_to_tensor exception {e}")
    try:
        return torch.cat(tensors)
    except Exception as e:
        logger.critical("Dict_to_Tensor return torch.cat(tensors) failed")
        try:
            if data_dict.dtype == torch.float32:
                tensors = data_dict
                logger.critical("Dict_to_Tensor was able to return a tensor")
                return torch.tensor(tensors)
        except Exception as e:
            logger.critical("Dict_to_Tensor was NOT able to return a tensor.  Returning tensor of 0's")
            return torch.zeros(NODELEN, dtype=torch.float32)

def CalculateFeaturesLength(feature_list, feature_keys):
    length = 0
    for key in enumerate(feature_keys):
        try:
            featurelen = len(feature_list[0][key[1]])
            length = length + featurelen
        except TypeError:
            logger.warning(
                f"TypeError: {key[1]} trying to take length of 0-d tensor")
            length = length + 1
    return length

import torch.nn.functional as F


class AutoEncoder(nn.Module):
    def __init__(self, input_dim=NODELEN*2, embedding_dim= 5, hidden_dim1=15,
                 hidden_dim2=10, p=0.2):
        super(AutoEncoder, self).__init__()
        self.input_dim = input_dim
        self.embedding_dim = embedding_dim
        self.hidden_dim1 = hidden_dim1
        self.hidden_dim2 = hidden_dim2

        # Encoder layers
        self.encoder_fc1 = nn.Linear(input_dim, hidden_dim1)
        self.encoderdropout_fc1 = nn.Dropout(p)
        self.encoder_fc2 = nn.Linear(hidden_dim1, hidden_dim2)
        self.encoderdropout_fc2 = nn.Dropout(p)
        self.encoder_fc3 = nn.Linear(hidden_dim2, embedding_dim)
         
         # Decoder layers
        self.decoder_fc1 = nn.Linear(embedding_dim, hidden_dim2)
        self.decoderdropout_fc1 = nn.Dropout(p)
        self.decoder_fc2 = nn.Linear(hidden_dim2, hidden_dim1)
        self.decoderdropout_fc2 = nn.Dropout(p)
        self.decoder_fc3 = nn.Linear(hidden_dim1, input_dim)
        self.optimizer = optim.Adam(self.parameters(), lr=0.01,
                                    betas=(0.9, 0.999), eps=1e-08, weight_decay=0)
        self.loss_fn = nn.MSELoss()

    def forward(self, x):
        # Encoding
        x = F.relu(self.encoder_fc1(x))
        x = self.encoderdropout_fc1(x)
        x = F.relu(self.encoder_fc2(x))
        x = self.encoderdropout_fc2(x)
        encoded = self.encoder_fc3(x)
        
        # Decoding
        x = F.relu(self.decoder_fc1(encoded))
        x = self.decoderdropout_fc1(x)
        x = F.relu(self.decoder_fc2(x))
        x = self.decoderdropout_fc2(x)
        decoded = self.decoder_fc3(x)
        
        return encoded, decoded

    def train(self, dataLoader: DataLoader, num_epochs: int) -> int:
        loss = []
        for epoch in tqdm(range(num_epochs)):
            for data in (dataLoader):
                self.optimizer.zero_grad()
                try:
                    _, decoded = self.forward(data[0])
                    loss = self.loss_fn(decoded, data[0])
                except Exception as e:
                    print(e)
                    logger.critical(e)
                loss.backward()
                self.optimizer.step()

            print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}")
            logger.debug(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}")
        return 0


"""
With this setup, the autoencoder will learn to produce embeddings in the encoder 
that retain as much information as possible about the original 18-dimensional 
vectors, such that the decoder can reconstruct them with minimal error.
Once trained, you can use only the encoder part of the autoencoder to generate 
the 5-dimensional embeddings for nodes in your graph.
User
is the embedding_dimension the same dimension as the 'bottleneck' in a standard autoencoder?
ChatGPT
Yes, you're correct. The embedding_dim in the GraphSAGEAutoencoder structure we discussed is analogous to the 
bottleneck" in a standard autoencoder.

In autoencoder terminology:

The encoder compresses the input data down to this bottleneck dimension.
The decoder then tries to reconstruct the original data from this bottleneck representation.
The bottleneck (or embedding_dim in our context) represents a compressed, lower-dimensional representation of the 
original data. The idea is to capture the most essential features or patterns in this compressed representation, such 
that the original data can be reconstructed with minimal loss of information.
---------------------------------------------------------------------------
"""


class GraphSAGEAutoencoder(nn.Module):
    def __init__(self, adjacency_list, nodes_feature_list, feature_keys, 
                 input_dim=NODELEN*2, embedding_dim=5, hidden_dim1=15,
                 hidden_dim2=10, p=0.2):
        super(GraphSAGEAutoencoder, self).__init__()
        self.adj_list = adjacency_list
#        self.adj_list[35010]={}
        self.features = nodes_feature_list
        self.feature_keys = feature_keys
        self.concatenated_dim = 2*CalculateFeaturesLength(self.features, self.feature_keys)
        self.autoencoder = AutoEncoder(input_dim, embedding_dim, hidden_dim1, hidden_dim2, p)
        
    def aggregate_neighbors(self, node, depth=2):
        # Recursively aggregate features from neighbors
        nan_value = np.nan
        ii = 0
        if depth == 0:
            return self.features[node]
        if node >= 35010:
            dummy_node = torch.tensor(np.zeros((NODELEN,), dtype=np.float32))
            return dummy_node
        neighbors = self.adj_list[int(node)]
        neighbor_feats = [self.aggregate_neighbors(
            neighbor, depth-1) for neighbor in neighbors]

        features_dict = []  # list of dictionaries
        for nfeats in (neighbor_feats):
            features_list = [nfeats]  # This is how your data looks like
            # Extracting the dictionary from the list
            features_dict.append(features_list[0])
            ii = ii+1
        # Mean aggregation
        if not features_dict:
            return nan_value
        neighborhood_tensors_list = [
            dict_to_tensor(features_dict[jj]) for jj in range(ii)
        ]
        aggregated_feat = torch.mean(torch.stack(neighborhood_tensors_list, dim=0), dim=0)
        aggregated_feat[0] = 0
        return aggregated_feat

    
    def generate_embedding(self, node, TRAIN=False):
        # Generate embedding for a node using its features and aggregated neighbor features
        #        del node['node_name']
        concatenated_embedding = []
        aggregated_feat  = []
        if isinstance(node, tuple):
            node = node[1]
        if int(node['node_name'])==35005:
            print('node 35005')
#            node_feat=np.zeros((NODELEN,), dtype=np.float32)
#            aggregated_feat = np.zeros((NODELEN,), dtype=np.float32)

        dummy_node = np.zeros((NODELEN,), dtype=np.float32)
        try:
            node_feat = dict_to_tensor(self.features[node['node_name']])
        except IndexError:
            print(f"IndexError on node_feat node {node['node_name']}")
            logger.critical(f"IndexError on node_feat node {node['node_name']}")
            node_feat = torch.tensor(dummy_node)
            print(f"node_feat node {node['node_name']} zero-padded")
            logger.critical(f"node_feat node {node['node_name']} zero-padded")
        try:
            aggregated_feat = np.float32(self.aggregate_neighbors(node['node_name']))
        except IndexError:
            print(f"IndexError on node aggregated_feat {node['node_name']}")
            logger.critical(f"IndexError on node aggregated_feat {node['node_name']}")

        try:
            if(isinstance(aggregated_feat, dict)):
                if (np.any(aggregated_feat)) & (aggregated_feat.size==1):
            # aggregated_feat has a nan and is length 1 so don't concatenate               
                    embedding_tensor = node_feat
            elif(isinstance(aggregated_feat, float)):
                if (np.any(aggregated_feat)) & (len(aggregated_feat)==1):
                    aggregated_feat = dummy_node
        except AttributeError:
            logger.critical("aggregated_feat is length 1, type float, value NAN")
            print("aggregated_feat is length 1, type float, value NAN")
            aggregated_feat = {}
            embedding_tensor = node_feat
            
        if isinstance(aggregated_feat, dict) and aggregated_feat:
            values_array = np.array(list(aggregated_feat.values()), dtype=np.float32)
            concatenated_embedding = np.concatenate(np.array(node_feat, dtype=np.float32), values_array)
#        elif isinstance(aggregated_feat, dict) and (aggregated_feat.size>1) and (len(np.nonzero(np.isnan(aggregated_feat))[0])==0):
        elif (isinstance(aggregated_feat, float)) and (len(np.nonzero(np.isnan(aggregated_feat))[0])==0):
            concatenated_embedding = np.concatenate([node_feat.numpy(), aggregated_feat])
        else:
            concatenated_embedding = np.concatenate([node_feat.numpy(), dummy_node])

        embedding_tensor = torch.tensor(concatenated_embedding)
        if int(node['node_name'])>=35000:
            print(f"Node name {node['node_name']}:  {embedding_tensor}")


        # zero out the node name since this is simply an integer that increments by one for each node
        embedding_tensor[0] = 0.0
        embedding_tensor[NODELEN] = 0.0
        # Get reduced embedding
        encoder, decoder = self.autoencoder.forward(embedding_tensor)
        return encoder, decoder, embedding_tensor


"""
This code introduces a simple one-layer network for dimensionality reduction. After creating an 
instance of GraphSAGE, you can call the train_reducer method to train the neural network. Once 
trained, the generate_embedding method will produce the reduced-dimensionality embeddings.
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
