# -*- coding: utf-8 -*-
"""
Created on Sat Oct 07 23:12 2023

@author: cstan
"""

from sklearn.model_selection import train_test_split
import torch
import argparse
from torch.utils.data import DataLoader, TensorDataset
import dgl
import numpy as np
import logging
import sys
import json
import joblib
from joblib import dump
from tqdm import tqdm
from GraphSAGEAISV12NN import Cosine_Similarity, GraphSAGEAutoencoder, convert_to_json_serializable, \
    CalculateFeaturesLength
param = 'ais-dgl.bin'
BATCHSIZE = 32
NUM_EPOCHS = 100

def ConfigureLogger(logfilename):
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)

    initial_file_handler = logging.FileHandler('./logs/'+logfilename)
    logger.addHandler(initial_file_handler)

    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    initial_file_handler.setFormatter(formatter)

    logger.debug(f"This will be logged to {logfilename}")
    return logger

def main(args):

    logger = ConfigureLogger(args.logfile)
    loaded_graphs, labels = dgl.load_graphs(args.graphfile)
    hiddenlayer1 = args.hidden_layer_1_nodes
    hiddenlayer2 = args.hidden_layer_2_nodes
    jsonoutputfile = args.json_outputfile
    joblibfile = args.joblib_outputfile
    graphDGL = loaded_graphs[0]
    G_nx = graphDGL.to_networkx()
    node_names = list(G_nx.nodes())

    adjacency_list = dict(G_nx.adjacency())
    feature_keys = list(graphDGL.ndata.keys())
    logger.debug(f"Feature Keys: {feature_keys}")
    all_node_features = {key: graphDGL.ndata[key].numpy() for key in feature_keys}
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
                if(nan_mask):
                    node_data[key] = torch.tensor(0.0, dtype=torch.float32)
            except TypeError:
                logger.debug(f"Trying to mask a 0-d tensor. node_data[key][nan_mask]: {node_data[key][nan_mask]}")
            except IndexError:
                logger.debug(
                    f"IndexError too many indices for tensor of dim 0: key-{key} value- {node_data[key]}")
                if np.isnan(node_data[key]) == 1:
                    node_data[key] = torch.tensor(0.0, dtype=torch.float32)
        nodes_feature_list.append(node_data)
        
#    nodes_feature_list = nodes_feature_list[0:len(nodes_feature_list)-10]

    NODES = len(nodes_feature_list)
   
    # Now, `nodes_feature_list` contains dictionaries for each node with its features
    # This code will give you a list called nodes_data_list where each item is a dictionary. The dictionary has a key
    # 'node_name' representing the name (or ID) of the node, and then it contains keys for each of the
    # features with their corresponding values for that node.
    # For example, to get the value of feature 'feature_name' for node 0, you would use:

    value_for_node_0 = nodes_feature_list[0]['label']
    # logging.debug(f"nodes_feature_list: {nodes_feature_list}")

    # Extract feature keys
    feature_keys = []
#    for key in nodes_feature_list[0].keys():
#        if key != 'node_name':
#            feature_keys.append(key)
    for key in nodes_feature_list[0].keys():
        feature_keys.append(key)
    # Concatenate features for each node into a numpy array
    # feature_arrays = [np.array([node_data[feature] for feature in feature_keys]) for node_data in nodes_data_list]
    feature_arrays = [([node_data[feature] for feature in feature_keys]) for node_data in nodes_feature_list]
    
    # At this point, feature_arrays[i] will give you the concatenated feature array for node i.
    NODELEN=(CalculateFeaturesLength(nodes_feature_list, feature_keys))
    # Initialize GraphSAGE with our adjacency list and features
    graphsage = GraphSAGEAutoencoder(adjacency_list, nodes_feature_list, feature_keys,
                                     input_dim=NODELEN * 2, embedding_dim=5, hidden_dim1=hiddenlayer1,
                                     hidden_dim2=hiddenlayer2, p=0.2)
    print("Start embeddings")
    logger.debug("Start embeddings generation")
    embeddings_tuple = {index: graphsage.generate_embedding(node_features, True) for index,
                        node_features in tqdm(enumerate(nodes_feature_list))}
    print("Completed embeddings")
    logger.debug("Complete embeddings generation")

    # Extracting the arrays
    arrays_bottleneck = [embeddings_tuple[i][0] for i in range(NODES)]
    arrays_decoded_first = [embeddings_tuple[i][1] for i in range(NODES)]
    node_embeddings = [embeddings_tuple[i][2] for i in range(NODES)]

    #print(len(arrays_384), len(arrays_35898_first), len(node_embeddings))
    logger.debug(f"GraphSageAutoEncoder.generate_embedding returns lists: {len(arrays_bottleneck)}, {len(arrays_decoded_first)}, {len(node_embeddings)}")

    y = torch.zeros(len(node_embeddings))
    node_embeddings_train, node_embeddings_test, y_train, y_test = train_test_split(node_embeddings, y, test_size=0.2, random_state=42)

    # Convert the list of arrays into a tensor
    #tensor_data_train = torch.tensor(node_embeddings_train)
    #tensor_data_train = node_embeddings_train
    node_embeddings_train_array = torch.stack(node_embeddings_train)
    node_tensors = []
    for node_embeddings_array in node_embeddings:
        if isinstance(node_embeddings_array, int):  # Convert integers to tensor with shape (1,)
            node_tensors.append(torch.tensor([node_embeddings_array]))
        elif isinstance(node_embeddings_array, torch.Tensor):
            if node_embeddings_array.dim() == 0:  # If tensor is a scalar, reshape it
                node_embeddings_array = node_embeddings_array.reshape(1)
            node_tensors.append(node_embeddings_array)

    node_tensors = torch.stack(node_tensors)
    from sklearn.preprocessing import MinMaxScaler
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(node_tensors.numpy())
    node_tensors = torch.tensor(X_scaled)

    # Create a TensorDataset
    dataset = TensorDataset(node_tensors)

    # Initialize a DataLoader
    batch_size = BATCHSIZE  # You can adjust this as needed
    print("Training Beginning")
    logger.info("Training Beginning")
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    try:
        graphsage.autoencoder.train(dataloader, NUM_EPOCHS)
    except Exception as e:
        logger.error("GraphsageAutoEncoder.train experienced unrecoverable error")
        logging.shutdown()
        sys.exit(10)

    # now that neural network is trained, go back and create embeddings, but use results of autoencoder for dim reduction
    print("Computing trained embeddings")
    logger.info("Computing trained embeddings")
    embeddings_final = {index: graphsage.generate_embedding(node_features, True) for index, node_features in tqdm(enumerate(nodes_feature_list))}
    arrays_384 = [embeddings_final[i][0] for i in range(NODES)]
    print("Completed trained embeddings")
    logger.info("completed trained embeddings")
    """
    The next step is to measure the similarity between these embeddings to detect correlated event patterns. 
    A common way to measure similarity between vectors is using the cosine similarity. We'll compute pairwise 
    cosine similarities between the embeddings to identify the IPs with correlated event patterns.
    """
    arrays_384 = torch.stack(arrays_384)
    # Compute pairwise cosine similarities
    similarity_matrix = Cosine_Similarity(list(arrays_384.detach().numpy()))
    print("Computed Similarity Matrix")
    logger.critical(f"{similarity_matrix}")

    # Create a dictionary of IP pairs with their cosine similarity values
    node_pairs = {(n1, n2): similarity_matrix[i, j]
                  for i, n1 in enumerate(node_names)
                  for j, n2 in enumerate(node_names) if i < j}

    # Sort the IP pairs by similarity values in descending order
    sorted_node_pairs = dict(
        sorted(node_pairs.items(), key=lambda item: item[1], reverse=True))
    logger.info(f"sorted_node_pairs:  {sorted_node_pairs.items()}")

    dump(sorted_node_pairs, './'+args.joblib_outputfile)

    # Load data from joblib file

    data = joblib.load('./'+args.joblib_outputfile)

    # Convert data to be JSON serializable
    json_serializable_data = convert_to_json_serializable(data)

    # Save data to JSON file
    json_file = jsonoutputfile #'SortedNodePairs_dgl_20231117.json'
    with open(json_file, 'w') as f:
        json.dump(json_serializable_data, f)

    print(f"Data saved to {json_file}")
    logging.info(f"Data saved to {json_file}")

    print("GraphSAGEAISNN Terminated")
    logger.info("Program Terminated.")
    logging.shutdown()
    sys.exit(0)
    # This checks if the script is being run as the main program

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="A program for predicting correlated events through graph theory.")
    parser.add_argument("graphfile", type=str,
    help="Name of <graph>.bin file to use.  If not in same folder as main, give absolute path",
                        default='ais-dgl.bin')
    parser.add_argument("--logfile", type=str, help="Name of Log File", default='GraphSageAIS_20231129.log')
    parser.add_argument("--joblib_outputfile", type=str,
                        help="Name of <name>.joblib file to use.  If not in same folder as main, give absolute path",
                        default='sortedpairs_dgl_20231129.joblib')
    parser.add_argument("--json_outputfile", type=str,
                        help="Name of <graph>.json file to use.  If not in same folder as main, give absolute path",
                        default='sortedpairs_dgl_20231129.json')
    parser.add_argument("--hidden_layer_1_nodes", type=int, help="Autoencoder hidden layer 1 size (default 15)",
                        default=15)
    parser.add_argument("--hidden_layer_2_nodes", type=int, help="Autoencoder hidden layer 2 size (default 10)",
                        default=10)
    parser.add_argument("--p", type=float, help="Autoencoder dropout rate >0, <1", default=.2)
    parser.add_argument("--num_epochs", type=int, help="Number of training epochs, default is 100", default=100)
    args = parser.parse_args()
    main(args)
