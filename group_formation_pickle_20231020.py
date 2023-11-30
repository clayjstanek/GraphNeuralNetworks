import logging
import time
import sys
import pickle

import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import torch
import dgl.nn as dglnn
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import seaborn as sns
import plotly.graph_objects as go
import joblib
import json

from sklearn.model_selection import train_test_split
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

NODELEN = 393
BATCHSIZE = 32
NUM_EPOCHS = 100

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger()

def create_graph(db_credentials, start_time, end_time):

    # read data from pickle files
    nodes_df = pd.read_pickle('./24h_nodes_df.pkl')
    edges_df = pd.read_pickle('./24h_edges_df.pkl')

    # re-index nodes starting from 0 to fix indexing problems
    graph = nx.from_pandas_edgelist(edges_df, 'src_id_entity', 'dst_id_entity')
    plot_graph(graph, "input_graph.html")

    # initialize sentence transformer
    transformer = SentenceTransformer('all-MiniLM-L6-v2')

    # convert data in dataframe to float format
    nodes_df.ner_category = nodes_df.ner_category.astype('category').cat.codes.astype(float) # TODO: needs to be mode
    nodes_df.gpe_country = nodes_df.gpe_country.astype('category').cat.codes.astype(float)
    nodes_df.lat = nodes_df.lat.fillna(0)
    nodes_df.long = nodes_df.long.fillna(0)
    nodes_df.is_misinfo = nodes_df.is_misinfo.astype(float)

    # run transformer on node labels and save those labels in a mapping table
    label_embeds = transformer.encode(nodes_df.label)
    nodes_df['label_embeds'] = label_embeds.tolist()
    label_map = nodes_df[['entity_id', 'label_embeds']]
    label_map = label_map.set_index('entity_id')

    # group node dataframe by entity_id and remove non-numeric data
    nodes_df = nodes_df.drop(columns=['label', 'label_embeds'])
    nodes_df = nodes_df.groupby('entity_id').mean()
    nodes_df = nodes_df.join(label_map)

    all_features = list(nodes_df.columns)

    # build dict of feats from dataframe to insert into graph
    feature_dict = {}
    for idx, node in nodes_df.iterrows():
        feature_dict[idx] = {}
        for feature in all_features:
            if feature != 'label' and feature != 'node_id':
                # if feature == 'label_embeds':
                feature_dict[idx][feature] = torch.tensor(node[feature])
                # else:
                #     feature_dict[idx][feature] = node[feature]

    # set graph attributes based on feature dict
    nx.set_node_attributes(graph, feature_dict)

    remove = [] # check for nodes without feature data
    for node in graph.nodes.data():
        if node[1] == {}:
            remove.append(node[0])
    graph.remove_nodes_from(remove)

    # reindex graph starting from 0
    reindexed_graph = nx.relabel.convert_node_labels_to_integers(graph, first_label=0, ordering='default')
    node_names = list(reindexed_graph.nodes())

    nodes_feature_list = []

    for node in list(reindexed_graph.nodes.data()):
        node_data = node[1]
        node_data['node_name'] = node[0]
        nodes_feature_list.append(node_data)

    adj_list = dict(reindexed_graph.adjacency())

    # Extract feature keys
    feature_keys = [key for key in nodes_feature_list[0].keys()
                    if (key != 'node_id' and key != 'label')]

    return feature_dict, graph, node_names, adj_list, feature_keys, nodes_feature_list

def Cosine_Similarity(values):
    similarity_matrix = cosine_similarity(values)
    with open('sim_matrix.pkl', 'wb') as out_file:
        pickle.dump(similarity_matrix, out_file)
    return similarity_matrix

def dict_to_tensor(data_dict):
    tensors = []
    if not isinstance(data_dict, torch.Tensor):
        try:
            for key, value in data_dict.items():
                if isinstance(value, int):  # Convert integers to tensor with shape (1,)
                    tensors.append(torch.tensor([value]))
                elif isinstance(value, torch.Tensor):
                    if value.dim() == 0:  # If tensor is a scalar, reshape it
                        value = value.reshape(1)
                    tensors.append(value)
                elif isinstance(value, float):  # Convert floats to tensor with shape (1,)
                    tensors.append(torch.tensor([value]))
                else:
                    raise ValueError(logger.warning(f"Unexpected data type for key {key}: {type(value)}"))
        except AttributeError:
            logger.critical("Could not find items in data_dict term")
        except Exception as e:
            logger.critical("dict_to_tensor exception")
        try:
            tensor_list = torch.cat(tensors)
            if tensor_list.size(dim=0) < 393:
                print(data_dict)
                print(tensor_list)
            return torch.cat(tensors)
        except Exception as e:
            logger.critical("Dict_to_Tensor return torch.cat(tensors) failed")
            try:
                if data_dict.dtype == torch.float32:
                    tensors = data_dict
                    if tensors.size(dim=0) < 393:
                        print(data_dict)
                        print(tensors)
                    return tensors
            except Exception as e:
                return torch.zeros(NODELEN, dtype=torch.float32)
    else:
        if data_dict.size(dim=0) < 393:
            print(data_dict)
        return data_dict

def CalculateFeaturesLength(feature_list, feature_keys):
    length = 0
    for key in enumerate(feature_keys):
        try:
            featurelen = len(feature_list[0][key[1]])
            length = length + featurelen
        except TypeError:
            logger.warning(
                f"TypeError: {key[1]} trying to take length of 0-d tensor")
            print(feature_list[0])
            length = length + 1
    return length

import torch.nn.functional as F


class AutoEncoder(nn.Module):
    def __init__(self, input_dim=NODELEN*2, embedding_dim= 128, hidden_dim1=356,
                 hidden_dim2=192, p=0.2):
        super(AutoEncoder, self).__init__()
        self.input_dim = input_dim
        self.embedding_dim = embedding_dim
        self.hidden_dim1 = hidden_dim1
        self.hidden_dim2 = hidden_dim2

        # Encoder layers
        self.encoder_fc1 = nn.Linear(input_dim, hidden_dim2)
        self.encoderdropout_fc1 = nn.Dropout(p)
#        self.encoder_fc2 = nn.Linear(input_dim, hidden_dim2)
#        self.encoderdropout_fc2 = nn.Dropout(p)
        self.encoder_fc3 = nn.Linear(hidden_dim2, embedding_dim)
         
         # Decoder layers
        self.decoder_fc1 = nn.Linear(embedding_dim, hidden_dim2)
        self.decoderdropout_fc1 = nn.Dropout(p)
#        self.decoder_fc2 = nn.Linear(hidden_dim2, hidden_dim1)
#        self.decoderdropout_fc2 = nn.Dropout(p)
        self.decoder_fc3 = nn.Linear(hidden_dim2, input_dim)
        self.optimizer = optim.Adam(self.parameters(), lr=0.01,
                                    betas=(0.9, 0.999), eps=1e-08, weight_decay=0)
        self.loss_fn = nn.MSELoss()

    def forward(self, x):
        # Encoding
        x = F.relu(self.encoder_fc1(x))
        x = self.encoderdropout_fc1(x)
#        x = F.relu(self.encoder_fc2(x))
#        x = self.encoderdropout_fc2(x)
        encoded = self.encoder_fc3(x)
        
        # Decoding
        x = F.relu(self.decoder_fc1(encoded))
        x = self.decoderdropout_fc1(x)
#        x = F.relu(self.decoder_fc2(x))
#        x = self.decoderdropout_fc2(x)
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

class GraphSAGEAutoencoder(nn.Module):
    def __init__(self, adjacency_list, nodes_feature_list, feature_keys, 
                 input_dim=NODELEN*2, embedding_dim= 128, hidden_dim1=356,
                 hidden_dim2=192, p=0.4):
        super(GraphSAGEAutoencoder, self).__init__()
        self.adj_list = adjacency_list
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

        neighbors = self.adj_list[node]
        # print("number of neighbors: " + str(len(neighbors)) + " at depth " + str(depth))
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
        aggregated_feat = torch.mean(torch.stack(neighborhood_tensors_list, dim=0), dim=0, dtype=torch.float32)
        aggregated_feat[0] = 0
        return aggregated_feat

    
    def generate_embedding(self, node, TRAIN=False):
        start_gen_time = time.time()
        # Generate embedding for a node using its features and aggregated neighbor features
        #        del node['node_name']
        concatenated_embedding = []
        aggregated_feat  = []
        if isinstance(node, tuple):
            node = node[1]
        node_feat = dict_to_tensor(self.features[node['node_name']])
        aggregated_feat = np.float32(self.aggregate_neighbors(node['node_name']))
        dummy_node = np.zeros((NODELEN,), dtype=np.float32)

        try:
            if (np.any(aggregated_feat)) & (aggregated_feat.size==1):
            # aggregated_feat has a nan and is length 1 so don't concatenate               
                embedding_tensor = node_feat
        except AttributeError:
            logger.critical("aggregated_feat is length 1, type float, value NAN")
            print("aggregated_feat is length 1, type float, value NAN")
            aggregated_feat = {}
            embedding_tensor = node_feat
        if isinstance(aggregated_feat, dict) and aggregated_feat:
            values_array = np.array(list(aggregated_feat.values()), dtype=np.float32)
            concatenated_embedding = np.concatenate(np.array(node_feat, dtype=np.float32), values_array)
        elif (aggregated_feat.size>1) and (len(np.nonzero(np.isnan(aggregated_feat))[0])==0):
            concatenated_embedding = np.concatenate([node_feat.numpy(), aggregated_feat])
        else:
            concatenated_embedding = np.concatenate([node_feat.numpy(), dummy_node])

        embedding_tensor = torch.tensor(concatenated_embedding, dtype=torch.float32)
        # zero out the node name since this is simply an integer that increments by one for each node
        embedding_tensor[NODELEN - 1] = 0.0
        embedding_tensor[(NODELEN * 2) - 1] = 0.0
        # Get reduced embedding
        # print("tensor size: " + str(embedding_tensor.size()))
        encoder, decoder = self.autoencoder.forward(embedding_tensor)
        end_gen_time = time.time() - start_gen_time
        print(f"Made node embedding in {end_gen_time} seconds")
        return encoder, decoder, embedding_tensor

def plot_graph(G, file_name):
    pos = nx.spring_layout(G)
    edge_x = []
    edge_y = []
    for edge in G.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x.append(x0)
        edge_x.append(x1)
        edge_x.append(None)
        edge_y.append(y0)
        edge_y.append(y1)
        edge_y.append(None)

    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=0.5, color='#888'),
        hoverinfo='none',
        mode='lines')

    node_x = []
    node_y = []
    for node in G.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)

    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode='markers',
        hoverinfo='text',
        marker=dict(
            showscale=True,
            # colorscale options
            #'Greys' | 'YlGnBu' | 'Greens' | 'YlOrRd' | 'Bluered' | 'RdBu' |
            #'Reds' | 'Blues' | 'Picnic' | 'Rainbow' | 'Portland' | 'Jet' |
            #'Hot' | 'Blackbody' | 'Earth' | 'Electric' | 'Viridis' |
            colorscale='YlGnBu',
            reversescale=True,
            color=[],
            size=10,
            colorbar=dict(
                thickness=15,
                title='Node Connections',
                xanchor='left',
                titleside='right'
            ),
            line_width=2))

    fig = go.Figure(data=[edge_trace, node_trace],
             layout=go.Layout(
                title='<br>Network graph made with Python',
                titlefont_size=16,
                showlegend=False,
                hovermode='closest',
                margin=dict(b=20,l=5,r=5,t=40),
                annotations=[ dict(
                    text="Python code: <a href='https://plotly.com/ipython-notebooks/network-graphs/'> https://plotly.com/ipython-notebooks/network-graphs/</a>",
                    showarrow=False,
                    xref="paper", yref="paper",
                    x=0.005, y=-0.002 ) ],
                xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                yaxis=dict(showgrid=False, zeroline=False, showticklabels=False))
                )
    fig.write_html(file_name)

def run_group_formation_module(db_credentials, start_time, end_time):
    logger.info('Beginning Group Formation Analysis')
    feature_dict, graph, node_names, adj_list, feature_keys, nodes_feature_list = create_graph(db_credentials, start_time, end_time)

    # modular_coms = nx.community.greedy_modularity_communities(graph, resolution=1.2)
    # mod_subgraphs = []
    # for com in sorted(modular_coms):
    #     mod_subgraphs.append(graph.subgraph(com))

    # i = 0
    # for subgraph in mod_subgraphs:
    #     _pos = nx.spring_layout(subgraph)
    #     labels = dict(subgraph.nodes(data="label"))
    #     plt.figure(figsize=(8, 8))
    #     nx.draw_networkx_edges(subgraph, _pos, alpha=0.2)
    #     nx.draw_networkx_nodes(subgraph, _pos, alpha=0.6)
    #     nx.draw_networkx_labels(subgraph, _pos, labels, font_size=14)
    #     plt.axis("off")
    #     plt.savefig(f'modular_subgraph_{i}.png')
    #     plt.clf()
    #     i += 1

    # louv_coms = nx.community.louvain_communities(graph)
    # louv_subgraphs = []
    # for com in sorted(louv_coms):
    #     louv_subgraphs.append(graph.subgraph(com))

    # i = 0
    # for subgraph in louv_subgraphs:
    #     _pos = nx.spring_layout(subgraph)
    #     labels = dict(subgraph.nodes(data="label"))
    #     plt.figure(figsize=(8, 8))
    #     nx.draw_networkx_edges(subgraph, _pos, alpha=0.2)
    #     nx.draw_networkx_nodes(subgraph, _pos, alpha=0.6)
    #     nx.draw_networkx_labels(subgraph, _pos, labels, font_size=14)
    #     plt.axis("off")
    #     plt.savefig(f'louv_subgraph_{i}.png')
    #     plt.clf()
    #     i += 1





    node_len=(CalculateFeaturesLength(nodes_feature_list, feature_keys))

    # Initialize GraphSAGE with our adjacency list and features
    logger.info('Initializing GraphSAGE')
    graphsage = GraphSAGEAutoencoder(adj_list, nodes_feature_list, feature_keys,
                                     input_dim=node_len*2, embedding_dim=128, hidden_dim1=300,
                                     hidden_dim2=200, p=0.2)

    # Generate embeddings for all nodes
    # embeddings = {node: graphsage.generate_embedding(node) for node in node_list}
    logger.info('Generating initial embeddings')
    start_time = time.time()
    embeddings_tuple = {index: graphsage.generate_embedding(node_features,True) for index, node_features in enumerate(nodes_feature_list)}
    init_embed_time = time.time() - start_time
    print(f"Completed embeddings in {init_embed_time} seconds")
    logger.debug("Complete embeddings generation")

    # Extracting the arrays
    NODES = len(nodes_feature_list)
    # arrays_384 = [embeddings_tuple[i][0] for i in range(NODES)]
    # arrays_35898_first = [embeddings_tuple[i][1] for i in range(NODES)]
    node_embeddings = [embeddings_tuple[i][2] for i in range(NODES)]

    # print(len(arrays_384), len(arrays_35898_first), len(node_embeddings))
    # logger.debug(f"GraphSageAutoEncoder.generate_embedding returns lists: {len(arrays_384)}, {len(arrays_35898_first)}, {len(node_embeddings)}")
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
    logger.debug("Training Beginning")
    start_time = time.time()
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    try:
        graphsage.autoencoder.train(dataloader, NUM_EPOCHS)
    except Exception as e:
        logger.error("GraphsageAutoEncoder.train experienced unrecoverable error")
        logging.shutdown()
        sys.exit(10)
    training_time = time.time() - start_time
    print(f"Trained the network in {training_time} seconds")

    #now that neural network is trained, go back and create embeddings, but use results out of autoencoder for dim reduction
    logger.info('Generating final embeddings')
    start_time = time.time()
    embeddings_final = {index: graphsage.generate_embedding(node_features,True) for index, node_features in enumerate(nodes_feature_list)}
    final_embed_time = time.time() - start_time
    print(f"Generated final embeddings in {final_embed_time} seconds")
    NODES = len(nodes_feature_list )
    arrays_384 = [embeddings_final[i][0] for i in range(NODES)]
    """
    The next step is to measure the similarity between these embeddings to detect correlated event patterns. 
    A common way to measure similarity between vectors is using the cosine similarity. We'll compute pairwise 
    cosine similarities between the embeddings to identify the IPs with correlated event patterns.
    """
    arrays_384 = torch.stack(arrays_384)
    # Compute pairwise cosine similarities
    similarity_matrix = Cosine_Similarity(list(arrays_384.detach().numpy()))
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
    dump(sorted_node_pairs, './SortedNodePairs_dgl_20230920.joblib')
    """
    Here's the pairwise cosine similarity between the embeddings of the nodes:
    You can set a threshold on similarity values to determine which pairs are 
    considered to have "correlated" event patterns.

    This approach provides a way to detect IP addresses with correlated activity patterns based 
    on their event features and their relationships in the network. By applying this method to a 
    real-world IP network with more features and larger datasets, you can gain insights into 
    correlated activities and potential anomalies in the network.
    """

    # Load data from joblib file
    joblib_file = './SortedNodePairs_dgl_20230920.joblib'
    data = joblib.load(joblib_file)

    # Convert data to be JSON serializable
    json_serializable_data = convert_to_json_serializable(data)

    # Save data to JSON file
    json_file = 'SortedNodePairs_DGL_20230920.json'
    with open(json_file, 'w') as f:
        json.dump(json_serializable_data, f)

    print(f"Data saved to {json_file}")





    # # print(embeddings)

    # # Convert the aggregated features to a numpy array
    # feature_vectors = np.array(list(embeddings.values()))

    # # Compute pairwise cosine similarities
    # similarities = cosine_similarity(feature_vectors)
    # print(similarities)

    plt.imshow(similarity_matrix, cmap='autumn')
    plt.title('Cosine Similarities of Nodes')
    plt.savefig('heatmap.png')
    plt.clf()

    df_adj = pd.DataFrame(similarity_matrix, index=node_names, columns=node_names)
    print(df_adj)

    adj_graph = nx.from_pandas_adjacency(df_adj)

    edge_weights = nx.get_edge_attributes(adj_graph,'weight')
    adj_graph.remove_edges_from((e for e, w in edge_weights.items() if w < .8))
    adj_graph.remove_edges_from((e for e, w in edge_weights.items() if w == 1.0))
    nx.draw(adj_graph)
    plt.savefig('test.png')
    plot_graph(adj_graph, "communities.html")

    # model_DB = DBSCAN(eps = 0.005, min_samples = 20, metric = 'cosine').fit(similarities)
    # labels = model_DB.labels_
    # print(labels)

    # unique, counts = np.unique(labels, return_counts = True)
    # print(dict(zip(unique, counts)))

    # p = sns.scatterplot(data = df_adj, x = "src node", y = "dst node", hue = model_DB.labels_, legend = "full", palette = "deep")
    # sns.move_legend(p, "upper right", bbox_to_anchor = (1.17, 1.), title = 'Clusters')
    # plt.savefig('scatter.png')

    return 200

def convert_to_json_serializable(item):
    if isinstance(item, dict):
        return {convert_to_json_serializable(key): convert_to_json_serializable(value) for key, value in item.items()}
    elif isinstance(item, tuple):
        return str(item)
    elif isinstance(item, np.float32):
        return float(item)
    else:
        return item
    
if __name__ == "__main__":
    run_group_formation_module(0, 0, 0)