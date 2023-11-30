# -*- coding: utf-8 -*-
"""
Created on Fri Sep 15 05:43:58 2023
env: gnnv1

@author: cstan
"""
import torch
import dgl
import networkx as nx
import numpy as np
from sentence_transformers import SentenceTransformer

model = SentenceTransformer('all-MiniLM-L6-v2').cuda() #using a relatively smaller size model from the api
embedding_size = 384

def get_embedding(string):
   return model.encode(string,convert_to_tensor=True).cuda()

G = nx.Graph()
a_name = 'a'
a_properties = {'cat': 12, 'dog':11, 'other':[1,1,3], 'str_field': get_embedding('test1!'), 'str_field_': get_embedding("check it out")}

b_name = 'b'
b_properties = {'fish': 10, 'dog':13,'other':[1,1,5], 'str_field': get_embedding('Strings are in the field')}

#add source node
G.add_node(a_name, str_label = model.encode('test', convert_to_tensor=True).cuda(), int_label = 3, **a_properties)

#Add dest node
G.add_node(b_name, str_label = model.encode('test', convert_to_tensor=True).cuda(), int_label = 4, **b_properties)

# Add edge          
G.add_edge(a_name, b_name, relationship='related')

 #Get all attributes
all_attributes = set([k for n in G.nodes for k in G.nodes[n].keys()])
all_attributes_types = {k:type(G.nodes[n][k]) for n in G.nodes for k in G.nodes[n].keys()}

#Add missing attributes as Nans, or tensor of NaNs if attribute is string embedding.
for n in G.nodes:
    missing_attributes = all_attributes - set(G.nodes[n].keys())
    for attr in missing_attributes:

        #check if missing attribute needs to be a tensor of nans
        if all_attributes_types[attr] == torch.Tensor:
            G.nodes[n][attr] = ( torch.tensor(float('nan')).cuda()).repeat(embedding_size)
        else:
            G.nodes[n][attr] = np.nan

graph= dgl.from_networkx(G, node_attrs=list(all_attributes))
graph.ndata