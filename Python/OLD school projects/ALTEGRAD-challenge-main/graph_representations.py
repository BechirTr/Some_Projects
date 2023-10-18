import pandas as pd
import numpy as np
import networkx as nx
from nodevectors import Node2Vec, ProNE # import Node2Vec model to encode the graph


# load the graph
G = nx.read_edgelist('data/collaboration_network.edgelist', delimiter=' ', nodetype=int)
n_nodes = G.number_of_nodes()
n_edges = G.number_of_edges()
print('Number of nodes:', n_nodes)
print('Number of edges:', n_edges)

# Create an embedding of the graph
g2v = Node2Vec(n_components= 32, walklen= 10)
#g2v = ProNE(step = 6,n_components= 32) # other model that we tried
# Fit the model
g2v.fit(G)

# Get the embedding of each node
Embeddings = {}
for u in G.nodes:
  Embeddings[u] = g2v.predict(u)

# transform the embedding to pandas array and save it
df = pd.DataFrame.from_dict(Embeddings)
df.to_csv('data/graph_embedding.csv', index=False)
