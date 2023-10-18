import pandas as pd
import numpy as np
import networkx as nx
from tqdm import tqdm

# load the graph    
G = nx.read_edgelist('data/collaboration_network.edgelist', delimiter=' ', nodetype=int)
n_nodes = G.number_of_nodes()
n_edges = G.number_of_edges() 
print('Number of nodes:', n_nodes)
print('Number of edges:', n_edges)

#### computes structural features for each node
core_number = nx.core_number(G) # It's core number 
degree_centrality = nx.degree_centrality(G) # It's degree of centrality
avg_neighbor_degree = nx.average_neighbor_degree(G) # The average degree of it's neighbors
eig_centr = nx.eigenvector_centrality(G) # The eigenvector centrality
Bctr = nx.betweenness_centrality(G, k= 100) # The betweenness centrality with only a sub graph of k nodes
hits = nx.hits(G, max_iter= 1000) # compute the hubs and authorities of a graph (in our case of undirected graph hubs = authorities)
hits = hits[0]

annd = {} # The average degree of the average neighbors degree of neighbors
for node in tqdm(avg_neighbor_degree.keys()):
    annd[node] = np.mean([avg_neighbor_degree[neighbor] for neighbor in G.neighbors(node)])

cnn = {} # The average core number of the neighbors of neighbors
for node in tqdm(core_number.keys()):
    cnn[node] = np.mean([core_number[neighbor] for neighbor in G.neighbors(node)])


# create the feature matrix. each node is represented as a vector of 11 columns:
# (0) node ID.
# (1) its degree.
# (2) its core number.
# (3) the average degree of its neighbors.
# (4) the total degree of the node neighbors.
# (5) The std of the node neighbors degree.
# (6) The average degree of the average neighbors degree of neighbors.
# (7) The average core number of the neighbors of neighbors
# (8) The eigenvector centrality.
# (9) The clustering coefficient.
# (10) The betweenness centrality.
# (11) The hubs value of a node.
# (12) The maximum of the neighbors degree.
# (13) The degree centrality of a node.

ftr_matrix = np.zeros((n_nodes, 14))
nodes = list(G.nodes())
for i,node in tqdm(enumerate(nodes)):
    ftr_matrix[i,0] = node
    ftr_matrix[i,1] = G.degree(node)
    ftr_matrix[i,2] = core_number[node]
    ftr_matrix[i,3] = avg_neighbor_degree[node]
    ftr_matrix[i,4] = np.sum([G.degree(n) for n in G.neighbors(node)])
    ftr_matrix[i,5] = np.std([G.degree(n) for n in G.neighbors(node)])
    ftr_matrix[i,6] = annd[node]
    ftr_matrix[i,7] = cnn[node]
    ftr_matrix[i,8] = eig_centr[node]
    ftr_matrix[i,9] = nx.clustering(G, node)
    ftr_matrix[i,10] = Bctr[node]
    ftr_matrix[i,11] = hits[node]
    for neighbor in G.neighbors(node):
        ftr_matrix[i,12] = max(ftr_matrix[i,12], G.degree(neighbor))
    ftr_matrix[i,13] = degree_centrality[node]

# write the ftr_matrix in a csv file
df = open('data/graph_features.csv', "w")

for i, author in enumerate(nodes):
    df.write(str(author)+","+",".join(map(lambda x:"{:.8f}".format(round(x, 8)), ftr_matrix[i,:]))+"\n")
df.close()