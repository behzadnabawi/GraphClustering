import numpy as np
from numpy import linalg as LA
import matplotlib.pyplot as plt
import networkx as nx
import os
def show_graph(adjacency_matrix, labels=None, node_size=500):
    color_map = {1: 'blue', 2: 'green', 3: 'red', 4: 'yellow'}
    colors = [color_map[x] for x in labels] if labels is not None else None
        
    rows, cols = np.where(adjacency_matrix == 1)
    edges = zip(rows.tolist(), cols.tolist())
    gr = nx.Graph()
    gr.add_edges_from(edges)
    nx.draw(gr, node_size=node_size, node_color=np.array(colors)[list(gr.nodes)] if labels is not None else None)
    plt.show()
adj = np.array([[0., 1., 0., 1., 0., 0., 0., 0., 0.],
                [1., 0., 1., 0., 0., 0., 0., 0., 0.],
                [0., 1., 0., 1., 1., 0., 0., 0., 0.],
                [1., 0., 1., 0., 0., 0., 0., 0., 0.],
                [0., 0., 1., 0., 0., 1. , 0., 1., 1.],
                [0., 0., 0., 0., 1., 0., 1., 0., 0.],
                [0., 0., 0., 0., 0., 1., 0., 1., 1.],
                [0., 0., 0., 0., 1., 0., 1., 0., 1.],
                [0., 0., 0., 0., 1., 0., 1., 1., 0.]])
def get_eigen_vectors(adj): 
    sum = adj.sum(axis=1).flatten()
    lapl = np.diag(sum) - adj
    eigen_values, eigen_vectors = LA.eig(lapl)
    idx = np.argsort(eigen_values)
    eigen_values = eigen_values[idx]
    eigen_vectors = eigen_vectors[:,idx]
    return eigen_vectors
eigen_vectors = get_eigen_vectors(adj)
labels = []
for i in range(eigen_vectors.shape[1]):
    if (eigen_vectors.T[1][i] < 0):
        labels += [1]
    else:
        labels += [2]
labels = []
for i in range(eigen_vectors.shape[1]):
    entry1 = eigen_vectors.T[1][i]
    entry2 = eigen_vectors.T[2][i]
    if (entry1 > 0 and entry2 > 0):
        labels += [1]
    elif(entry1 > 0 and entry2 < 0):
        labels += [2]
    elif(entry1 < 0 and entry2 > 0):
        labels += [3]
    else:
        labels += [4]
show_graph(adj, labels)

adj_final = np.zeros((100, 100))
path = __file__.replace("/PQ2.py", "") + "/data/data.txt"
file1 = open(path , 'r')
lines = file1.readlines()
print(len(lines))
for l in lines:
    i, j = l.split()
    adj_final[int(i) - 1, int(j) - 1] = 1
    adj_final[int(j) - 1, int(i) - 1] = 1
eigen_vectors = get_eigen_vectors(adj_final)
labels1 = []
for i in range(eigen_vectors.shape[1]):
    if (eigen_vectors.T[1][i] < 0):
        labels1 += [1]
    else:
        labels1 += [2]

labels2 = []
for i in range(eigen_vectors.shape[1]):
    entry1 = eigen_vectors.T[1][i]
    entry2 = eigen_vectors.T[2][i]
    if (entry1 > 0 and entry2 > 0):
        labels2 += [1]
    elif(entry1 > 0 and entry2 < 0):
        labels2 += [2]
    elif(entry1 < 0 and entry2 > 0):
        labels2 += [3]
    else:
        labels2 += [4]
#show_graph(adj_final, labels1, 100)
#show_graph(adj_final, labels2, 100)
print(eigen_vectors.T[1])
print(eigen_vectors.T[2])