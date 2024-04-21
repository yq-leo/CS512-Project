import numpy as np
import networkx as nx
import torch
from torch_geometric.data import Data
from torch_geometric.utils import to_dense_adj, degree


def load_data(file_name, p, use_attr):
    """
    Load dataset.
    :param file_name: file name of the dataset
    :param p: training ratio
    :param use_attr: whether to use input node attributes
    :return:
        edge_index1, edge_index2: edge list of graph G1, G2
        x1, x2: input node attributes of graph G1, G2
        anchor_links: training node alignments, i.e., anchor links
        test_pairs: test node alignments
    """

    data = np.load(f'{file_name}_{p:.1f}.npz')
    edge_index1, edge_index2 = data['edge_index1'].T.astype(np.int64), data['edge_index2'].T.astype(np.int64)
    anchor_links, test_pairs = data['pos_pairs'].astype(np.int64), data['test_pairs'].astype(np.int64)
    if use_attr:
        x1, x2 = data['x1'].astype(np.float32), data['x2'].astype(np.float32)
    else:
        x1, x2 = None, None

    return edge_index1, edge_index2, x1, x2, anchor_links, test_pairs


def build_nx_graph(edge_index, anchor_nodes, x=None):
    """
    Build a networkx graph from edge list and node attributes.
    :param edge_index: edge list of the graph
    :param anchor_nodes: anchor nodes
    :param x: node attributes of the graph
    :return: a networkx graph
    """

    G = nx.Graph()
    if x is not None:
        G.add_nodes_from(np.arange(x.shape[0]))
        G.x = x
    G.add_edges_from(edge_index)
    if x is None:
        G.x = np.ones((G.number_of_nodes(), 1))
    for edge in G.edges():
        G[edge[0]][edge[1]]['weight'] = 1
    G.anchor_nodes = anchor_nodes
    return G


def build_tg_graph(num_nodes, edge_index, x, anchor_nodes, dists):
    """
    Build a PyG Data object from edge list and node attributes.
    :param num_nodes: number of nodes in the graph
    :param edge_index: edge list of the graph
    :param x: node attributes of the graph
    :param anchor_nodes: anchor nodes
    :param dists: distance metric scores (num nodes x num anchor nodes)
    :return: a PyG Data object
    """

    edge_index_tensor = torch.tensor(edge_index.T, dtype=torch.long)
    x_tensor = torch.tensor(x, dtype=torch.float) if x is not None else torch.ones((num_nodes, 1), dtype=torch.float)
    data = Data(x=x_tensor, edge_index=edge_index_tensor)
    data.anchor_nodes = torch.from_numpy(anchor_nodes).long()
    data.dists = torch.from_numpy(dists).float()
    data.adj = to_dense_adj(edge_index_tensor).squeeze(0)
    return data
