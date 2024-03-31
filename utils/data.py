import numpy as np
import networkx as nx


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


def build_nxgraph(edge_index, x=None):
    """
    Build a networkx graph from edge list and node attributes.
    :param edge_index: edge list of the graph
    :param x: node attributes of the graph
    :return: a networkx graph
    """

    G = nx.Graph()
    if x is not None:
        G.add_nodes_from(np.arange(x.shape[0]))
    G.add_edges_from(edge_index)
    for edge in G.edges():
        G[edge[0]][edge[1]]['weight'] = 1
    return G
