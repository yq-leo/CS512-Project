import numpy as np
import scipy
import os

import torch
from torch_geometric.utils import to_dense_adj, degree


def compute_distance_matrix(embedding1, embedding2, dist_type='l1'):
    """
    Compute distance matrix between two sets of embeddings
    :param embedding1: node embeddings 1
    :param embedding2: node embeddings 2
    :param dist_type: distance function
    :return: distance matrix
    """
    assert dist_type in ['l1', 'cosine'], 'Similarity function not supported'

    if dist_type == 'l1':
        return scipy.spatial.distance.cdist(embedding1, embedding2, metric='cityblock')
    elif dist_type == 'cosine':
        return scipy.spatial.distance.cdist(embedding1, embedding2, metric='cosine')


def compute_ot_cost_matrix(G1_data, G2_data):
    """
    Compute optimal transport cost matrix between two sets of embeddings
    :param G1_data: PyG Data object for graph 1
    :param G2_data: PyG Data object for graph 2
    :return: cost_rwr: cost matrix
    """

    alpha = 1
    beta = 0.5
    gamma = 0.8

    r1, r2 = G1_data.dists, G2_data.dists
    x1, x2 = G1_data.x, G2_data.x
    cost_node = alpha * np.exp(-r1 @ r2.T) + (1-alpha) * np.exp(-x1 @ x2.T)

    A1, A2 = to_dense_adj(G1_data.edge_index)[0], to_dense_adj(G2_data.edge_index)[0]
    D1, D2 = torch.diag(degree(G1_data.edge_index[0])), torch.diag(degree(G2_data.edge_index[0]))
    W1, W2 = (torch.inverse(D1) @ A1).T, (torch.inverse(D2) @ A2).T

    cost_rwr = np.copy(cost_node)
    cnt = 0
    while True:
        cost_rwr_prev = np.copy(cost_rwr)
        cost_rwr = (1+beta) * cost_node + (1-beta) * gamma * (W1 @ cost_rwr @ W2.T)
        if np.linalg.norm(cost_rwr - cost_rwr_prev) < 1e-6:
            break
        cnt += 1
    print(f"OT Cost converged in {cnt} iterations")
    return cost_rwr


def compute_metrics(distances1, distances2, test_pairs, hit_top_ks=(1, 5, 10, 30, 50, 100)):
    """
    Compute metrics for the model (HITS@k, MRR)
    :param distances1: distance matrix 1 (G1 to G2)
    :param distances2: distance matrix 2 (G2 to G1)
    :param test_pairs: test pairs
    :param hit_top_ks: list of k for HITS@k
    :return:
        hits: HITS@k
        mrr: MRR
    """

    hits = {}

    ranks1 = np.argsort(distances1, axis=1)
    ranks2 = np.argsort(distances2, axis=1)

    signal1_hit = ranks1[:, :hit_top_ks[-1]] == np.expand_dims(test_pairs[:, 1], -1)
    signal2_hit = ranks2[:, :hit_top_ks[-1]] == np.expand_dims(test_pairs[:, 0], -1)
    for k in hit_top_ks:
        hits_ltr = np.sum(signal1_hit[:, :k]) / test_pairs.shape[0]
        hits_rtl = np.sum(signal2_hit[:, :k]) / test_pairs.shape[0]
        hits[k] = max(hits_ltr, hits_rtl)

    mrr_ltr = np.mean(1 / (np.where(ranks1 == np.expand_dims(test_pairs[:, 1], -1))[1] + 1))
    mrr_rtl = np.mean(1 / (np.where(ranks2 == np.expand_dims(test_pairs[:, 0], -1))[1] + 1))
    mrr = max(mrr_ltr, mrr_rtl)

    return hits, mrr


def log_path(dataset):
    if not os.path.exists(f'logs/{dataset}_results'):
        os.makedirs(f'logs/{dataset}_results')
    runs = len([f for f in os.listdir(f'logs/{dataset}_results') if os.path.isdir(f'logs/{dataset}_results/{f}')])
    return f'logs/{dataset}_results/run_{runs}'
