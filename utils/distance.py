import numpy as np
from sklearn.preprocessing import normalize
import networkx as nx
import os
from tqdm import tqdm


def get_distance_matrix(G1, G2, anchor_links, dataset, ratio, distance='rwr', use_attr=False, **kwargs):
    """
    Get distance matrix of the network
    :param G1: input graph 1
    :param G2: input graph 2
    :param anchor_links: anchor links
    :param dataset: dataset name
    :param ratio: training ratio
    :param distance: distance metric (e.g., rwr)
    :param use_attr: whether to use node attributes
    :return: distance matrix (num of nodes x num of anchor nodes)
    """
    if not os.path.exists(f'datasets/{distance}'):
        os.makedirs(f'datasets/{distance}')

    if os.path.exists(f'datasets/{distance}/{distance}_emb_{dataset}_{ratio:.1f}.npz') or \
            os.path.exists(f'datasets/{distance}/{distance}_emb_{dataset}_attr_{ratio:.1f}.npz'):
        print(f"Loading {distance} scores from datasets/{distance}/{distance}_emb_{dataset}_{ratio:.1f}.npz...",
              end=" ")
        if dataset == 'ACM-DBLP' and distance == 'otcost' and use_attr:
            data = np.load(f'datasets/{distance}/{distance}_emb_{dataset}_attr_{ratio:.1f}.npz')
        else:
            data = np.load(f'datasets/{distance}/{distance}_emb_{dataset}_{ratio:.1f}.npz')
        
        dists_score1, dists_score2 = data['dists_score1'], data['dists_score2']
        
        print("Done")
    else:
        assert f'{distance}_scores' in globals(), f'{distance}_scores function is not defined. IMPLEMENT IT FIRST!'
        dists_score1, dists_score2 = globals()[f'{distance}_scores'](G1, G2, anchor_links, dataset, ratio, **kwargs)
        if not os.path.exists(f'datasets/{distance}'):
            os.makedirs(f'datasets/{distance}')
        print(f"Saving {distance} scores to datasets/{distance}/{distance}_emb_{dataset}_{ratio:.1f}.npz...", end=" ")
        if dataset == 'ACM-DBLP' and distance == 'otcost' and use_attr:
            np.savez(f'datasets/{distance}/{distance}_emb_{dataset}_attr_{ratio:.1f}.npz',
                     dists_score1=dists_score1, dists_score2=dists_score2)
        else:
            np.savez(f'datasets/{distance}/{distance}_emb_{dataset}_{ratio:.1f}.npz',
                     dists_score1=dists_score1, dists_score2=dists_score2)
        print("Done")

    return dists_score1, dists_score2


def otrwr_scores(G1, G2, anchor_links, dataset, ratio, alpha=0.1, beta=0.15, gamma=0.8, **kwargs):
    """
    Compute initial node embedding vectors by OT-RWR
    :param G1: network G1, i.e., networkx graph
    :param G2: network G2, i.e., networkx graph
    :param anchor_links: anchor links
    :param dataset: dataset name
    :param ratio: training ratio
    :param alpha: hyperparameter
    :param beta: hyperparameter
    :param gamma: hyperparameter
    :return: otrwr_score1, otrwr_score2: OT-RWR vectors of the networks
    """

    r1, r2 = get_distance_matrix(G1, G2, anchor_links, dataset, ratio, distance='rwr', **kwargs)
    x1, x2 = G1.x, G2.x

    r1, r2 = normalize(r1, norm='l2', axis=1), normalize(r2, norm='l2', axis=1)
    x1, x2 = normalize(x1, norm='l2', axis=1), normalize(x2, norm='l2', axis=1)

    cost_node = alpha * np.exp(-(r1 @ r2.T)) + (1 - alpha) * np.exp(-(x1 @ x2.T))
    for i in tqdm(range(100), desc="Computing OT-RWR scores"):
        cost_node = (1 + beta) * cost_node


def otcost_scores(G1, G2, anchor_links, dataset, ratio, alpha=0.1, **kwargs):
    """
    Compute initial node embedding vectors by OT-Cost
    :param G1: network G1, i.e., networkx graph
    :param G2: network G2, i.e., networkx graph
    :param anchor_links: anchor links
    :param dataset: dataset name
    :param ratio: training ratio
    :param alpha: hyperparameter
    :return: otcost_score1, otcost_score2: OT-Cost vectors of the networks
    """

    r1, r2 = get_distance_matrix(G1, G2, anchor_links, dataset, ratio, distance='rwr', **kwargs)
    x1, x2 = G1.x, G2.x

    r1, r2 = normalize(r1, norm='l2', axis=1), normalize(r2, norm='l2', axis=1)
    x1, x2 = normalize(x1, norm='l2', axis=1), normalize(x2, norm='l2', axis=1)

    ot_cost = alpha * np.exp(-(r1 @ r2.T)) + (1 - alpha) * np.exp(-(x1 @ x2.T))
    ot_cost[(G1.anchor_nodes, G2.anchor_nodes)] = 0
    otcost_score1 = 1 - ot_cost[:, anchor_links[:, 1]]
    otcost_score2 = 1 - ot_cost.T[:, anchor_links[:, 0]]

    return otcost_score1, otcost_score2


def rwr_scores(G1, G2, anchor_links, *args, **kwargs):
    """
    Compute initial node embedding vectors by random walk with restart
    :param G1: network G1, i.e., networkx graph
    :param G2: network G2, i.e., networkx graph
    :param anchor_links: anchor links
    :return: rwr_score1, rwr_score2: RWR vectors of the networks
    """

    rwr_score1 = rwr_score(G1, anchor_links[:, 0], desc="Computing RWR scores for G1", **kwargs)
    rwr_score2 = rwr_score(G2, anchor_links[:, 1], desc="Computing RWR scores for G2", **kwargs)

    return rwr_score1, rwr_score2


def rwr_score(G, anchors, restart_prob=0.15, desc='Computing RWR scores'):
    """
    Random walk with restart for a single graph
    :param G: network G, i.e., networkx graph
    :param anchors: anchor nodes
    :param restart_prob: restart probability
    :param desc: description for tqdm
    :return: rwr: rwr vectors of the network
    """

    n = G.number_of_nodes()
    score = []

    for node in tqdm(anchors, desc=desc):
        s = nx.pagerank(G, personalization={node: 1}, alpha=1-restart_prob)
        s_list = [0] * n
        for k, v in s.items():
            s_list[k] = v
        score.append(s_list)

    rwr = np.array(score).T

    return rwr


def column_normalize(scores):
    """Normalize each column of the matrix so that the sum of each column equals 1."""
    column_sums = scores.sum(axis=0, keepdims=True)
    return scores / column_sums

def simRank_scores(G1, G2, anchor_links, *args, **kwargs):
    simRank_score1 = simRank_score(G1, anchor_links[:, 0], desc = "Computing simRank scores for G1", **kwargs)
    print(simRank_score1.shape)
    simRank_score2 = simRank_score(G2, anchor_links[:, 1], desc = "Computing simRank scores for G2", **kwargs)
    return simRank_score1, simRank_score2

def simRank_score(G, anchors, desc = "Computing simRank scores", max_iter = 100, r = 0.8):
    """

    """
    
    nodes = list(G.nodes())
    n = G.number_of_nodes()
    d = len(anchors)
    score = np.zeros((n, d))

    #node_idx = {node: i for i, node in enumerate(nodes)}
    print("start calculating the simrank for whole matrix")
    sim = nx.simrank_similarity(G, importance_factor = r, max_iterations= max_iter)
    node_idx = {node: idx for idx, node in enumerate(nodes)}

    for i, node in tqdm(enumerate(nodes)):
        for j, anchor in enumerate(anchors):
            score[i,j] = sim[node][anchor]
    print(score)
    
    return score


def conductance_scores(G1, G2, anchor_links, *args, **kwargs):
    conductance_score1 = conductance_score(G1, anchor_links[:, 0], desc = "Computing conductance scores for G1", **kwargs)
    conductance_score2 = conductance_score(G2, anchor_links[:, 1], desc = "Computing conductance scores for G2", **kwargs)
    return column_normalize(conductance_score1), column_normalize(conductance_score2)

def conductance_score(G, anchors, desc = "Computing conductance scores"):
    
    print(desc)
    L = nx.laplacian_matrix(G).toarray()
    # Compute the pseudoinverse of the Laplacian matrix
    L_pinv = np.linalg.pinv(L)
    
    nodes = list(G.nodes())
    n = len(nodes)
    d = len(anchors)
    conductance_matrix = np.zeros((n, d))
    
    node_idx = {node: i for i, node in enumerate(nodes)}
    
    # Calculate effective resistance and convert to conductance
    for i, node in tqdm(enumerate(nodes)):
        for j, anchor in enumerate(anchors):
            if node == anchor:
                conductance_matrix[i, j] = 0  # TODO: change here? Infinite conductance for self-loops
            else:
                resistance = L_pinv[node_idx[node], node_idx[node]] + L_pinv[node_idx[anchor], node_idx[anchor]] - 2 * L_pinv[node_idx[node], node_idx[anchor]]
                conductance_matrix[i, j] = 1 / resistance if resistance != 0 else 0
    
    return conductance_matrix
