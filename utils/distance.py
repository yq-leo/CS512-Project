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
