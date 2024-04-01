import numpy as np
import networkx as nx
import os
from tqdm import tqdm


def get_distance_matrix(G1, G2, anchor_links, dataset, ratio, distance):
    """
    Get distance matrix of the network
    :param G1: input graph 1
    :param G2: input graph 2
    :param anchor_links: anchor links
    :param dataset: dataset name
    :param ratio: training ratio
    :param distance: distance metric (e.g., rwr)
    :return: distance matrix (num of nodes x num of anchor nodes)
    """
    if not os.path.exists(f'datasets/{distance}'):
        os.makedirs(f'datasets/{distance}')

    if not os.path.exists(f'datasets/{distance}/{distance}_emb_{dataset}_{ratio:.1f}.npz'):
        assert f'{distance}_score' in globals(), f'{distance}_score function is not defined. IMPLEMENT IT FIRST!'
        dists_score1 = globals()[f'{distance}_score'](G1, anchors=anchor_links[:, 0], desc=f'Computing {distance} scores for G1')
        dists_score2 = globals()[f'{distance}_score'](G2, anchors=anchor_links[:, 1], desc=f'Computing {distance} scores for G2')
        if not os.path.exists(f'datasets/{distance}'):
            os.makedirs(f'datasets/{distance}')
        print(f"Saving {distance} scores to datasets/{distance}/{distance}_emb_{dataset}_{ratio:.1f}.npz...", end=" ")
        np.savez(f'datasets/{distance}/{distance}_emb_{dataset}_{ratio:.1f}.npz',
                 dists_score1=dists_score1, dists_score2=dists_score2)
        print("Done")
    else:
        print(f"Loading {distance} scores from datasets/{distance}/{distance}_emb_{dataset}_{ratio:.1f}.npz...", end=" ")
        data = np.load(f'datasets/{distance}/{distance}_emb_{dataset}_{ratio:.1f}.npz')
        dists_score1, dists_score2 = data['dists_score1'], data['dists_score2']
        print("Done")

    return dists_score1, dists_score2


def rwr_score(G, anchors, restart_prob=0.15, desc='Computing RWR scores'):
    """
    Compute initial node embedding vectors by random walk with restart
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
