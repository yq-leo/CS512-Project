import numpy as np
import networkx as nx
from tqdm import tqdm


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
