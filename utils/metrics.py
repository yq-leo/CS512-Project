import numpy as np
import scipy
import os


def compute_distance_matrix(embedding1, embedding2, dist_type='l1', use_attr=False, x1=None, x2=None):
    """
    Compute distance matrix between two sets of embeddings
    :param embedding1: node embeddings of graph 1
    :param embedding2: node embeddings of graph 2
    :param dist_type: distance function
    :param use_attr: whether it's attributed network
    :param x1: node attributes of graph 1
    :param x2: node attributes of graph 2
    :return: distance matrix
    """
    assert dist_type in ['l1', 'cosine'], 'Similarity function not supported'

    dists = None
    if dist_type == 'l1':
        dists = scipy.spatial.distance.cdist(embedding1, embedding2, metric='cityblock')
    elif dist_type == 'cosine':
        dists = scipy.spatial.distance.cdist(embedding1, embedding2, metric='cosine')

    if use_attr:
        assert x1 is not None and x2 is not None, 'Node attributes are not provided'
        x1 = x1 / np.linalg.norm(x1, ord=2, axis=1, keepdims=True)
        x2 = x2 / np.linalg.norm(x2, ord=2, axis=1, keepdims=True)
        dists = 0.1 * np.exp(1 - dists) + 0.9 * np.exp(-(x1 @ x2.T))

    return dists


def compute_metrics(dissimilarity, test_pairs, hit_top_ks=(1, 5, 10, 30, 50, 100)):
    """
    Compute metrics for the model (HITS@k, MRR)
    :param dissimilarity: dissimilarity matrix (n1 x n2)
    :param test_pairs: test pairs
    :param hit_top_ks: list of k for HITS@k
    :return:
        hits: HITS@k
        mrr: MRR
    """

    distances1 = dissimilarity[test_pairs[:, 0]]
    distances2 = dissimilarity.T[test_pairs[:, 1]]

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


def log_path(dataset, use_attr=False):
    if dataset == 'ACM-DBLP':
        dataset = 'ACM-DBLP_attr' if use_attr else 'ACM-DBLP'

    if not os.path.exists(f'logs/{dataset}_results'):
        os.makedirs(f'logs/{dataset}_results')
    runs = len([f for f in os.listdir(f'logs/{dataset}_results') if os.path.isdir(f'logs/{dataset}_results/{f}')])
    runs_str = str(runs).zfill(3)
    return f'logs/{dataset}_results/run_{runs_str}'
