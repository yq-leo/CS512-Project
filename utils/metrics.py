import numpy as np
import scipy
import os


def compute_metrics(out1, out2, test_pairs, hit_top_ks=(1, 5, 10, 30, 50, 100)):
    """
    Compute metrics for the model (HITS@k, MRR)
    :param out1: output node embeddings of graph 1
    :param out2: output node embeddings of graph 2
    :param test_pairs: test pairs
    :param hit_top_ks: list of k for HITS@k
    :return:
        hits: HITS@k
        mrr: MRR
    """

    hits = {}

    anchor1_embeddings = out1[test_pairs[:, 0]]
    anchor2_embeddings = out2[test_pairs[:, 1]]

    distances1 = scipy.spatial.distance.cdist(anchor1_embeddings, out2, metric='cityblock')
    distances2 = scipy.spatial.distance.cdist(anchor2_embeddings, out1, metric='cityblock')
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