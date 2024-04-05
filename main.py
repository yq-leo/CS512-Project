import os
import networkx as nx
import torch.cuda
from torch_geometric.data import Data

from utils import *
from args import *

if __name__ == '__main__':
    args = make_args()

    # check compatibility between dataset and use_attr
    if args.dataset == 'noisy-cora1-cora2':
        assert args.use_attr is True, 'noisy-cora1-cora2 requires using node attributes'
    elif args.dataset == 'fourquare-twitter':
        assert args.use_attr is False, 'foursquare-twitter does not have node attributes'

    # load data and build networkx graphs
    print("Loading data...")
    edge_index1, edge_index2, x1, x2, anchor_links, test_pairs = load_data(f"datasets/{args.dataset}", args.ratio, args.use_attr)
    G1, G2 = build_nx_graph(edge_index1, x1), build_nx_graph(edge_index2, x2)

    # compute distance metric scores (e.g. random walk with restart (rwr))
    dists_score1, dists_score2 = get_distance_matrix(G1, G2, anchor_links, args.dataset, args.ratio, args.distance)

    # device setting
    assert torch.cuda.is_available() or args.device == 'cpu', 'CUDA is not available'

    # build PyG Data objects
    G1_data = build_tg_graph(edge_index1, x1, anchor_links[:, 0], dists_score1)
    G2_data = build_tg_graph(edge_index2, x2, anchor_links[:, 1], dists_score2)

    G1_data.dists_max, G1_data.dists_argmax, G2_data.dists_max, G2_data.dists_argmax = (
        preselect_anchor(G1_data, G2_data, random=False, device=args.device))
