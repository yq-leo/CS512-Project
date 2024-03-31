import os
import networkx as nx
import torch.cuda

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
    G1, G2 = build_nxgraph(edge_index1, x1), build_nxgraph(edge_index2, x2)

    # compute / load random walk with restart (RWR) scores for nodes
    if not os.path.exists(f'datasets/rwr/rwr_emb_{args.dataset}_{args.ratio:.1f}.npz'):
        rwr_score1 = rwr_score(G1, anchors=anchor_links[:, 0], desc='Computing RWR scores for G1')
        rwr_score2 = rwr_score(G2, anchors=anchor_links[:, 1], desc='Computing RWR scores for G2')
        if not os.path.exists('datasets/rwr'):
            os.makedirs('datasets/rwr')
        print(f"Saving RWR scores to datasets/rwr/rwr_emb_{args.dataset}_{args.ratio:.1f}.npz")
        np.savez(f'datasets/rwr/rwr_emb_{args.dataset}_{args.ratio:.1f}.npz', rwr_score1=rwr_score1, rwr_score2=rwr_score2)
    else:
        print(f"Loading RWR scores from datasets/rwr/rwr_emb_{args.dataset}_{args.ratio:.1f}.npz")
        data = np.load(f'datasets/rwr/rwr_emb_{args.dataset}_{args.ratio:.1f}.npz')
        rwr_score1, rwr_score2 = data['rwr_score1'], data['rwr_score2']

    # device setting
    assert torch.cuda.is_available() or args.device == 'cpu', 'CUDA is not available'
