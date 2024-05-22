from utils import *
from args import *
from model import *

if __name__ == '__main__':
    args = make_args()

    # check compatibility between dataset and use_attr
    if args.dataset == 'noisy-cora1-cora2':
        assert args.use_attr is True, 'noisy-cora1-cora2 requires using node attributes'
    elif args.dataset == 'foursquare-twitter':
        assert args.use_attr is False, 'foursquare-twitter does not have node attributes'

    # load data and build networkx graphs
    print("Loading data...")
    edge_index1, edge_index2, x1, x2, anchor_links, test_pairs = load_data(f"datasets/{args.dataset}", args.ratio, args.use_attr)
    G1, G2 = build_nx_graph(edge_index1, x1), build_nx_graph(edge_index2, x2)
    num_anchor_links = anchor_links.shape[0]

    # compute distance metric scores (e.g. random walk with restart (rwr))
    dists_score1, dists_score2 = get_distance_matrix(G1, G2, anchor_links, args.dataset, args.ratio, args.distance)

    # device setting
    assert torch.cuda.is_available() or args.device == 'cpu', 'CUDA is not available'
    device = torch.device(args.device)

    # build PyG Data objects
    G1_data = build_tg_graph(G1.number_of_nodes(), edge_index1, x1, anchor_links[:, 0], dists_score1).to(device)
    G2_data = build_tg_graph(G2.number_of_nodes(), edge_index2, x2, anchor_links[:, 1], dists_score2).to(device)
    if args.use_gcn:
        gcn_output = np.load(f'gcn_out/{args.dataset}_gcn_results_{args.num_gcn_layers}_layers.npz')
        G1_data.x = torch.tensor(gcn_output['x1'], dtype=torch.float).to(device)
        G2_data.x = torch.tensor(gcn_output['x2'], dtype=torch.float).to(device)

    # compute OT cost
    print("Computing OT cost...")
    for alpha in [0.1, 0.2, 0.3, 0.4, 0.5]:
        cost_rwr = compute_ot_cost_matrix(G1_data, G2_data, alpha).cpu().numpy()
        hits, mrr = compute_metrics(cost_rwr[test_pairs[:, 0]], cost_rwr.T[test_pairs[:, 1]], test_pairs)
        print(f'alpha={alpha}-{", ".join([f"Hits@{key}: {value:.4f}" for (key, value) in hits.items()])}, MRR: {mrr:.4f}')
