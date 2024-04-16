from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.tensorboard import SummaryWriter

from utils import *
from args import *
from model import *

if __name__ == '__main__':
    args = make_args()

    # check compatibility between dataset and use_attr
    if args.dataset == 'noisy-cora1-cora2':
        assert args.use_attr is True, 'noisy-cora1-cora2 requires using node attributes'
    elif args.dataset == 'foursquare-twitter' or args.dataset == 'phone-email':
        assert args.use_attr is False, f'{args.dataset} does not have node attributes'

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

    # model setting
    model_settings = {
        "input_dim": G1_data.x.shape[1],
        "feature_dim": args.feat_dim,
        "anchor_dim": args.c * int(np.log2(num_anchor_links)) ** 2 if args.random else num_anchor_links,
        "hidden_dim": args.hidden_dim,
        "output_dim": args.out_dim,
        "feature_pre": args.feature_pre,
        "num_layers": args.num_layers,
        "use_dropout": args.use_dropout,
        "dist_trainable": args.dist_trainable,
        "use_hidden": args.use_hidden,
        "mcf_type": args.mcf_type,
        "agg_type": args.agg_type
    }
    print(f"Model settings: {model_settings}")

    model = PGNN(**model_settings).to(device)
    # model = BRIGHT_U(anchor_dim, out_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = RankingLossL1(args.neg_sample_size, args.margin, args.dist_type).to(device)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)

    # train model
    if not os.path.exists('logs'):
        os.makedirs('logs')
    writer = SummaryWriter(log_path(args.dataset))

    for epoch in range(args.epochs):
        # training
        model.train()
        optimizer.zero_grad()
        G1_data.dists_max, G1_data.dists_argmax, G2_data.dists_max, G2_data.dists_argmax = (
            preselect_anchor(G1_data, G2_data, random=args.random, c=args.c, device=device))
        out1, out2 = model(G1_data, G2_data)
        loss = criterion(out1, out2, G1_data.anchor_nodes, G2_data.anchor_nodes)
        loss.backward()
        optimizer.step()
        print(f'Epoch: {epoch + 1}, Loss: {loss.item():.6f}', end=" ")

        # testing
        out1_np = out1.detach().cpu().numpy()
        out2_np = out2.detach().cpu().numpy()
        distances1 = compute_distance_matrix(out1_np[test_pairs[:, 0]], out2_np, dist_type=args.dist_type)
        distances2 = compute_distance_matrix(out2_np[test_pairs[:, 1]], out1_np, dist_type=args.dist_type)
        hits, mrr = compute_metrics(distances1, distances2, test_pairs)
        print(f'{", ".join([f"Hits@{key}: {value:.4f}" for (key, value) in hits.items()])}, MRR: {mrr:.4f}')

        writer.add_scalar('Loss', loss.item(), epoch)
        writer.add_scalar('MRR', mrr, epoch)
        for key, value in hits.items():
            writer.add_scalar(f'Hits@{key}', value, epoch)

        scheduler.step()

    writer.close()
