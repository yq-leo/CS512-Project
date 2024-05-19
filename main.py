from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.tensorboard import SummaryWriter
from collections import defaultdict
import time

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
    G1, G2 = build_nx_graph(edge_index1, anchor_links[:, 0], x1), build_nx_graph(edge_index2, anchor_links[:, 0], x2)
    if args.use_gcn:
        assert args.use_attr, 'use_attr must be True when using gcn'
        gcn_output = np.load(f'gcn_out/{args.dataset}_gcn_results_{args.num_gcn_layers}_layers.npz')
        x1, x2 = gcn_output['x1'], gcn_output['x2']

    # compute distance metric scores (e.g. random walk with restart (rwr))
    dists_score1, dists_score2 = get_distance_matrix(G1, G2, anchor_links, args.dataset, args.ratio, args.distance)

    # device setting
    assert torch.cuda.is_available() or args.device == 'cpu', 'CUDA is not available'
    device = torch.device(args.device)

    # build PyG Data objects
    anchor1, anchor2 = anchor_links[:, 0], anchor_links[:, 1]
    G1_data = build_tg_graph(G1.number_of_nodes(), edge_index1, G1.x, anchor1, dists_score1).to(device)
    G2_data = build_tg_graph(G2.number_of_nodes(), edge_index2, G2.x, anchor2, dists_score2).to(device)

    # model setting
    model_settings = {
        "input_dim": G1_data.x.shape[1],
        "feature_dim": args.feat_dim,
        "anchor_dim": args.c * int(np.log2(anchor_links.shape[0])) ** 2 if args.random else anchor_links.shape[0],
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

    # train model
    if not os.path.exists('logs'):
        os.makedirs('logs')
    writer = SummaryWriter(log_path(args.dataset, args.use_attr))

    max_hits_list = defaultdict(list)
    max_mrr_list = []

    for run in range(args.runs):

        model = PGNN(**model_settings).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        criterion = globals()[f'{args.loss_func}Loss'](G1_data=G1_data,
                                                       G2_data=G2_data,
                                                       k=args.neg_sample_size,
                                                       margin=args.margin,
                                                       lambda_rank=args.lambda_rank,
                                                       dist_type=args.dist_type,
                                                       device=device).to(device)
        scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)

        G1_data.dists_max, G1_data.dists_argmax, G2_data.dists_max, G2_data.dists_argmax = (
            preselect_anchor(G1_data, G2_data, random=args.random, c=args.c, device=device))

        print("Training...")
        max_hits = defaultdict(int)
        max_mrr = 0
        for epoch in range(args.epochs):
            model.train()
            start = time.time()
            optimizer.zero_grad()
            out1, out2 = model(G1_data, G2_data)
            loss = criterion(out1=out1, out2=out2, anchor1=anchor1, anchor2=anchor2)
            loss.backward()
            optimizer.step()
            print(f'Epoch: {epoch + 1}, Loss: {loss.item():.6f}', end=" ")

            # testing
            out1_np = out1.detach().cpu().numpy()
            out2_np = out2.detach().cpu().numpy()
            dissimilarity = compute_distance_matrix(out1_np, out2_np, dist_type=args.dist_type, use_attr=args.use_attr, x1=x1, x2=x2)
            hits, mrr = compute_metrics(dissimilarity, test_pairs)
            end = time.time()
            print(f'{", ".join([f"Hits@{key}: {value:.4f}" for (key, value) in hits.items()])}, MRR: {mrr:.4f}, Time: {end - start:.2f}s')
            max_mrr = max(max_mrr, mrr)
            for key, value in hits.items():
                max_hits[key] = max(max_hits[key], value)

            writer.add_scalar('Loss', loss.item(), epoch)
            writer.add_scalar('MRR', mrr, epoch)
            for key, value in hits.items():
                writer.add_scalar(f'Hits/Hits@{key}', value, epoch)

            scheduler.step()

        for key, value in max_hits.items():
            max_hits_list[key].append(value)
        max_mrr_list.append(max_mrr)

    max_hits = {}
    max_hits_std = {}
    for key, value in max_hits_list.items():
        max_hits[key] = np.array(value).mean()
        max_hits_std[key] = np.array(value).std()
    max_mrr = np.array(max_mrr_list).mean()
    max_mrr_std = np.array(max_mrr_list).std()
    writer.add_hparams(vars(args), {'hparam/MRR': max_mrr, **{f'hparam/Hits@{key}': value for key, value in max_hits.items()}})

    writer.close()
