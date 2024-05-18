from argparse import ArgumentParser


def make_args():
    parser = ArgumentParser()
    parser.add_argument('--dataset', dest='dataset', type=str, default='ACM-DBLP',
                        choices=['noisy-cora1-cora2', 'ACM-DBLP', 'foursquare-twitter', 'phone-email', 'Douban', 'flickr-lastfm'],
                        help='datasets: noisy-cora1-cora2; ACM-DBLP; foursquare-twitter; phone-email; Douban; flickr-lastfm')
    parser.add_argument('--ratio', dest='ratio', type=float, default=0.2,
                        choices=[0.2],
                        help='training ratio: 0.1; 0.2')
    parser.add_argument('--use_attr', dest='use_attr', default=False, action='store_true',
                        help='use input node attributes')
    parser.add_argument('--distance', dest='distance', type=str, default='rwr',
                        choices=['rwr', 'otcost'],
                        help='distance metric: rwr; otcost')
    parser.add_argument('--gpu', dest='device', action='store_const', const='cuda', default='cpu',
                        help='use GPU')
    parser.add_argument('--lr', dest='lr', type=float, default=1e-3, help='learning_rate')
    parser.add_argument('--epochs', dest='epochs', type=int, default=250, help='number of epochs')

    parser.add_argument('--feat_dim', dest='feat_dim', type=int, default=128, help='feature dimension')
    parser.add_argument('--hidden_dim', dest='hidden_dim', type=int, default=128, help='hidden dimension')
    parser.add_argument('--out_dim', dest='out_dim', type=int, default=128, help='output dimension')
    parser.add_argument('--feature_pre', dest='feature_pre', default=False, action='store_true',
                        help='use feature pre-processing')
    parser.add_argument('--num_layers', dest='num_layers', type=int, default=1, help='number of layers')
    parser.add_argument('--use_dropout', dest='use_dropout', default=False, action='store_true',
                        help='use dropout')
    parser.add_argument('--dist_trainable', dest='dist_trainable', default=False, action='store_true',
                        help='train distance metric')
    parser.add_argument('--use_hidden', dest='use_hidden', default=False, action='store_true',
                        help='use hidden layer')
    parser.add_argument('--mcf_type', dest='mcf_type', type=str, default='anchor',
                        choices=['anchor', 'concat', 'sum', 'mean', 'max', 'min'], help='message computation function type')
    parser.add_argument('--agg_type', dest='agg_type', type=str, default='mean',
                        choices=['mean', 'sum', 'max', 'min'], help='aggregation function type')

    parser.add_argument('--loss', dest='loss_func', type=str, default='Ranking',
                        choices=['Ranking', 'Consistency', 'RegularizedRanking', 'WeightedRanking', 'WeightedRegularizedRanking'],
                        help='loss function type: Ranking; Consistency; RegularizedRanking')
    parser.add_argument('--neg', dest='neg_sample_size', type=int, default=500, help='negative sample size')
    parser.add_argument('--margin', dest='margin', type=float, default=10, help='margin parameter of ranking loss')
    parser.add_argument('--dist_type', dest='dist_type', type=str, default='l1',
                        choices=['l1', 'cosine'], help='distance metric type (between two embeddings)')
    parser.add_argument('--lambda_rank', dest='lambda_rank', type=float, default=0.5, help='ranking loss weight')
    parser.add_argument('--lambda_edge', dest='lambda_edge', type=float, default=1e-2, help='edge loss weight')
    parser.add_argument('--lambda_node', dest='lambda_node', type=float, default=1e-1, help='node loss weight')
    parser.add_argument('--lambda_align', dest='lambda_align', type=float, default=1, help='alignment loss weight')

    parser.add_argument('--random', dest='random', default=False, action='store_true',
                        help='use random anchors')
    parser.add_argument('--c', dest='c', type=int, default=1, help='c parameter of anchor dimension')

    parser.add_argument('--runs', dest='runs', type=int, default=1, help='number of runs')
    parser.add_argument('--use_gcn', dest='use_gcn', default=False, action='store_true', help='use GCN for ablation study')
    parser.add_argument('--num_gcn_layers', dest='num_gcn_layers', type=int, default=1, help='number of GCN layers')
    args = parser.parse_args()
    return args
