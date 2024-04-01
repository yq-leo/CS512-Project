from argparse import ArgumentParser


def make_args():
    parser = ArgumentParser()
    parser.add_argument('--dataset', dest='dataset', type=str, default='noisy-cora1-cora2',
                        choices=['noisy-cora1-cora2', 'ACM-DBLP', 'foursquare-twitter'],
                        help='dataset name: noisy-cora1-cora2; ACM-DBLP; foursquare-twitter')
    parser.add_argument('--ratio', dest='ratio', type=float, default=0.2,
                        choices=[0.2],
                        help='training ratio: 0.1; 0.2')
    parser.add_argument('--use_attr', dest='use_attr', default=True, action='store_true',
                        help='use input node attributes')
    parser.add_argument('--distance', dest='distance', type=str, default='rwr',
                        choices=['rwr'],
                        help='distance metric: rwr')
    parser.add_argument('--gpu', dest='device', action='store_const', const='cuda:0', default='cpu',
                        help='use GPU')
    args = parser.parse_args()
    return args
