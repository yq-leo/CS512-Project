import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
import numpy as np
import scipy

from utils import compute_distance_matrix


class PGNNLayer(torch.nn.Module):
    def __init__(self, input_dim, anchor_dim, output_dim, dist_trainable=False, use_hidden=False,
                 mcf_type='default', agg_type='mean', **kwargs):
        """
        One PGNN Layer
        :param input_dim: input feature dimension
        :param anchor_dim: num of anchor nodes
        :param output_dim: output feature dimension
        :param dist_trainable: whether to use trainable distance metric scores
        :param mcf_type: type of message computation function (e.g. default, concat, mean, etc.)
        :param agg_type: type of message aggregation function (e.g. mean, sum, max, etc.)
        :param use_hidden: whether to use SLP after message computation function F
        :param kwargs: optional arguments
        """
        super(PGNNLayer, self).__init__()

        self.input_dim = input_dim
        self.anchor_dim = anchor_dim
        self.message_dim = input_dim if mcf_type != 'concat' else input_dim * 2
        self.output_dim = output_dim
        self.dist_trainable = dist_trainable
        self.mcf_type = mcf_type
        self.agg_type = agg_type
        self.use_hidden = use_hidden

        if self.dist_trainable:
            self.dist_compute = Nonlinear(1, output_dim, 1)

        self.linear_hidden = nn.Linear(self.message_dim, self.input_dim) if self.use_hidden else None
        self.linear_final = nn.Linear(self.anchor_dim, self.output_dim)
        self.act = nn.ReLU()

        # for m in self.modules():
        #     if isinstance(m, nn.Linear):
        #         m.weight.data = nn.init.xavier_uniform_(m.weight.data, gain=nn.init.calculate_gain('relu'))
        #         if m.bias is not None:
        #             m.bias.data = nn.init.constant_(m.bias.data, 0.0)

    def forward(self, x1, x2, dists_max_1, dists_max_2, dists_argmax_1, dists_argmax_2):
        if self.dist_trainable:
            dists_max_1 = self.dist_compute(dists_max_1.unsqueeze(-1)).squeeze()
            dists_max_2 = self.dist_compute(dists_max_2.unsqueeze(-1)).squeeze()

        anchor_features_1 = x1[dists_argmax_1, :]
        self_features_1 = x1.unsqueeze(1).repeat(1, dists_max_1.shape[1], 1)
        messages_1 = self.mcf(self_features_1, anchor_features_1, dists_max_1)
        del anchor_features_1, self_features_1

        anchor_features_2 = x2[dists_argmax_2, :]
        self_features_2 = x2.unsqueeze(1).repeat(1, dists_max_2.shape[1], 1)
        messages_2 = self.mcf(self_features_2, anchor_features_2, dists_max_2)
        del anchor_features_2, self_features_2

        if self.use_hidden:
            assert self.linear_hidden is not None, 'Hidden layer is not defined'
            messages_1 = self.linear_hidden(messages_1).squeeze()
            messages_1 = self.act(messages_1)
            messages_2 = self.linear_hidden(messages_2).squeeze()
            messages_2 = self.act(messages_2)

        out1_position = self.linear_final(torch.sum(messages_1, dim=2))  # zv (output)
        out1_structure = self.agg(messages_1)  # hv (feed to the next layer)
        out2_position = self.linear_final(torch.sum(messages_2, dim=2))
        out2_structure = self.agg(messages_2)

        return out1_position, out1_structure, out2_position, out2_structure

    def mcf(self, node_feat, anchor_feat, distances):
        """
        Message Computation Function F
        :param node_feat: node features (hv)
        :param anchor_feat: anchorset features (hu)
        :param distances: distances metric scores (s(v, u))
        :return:
            messages: messages F(v, u, hv, hu)
        """
        assert self.mcf_type in ['anchor', 'concat', 'sum', 'mean', 'max', 'min'], 'Invalid MCF type'

        if self.mcf_type == 'anchor':
            return distances.unsqueeze(-1) * anchor_feat
        elif self.mcf_type == 'concat':
            return distances.unsqueeze(-1) * torch.cat((node_feat, anchor_feat), dim=-1)
        elif self.mcf_type == 'sum':
            return distances.unsqueeze(-1) * torch.sum(torch.stack((node_feat, anchor_feat), dim=0), dim=0)
        elif self.mcf_type == 'mean':
            return distances.unsqueeze(-1) * torch.mean(torch.stack((node_feat, anchor_feat), dim=0), dim=0)
        elif self.mcf_type == 'max':
            return distances.unsqueeze(-1) * torch.max(torch.stack((node_feat, anchor_feat), dim=0), dim=0)[0]
        elif self.mcf_type == 'min':
            return distances.unsqueeze(-1) * torch.min(torch.stack((node_feat, anchor_feat), dim=0), dim=0)[0]

    def agg(self, messages):
        """
        Message Aggregation Function AGG
        :param messages: message matrix Mv
        :return:
            out: aggregated messages
        """
        assert self.agg_type in ['mean', 'sum', 'max', 'min'], 'Invalid AGG type'

        if self.agg_type == 'mean':
            return torch.mean(messages, dim=1)
        elif self.agg_type == 'sum':
            return torch.sum(messages, dim=1)
        elif self.agg_type == 'max':
            return torch.max(messages, dim=1)[0]
        elif self.agg_type == 'min':
            return torch.min(messages, dim=1)[0]


class Nonlinear(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(Nonlinear, self).__init__()

        self.linear1 = nn.Linear(input_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, output_dim)

        self.act = nn.ReLU()

    def forward(self, x):
        x = self.linear1(x)
        x = self.act(x)
        x = self.linear2(x)
        return x


class PGNN(torch.nn.Module):
    def __init__(self, input_dim, feature_dim, anchor_dim, hidden_dim, output_dim,
                 feature_pre=False, num_layers=2, use_dropout=False, **kwargs):
        super(PGNN, self).__init__()
        self.feature_pre = feature_pre
        self.num_layers = num_layers
        self.use_dropout = use_dropout
        if num_layers == 1:
            hidden_dim = output_dim

        if feature_pre:
            self.linear_pre = nn.Linear(input_dim, feature_dim)
            self.conv_first = PGNNLayer(feature_dim, anchor_dim, hidden_dim, **kwargs)
        else:
            self.conv_first = PGNNLayer(input_dim, anchor_dim, hidden_dim, **kwargs)

        if num_layers > 1:
            self.conv_hidden = nn.ModuleList([PGNNLayer(hidden_dim, anchor_dim, hidden_dim, **kwargs)] * (num_layers - 2))
            self.conv_out = PGNNLayer(hidden_dim, anchor_dim, output_dim, **kwargs)

    def forward(self, G1_data, G2_data):
        x1, x2 = G1_data.x, G2_data.x
        dists_argmax_1, dists_argmax_2 = G1_data.dists_argmax, G2_data.dists_argmax
        dists_max_1, dists_max_2 = G1_data.dists_max, G2_data.dists_max

        if self.feature_pre:
            x1 = self.linear_pre(x1)
            x2 = self.linear_pre(x2)
        x1_position, x1, x2_position, x2 = self.conv_first(x1, x2, dists_max_1, dists_max_2, dists_argmax_1, dists_argmax_2)
        x1, x2 = F.sigmoid(x1), F.sigmoid(x2)

        if self.num_layers == 1:
            x1_position = F.normalize(x1_position, p=2, dim=-1)
            x2_position = F.normalize(x2_position, p=2, dim=-1)
            return x1_position, x2_position

        if self.use_dropout:
            x1 = F.dropout(x1, training=self.training)
            x2 = F.dropout(x2, training=self.training)

        for i in range(self.num_layers-2):
            _, x1, _, x2 = self.conv_hidden[i](x1, x2, dists_max_1, dists_max_2, dists_argmax_1, dists_argmax_2)
            x1, x2 = F.sigmoid(x1), F.sigmoid(x2)
            if self.use_dropout:
                x1 = F.dropout(x1, training=self.training)
                x2 = F.dropout(x2, training=self.training)

        x1_position, x1, x2_position, x2 = self.conv_out(x1, x2, dists_max_1, dists_max_2, dists_argmax_1, dists_argmax_2)
        x1_position = F.normalize(x1_position, p=2, dim=-1)
        x2_position = F.normalize(x2_position, p=2, dim=-1)
        return x1_position, x2_position


class GCN(torch.nn.Module):
    def __init__(self, input_dim, output_dim, num_layers=1, **kwargs):
        super(GCN, self).__init__()
        self.num_layers = num_layers
        self.gcn_in = GCNConv(input_dim, output_dim)
        self.gcn = nn.ModuleList([GCNConv(output_dim, output_dim)] * (num_layers - 1))

    def forward(self, G1_data, G2_data):
        x1, x2 = G1_data.x, G2_data.x
        x1 = self.gcn_in(x1, G1_data.edge_index)
        x2 = self.gcn_in(x2, G2_data.edge_index)
        for i in range(self.num_layers - 1):
            x1 = self.gcn[i](x1, G1_data.edge_index)
            x2 = self.gcn[i](x2, G2_data.edge_index)
        x1 = F.normalize(x1, p=2, dim=-1)
        x2 = F.normalize(x2, p=2, dim=-1)
        return x1, x2


class BRIGHT_U(torch.nn.Module):
    def __init__(self, rwr_dim, dim):
        super(BRIGHT_U, self).__init__()
        self.lin1 = torch.nn.Linear(rwr_dim, dim)

    def forward(self, G1_data, G2_data):
        rwr1_emd, rwr2_emd = G1_data.dists, G2_data.dists
        pos_emd1 = self.lin1(rwr1_emd)
        pos_emd2 = self.lin1(rwr2_emd)
        pos_emd1 = F.normalize(pos_emd1, p=1, dim=1)
        pos_emd2 = F.normalize(pos_emd2, p=1, dim=1)
        return pos_emd1, pos_emd2


class RankingLoss(torch.nn.Module):
    def __init__(self, k, margin, dist_type='l1', device='cpu', **kwargs):
        """
        Marginal Ranking Loss
        :param k: number of negative samples
        :param margin: margin
        :param dist_type: distance metric type
        :param device: device
        """
        super(RankingLoss, self).__init__()
        self.k = k
        self.margin = margin
        self.dist_type = dist_type
        self.device = device

    def neg_sampling(self, out1, out2, anchor1, anchor2):
        """
        Negative sampling
        :param out1: output node embeddings of graph 1
        :param out2: output node embeddings of graph 2
        :param anchor1: anchor nodes of graph 1
        :param anchor2: anchor nodes of graph 2
        :return:
            neg_samples_1: negative samples of graph 1 -> (self.k, num of anchor nodes)
            neg_samples_2: negative samples of graph 2 -> (self.k, num of anchor nodes)
        """

        anchor_embeddings_1 = out1[anchor1]
        anchor_embeddings_2 = out2[anchor2]

        distances_1 = compute_distance_matrix(anchor_embeddings_1, out2, self.dist_type)
        ranks_1 = np.argsort(distances_1, axis=1)
        neg_samples_1 = ranks_1[:, :self.k]

        distances_2 = compute_distance_matrix(anchor_embeddings_2, out1, self.dist_type)
        ranks_2 = np.argsort(distances_2, axis=1)
        neg_samples_2 = ranks_2[:, :self.k]

        return neg_samples_1, neg_samples_2

    def forward(self, out1, out2, anchor1, anchor2):
        np_out1 = out1.detach().cpu().numpy()
        np_out2 = out2.detach().cpu().numpy()
        anchor1 = np.array(anchor1)
        anchor2 = np.array(anchor2)

        neg_samples_1, neg_samples_2 = self.neg_sampling(np_out1, np_out2, anchor1, anchor2)

        anchor_embeddings_1 = out1[anchor1]
        anchor_embeddings_2 = out2[anchor2]
        neg_embeddings_1 = out2[neg_samples_1, :]
        neg_embeddings_2 = out1[neg_samples_2, :]

        A = self.compute_dist(anchor_embeddings_1, anchor_embeddings_2)
        D = A + self.margin
        B1 = -self.compute_dist(anchor_embeddings_1.unsqueeze(1).repeat(1, self.k, 1).view(-1, anchor_embeddings_1.shape[-1]),
                                neg_embeddings_1.view(-1, neg_embeddings_1.shape[-1]))
        L1 = torch.sum(F.relu(D.unsqueeze(-1) + B1.view(-1, self.k)))
        B2 = -self.compute_dist(anchor_embeddings_2.unsqueeze(1).repeat(1, self.k, 1).view(-1, anchor_embeddings_2.shape[-1]),
                                neg_embeddings_2.view(-1, neg_embeddings_2.shape[-1]))
        L2 = torch.sum(F.relu(D.unsqueeze(-1) + B2.view(-1, self.k)))

        return (L1 + L2) / (anchor1.shape[0] * self.k)

    def compute_dist(self, embedding1, embedding2):
        assert self.dist_type in ['l1', 'cosine'], 'Similarity function not supported'

        if self.dist_type == 'l1':
            return torch.sum(torch.abs(embedding1 - embedding2), 1)
        elif self.dist_type == 'cosine':
            return 1 - torch.sum(embedding1 * embedding2, 1) / (torch.norm(embedding1, p=2, dim=1) * torch.norm(embedding2, p=2, dim=1))


class ConsistencyLoss(torch.nn.Module):
    def __init__(self, G1_data, G2_data, device='cpu', lambda_edge=1e-3, lambda_neigh=4e-2, lambda_align=1, margin=10, **kwargs):
        super(ConsistencyLoss, self).__init__()

        self.lambda_edge = lambda_edge
        self.lambda_neigh = lambda_neigh
        self.lambda_align = lambda_align
        self.margin = margin
        self.device = device

        x1, x2 = F.normalize(G1_data.x, p=2, dim=1), F.normalize(G2_data.x, p=2, dim=1)
        self.C1 = (torch.exp(-(x1 @ x1.T)) * G1_data.adj).to(self.device)
        self.C2 = (torch.exp(-(x2 @ x2.T)) * G2_data.adj).to(self.device)
        self.W1 = (self.pinv_diag(G1_data.adj.sum(1)) @ G1_data.adj).T.to(self.device)
        self.W2 = (self.pinv_diag(G2_data.adj.sum(1)) @ G2_data.adj).T.to(self.device)

        assert G1_data.anchor_nodes.shape[0] == G2_data.anchor_nodes.shape[0], \
            'Number of anchor links of G1 and G2 should be the same'
        self.H = torch.zeros(G1_data.x.shape[0], G2_data.x.shape[0]).float()
        for i in range(G1_data.anchor_nodes.shape[0]):
            self.H[G1_data.anchor_nodes[i], G2_data.anchor_nodes[i]] = 1
        self.H /= G1_data.anchor_nodes.shape[0]
        self.H = self.H.to(device)

    def forward(self, out1, out2, **kwargs):
        similarity = 1 - torch.exp(-(out1 @ out2.T))
        edge_loss = self.compute_edge_loss(similarity)
        neigh_loss = self.compute_neighborhood_loss(similarity)
        align_loss = self.compute_alignment_loss(similarity)
        return self.lambda_edge * edge_loss + self.lambda_neigh * neigh_loss + self.lambda_align * align_loss + self.margin

    def compute_edge_loss(self, similarity):
        n1, n2 = similarity.shape
        vec_u = torch.ones(n1, 1).to(self.device) / n1
        vec_v = torch.ones(n2, 1).to(self.device) / n2
        L = (self.C1 ** 2) @ vec_u @ (vec_v * n2).T + (vec_u * n1) @ vec_v.T @ (self.C2 ** 2) - 2 * self.C1 @ similarity @ self.C2.T
        edge_loss = torch.sum(L * similarity) / (n1 * n2)
        return edge_loss

    def compute_neighborhood_loss(self, similarity):
        n1, n2 = similarity.shape
        neigh_similarity = self.W1.T @ similarity @ self.W2
        neigh_loss = torch.sum((neigh_similarity - similarity) ** 2) / (n1 * n2)
        return neigh_loss

    def compute_alignment_loss(self, similarity):
        n1, n2 = similarity.shape
        align_loss = torch.sum((self.H - similarity) ** 2) / (n1 * n2)
        return align_loss

    @staticmethod
    def pinv_diag(vec):
        inv = 1 / vec
        for i in range(len(inv)):
            if inv[i] == torch.inf:
                inv[i] = 0
        return torch.diag(inv)


class RegularizedRankingLoss(RankingLoss):
    def __init__(self, G1_data, G2_data, k, margin, dist_type='l1', lambda_rank=0.5,
                 lambda_edge=1e-3, lambda_neigh=10, lambda_align=2,
                 device='cpu', **kwargs):
        super(RegularizedRankingLoss, self).__init__(k, margin, dist_type, device=device)

        self.lambda_reg = lambda_edge
        self.lambda_neigh = lambda_neigh
        self.lambda_align = lambda_align
        self.lambda_rank = lambda_rank

        x1, x2 = F.normalize(G1_data.x, p=2, dim=1), F.normalize(G2_data.x, p=2, dim=1)
        self.C1 = torch.exp(-(x1 @ x1.T)) * G1_data.adj
        self.C2 = torch.exp(-(x2 @ x2.T)) * G2_data.adj
        self.W1 = (self.pinv_diag(G1_data.adj.sum(1)) @ G1_data.adj).T
        self.W2 = (self.pinv_diag(G2_data.adj.sum(1)) @ G2_data.adj).T

        assert G1_data.anchor_nodes.shape[0] == G2_data.anchor_nodes.shape[0], \
            'Number of anchor links of G1 and G2 should be the same'
        self.H = torch.zeros(G1_data.x.shape[0], G2_data.x.shape[0]).float()
        for i in range(G1_data.anchor_nodes.shape[0]):
            self.H[G1_data.anchor_nodes[i], G2_data.anchor_nodes[i]] = 1

    def forward(self, out1, out2, anchor1, anchor2):
        ranking_loss = super(RegularizedRankingLoss, self).forward(out1, out2, anchor1, anchor2)

        similarity = 1 - torch.exp(-(out1 @ out2.T))
        edge_loss = self.compute_edge_loss(similarity)
        neigh_loss = self.compute_neighborhood_loss(similarity)
        align_loss = self.compute_alignment_loss(similarity)

        return (self.lambda_rank * ranking_loss + (1 - self.lambda_rank) *
                (self.lambda_reg * edge_loss + self.lambda_neigh * neigh_loss + self.lambda_align * align_loss))

    def compute_edge_loss(self, similarity):
        n1, n2 = similarity.shape
        vec_u = torch.ones(n1, 1).to(self.device) / n1
        vec_v = torch.ones(n2, 1).to(self.device) / n2
        C1, C2 = self.C1.to(self.device), self.C2.to(self.device)
        L = (C1 ** 2) @ vec_u @ (vec_v * n2).T + (vec_u * n1) @ vec_v.T @ (C2 ** 2) - 2 * C1 @ similarity @ C2.T
        edge_loss = torch.sum(L * similarity) / (n1 * n2)
        return edge_loss

    def compute_neighborhood_loss(self, similarity):
        n1, n2 = similarity.shape
        W1, W2 = self.W1.to(self.device), self.W2.to(self.device)
        neigh_similarity = W1.T @ similarity @ W2
        neigh_loss = torch.sum((neigh_similarity - similarity) ** 2) / (n1 * n2)
        return neigh_loss

    def compute_alignment_loss(self, similarity):
        n1, n2 = similarity.shape
        H = self.H.to(self.device)
        align_loss = torch.sum((H - similarity) ** 2) / (n1 * n2)
        return align_loss

    @staticmethod
    def pinv_diag(vec):
        inv = 1 / vec
        for i in range(len(inv)):
            if inv[i] == torch.inf:
                inv[i] = 0
        return torch.diag(inv)


class WeightedRankingLoss(torch.nn.Module):
    def __init__(self, G1_data, G2_data, k, margin, dist_type='l1', alpha=0.1, device='cpu', **kwargs):
        """
        Marginal Ranking Loss
        :param G1_data: PyG Data object for graph 1
        :param G2_data: PyG Data object for graph 2
        :param k: number of negative samples
        :param margin: margin
        :param dist_type: distance metric type
        """
        super(WeightedRankingLoss, self).__init__()
        self.k = k
        self.margin = margin
        self.dist_type = dist_type
        self.device = device

        r1, r2 = G1_data.dists, G2_data.dists
        x1 = G1_data.x if G1_data.x.shape[1] > 1 else G1_data.dists
        x2 = G2_data.x if G2_data.x.shape[1] > 1 else G2_data.dists

        r1, r2 = F.normalize(r1, p=2, dim=1), F.normalize(r2, p=2, dim=1)
        x1, x2 = F.normalize(x1, p=2, dim=1), F.normalize(x2, p=2, dim=1)

        ot_cost = alpha * torch.exp(-(r1 @ r2.T)) + (1 - alpha) * torch.exp(-(x1 @ x2.T))
        self.ot_cost = ot_cost.to(self.device)

    def neg_sampling(self, out1, out2, anchor1, anchor2):
        """
        Negative sampling
        :param out1: output node embeddings of graph 1
        :param out2: output node embeddings of graph 2
        :param anchor1: anchor nodes of graph 1
        :param anchor2: anchor nodes of graph 2
        :return:
            neg_samples_1: negative samples of graph 1 -> (self.k, num of anchor nodes)
            neg_samples_2: negative samples of graph 2 -> (self.k, num of anchor nodes)
        """

        anchor_embeddings_1 = out1[anchor1]
        anchor_embeddings_2 = out2[anchor2]

        distances_1 = compute_distance_matrix(anchor_embeddings_1, out2, self.dist_type)
        ranks_1 = np.argsort(distances_1, axis=1)
        neg_samples_1 = ranks_1[:, :self.k]

        distances_2 = compute_distance_matrix(anchor_embeddings_2, out1, self.dist_type)
        ranks_2 = np.argsort(distances_2, axis=1)
        neg_samples_2 = ranks_2[:, :self.k]

        return neg_samples_1, neg_samples_2

    def forward(self, out1, out2, anchor1, anchor2):
        np_out1 = out1.detach().cpu().numpy()
        np_out2 = out2.detach().cpu().numpy()
        anchor1 = np.array(anchor1)
        anchor2 = np.array(anchor2)

        neg_samples_1, neg_samples_2 = self.neg_sampling(np_out1, np_out2, anchor1, anchor2)

        anchor_embeddings_1 = out1[anchor1]
        anchor_embeddings_2 = out2[anchor2]
        neg_embeddings_1 = out2[neg_samples_1, :]
        neg_embeddings_2 = out1[neg_samples_2, :]

        # Weight the negative samples
        idx1 = torch.from_numpy(anchor1).unsqueeze(1).repeat(1, self.k).to(self.device)
        idx2 = torch.from_numpy(anchor2).unsqueeze(1).repeat(1, self.k).to(self.device)
        neg_embeddings_1 *= self.ot_cost[idx1, neg_samples_1].unsqueeze(-1)
        neg_embeddings_2 *= self.ot_cost.T[idx2, neg_samples_2].unsqueeze(-1)

        A = self.compute_dist(anchor_embeddings_1, anchor_embeddings_2)
        D = A + self.margin
        B1 = -self.compute_dist(anchor_embeddings_1.unsqueeze(1).repeat(1, self.k, 1).view(-1, anchor_embeddings_1.shape[-1]),
                                neg_embeddings_1.view(-1, neg_embeddings_1.shape[-1]))
        L1 = torch.sum(F.relu(D.unsqueeze(-1) + B1.view(-1, self.k)))
        B2 = -self.compute_dist(anchor_embeddings_2.unsqueeze(1).repeat(1, self.k, 1).view(-1, anchor_embeddings_2.shape[-1]),
                                neg_embeddings_2.view(-1, neg_embeddings_2.shape[-1]))
        L2 = torch.sum(F.relu(D.unsqueeze(-1) + B2.view(-1, self.k)))

        return (L1 + L2) / (anchor1.shape[0] * self.k)

    def compute_dist(self, embedding1, embedding2):
        assert self.dist_type in ['l1', 'cosine'], 'Similarity function not supported'

        if self.dist_type == 'l1':
            return torch.sum(torch.abs(embedding1 - embedding2), 1)
        elif self.dist_type == 'cosine':
            return 1 - torch.sum(embedding1 * embedding2, 1) / (torch.norm(embedding1, p=2, dim=1) * torch.norm(embedding2, p=2, dim=1))

    @staticmethod
    def pinv_diag(vec):
        inv = 1 / vec
        for i in range(len(inv)):
            if inv[i] == torch.inf:
                inv[i] = 0
        return torch.diag(inv)


class WeightedRegularizedRankingLoss(WeightedRankingLoss):
    def __init__(self, G1_data, G2_data, k, margin, dist_type='l1', lambda_rank=0.5,
                 lambda_edge=1e-3, lambda_neigh=4e-2, lambda_align=1,
                 device='cpu', **kwargs):
        super(WeightedRegularizedRankingLoss, self).__init__(G1_data, G2_data, k, margin, dist_type, device=device)

        self.lambda_reg = lambda_edge
        self.lambda_neigh = lambda_neigh
        self.lambda_align = lambda_align
        self.lambda_rank = lambda_rank

        x1, x2 = F.normalize(G1_data.x, p=2, dim=1), F.normalize(G2_data.x, p=2, dim=1)
        self.C1 = (torch.exp(-(x1 @ x1.T)) * G1_data.adj).to(self.device)
        self.C2 = (torch.exp(-(x2 @ x2.T)) * G2_data.adj).to(self.device)
        self.W1 = (self.pinv_diag(G1_data.adj.sum(1)) @ G1_data.adj).T.to(self.device)
        self.W2 = (self.pinv_diag(G2_data.adj.sum(1)) @ G2_data.adj).T.to(self.device)

        assert G1_data.anchor_nodes.shape[0] == G2_data.anchor_nodes.shape[0], \
            'Number of anchor links of G1 and G2 should be the same'
        self.H = torch.zeros(G1_data.x.shape[0], G2_data.x.shape[0]).float()
        for i in range(G1_data.anchor_nodes.shape[0]):
            self.H[G1_data.anchor_nodes[i], G2_data.anchor_nodes[i]] = 1
        self.H = self.H.to(self.device)

    def forward(self, out1, out2, anchor1, anchor2):
        ranking_loss = super(WeightedRegularizedRankingLoss, self).forward(out1, out2, anchor1, anchor2)

        similarity = 1 - torch.exp(-(out1 @ out2.T))
        edge_loss = self.compute_edge_loss(similarity)
        neigh_loss = self.compute_neighborhood_loss(similarity)
        align_loss = self.compute_alignment_loss(similarity)

        return (self.lambda_rank * ranking_loss + (1 - self.lambda_rank) *
                (self.lambda_reg * edge_loss + self.lambda_neigh * neigh_loss + self.lambda_align * align_loss))

    def compute_edge_loss(self, similarity):
        n1, n2 = similarity.shape
        vec_u = torch.ones(n1, 1).to(self.device) / n1
        vec_v = torch.ones(n2, 1).to(self.device) / n2
        L = (self.C1 ** 2) @ vec_u @ (vec_v * n2).T + (vec_u * n1) @ vec_v.T @ (self.C2 ** 2) - 2 * self.C1 @ similarity @ self.C2.T
        edge_loss = torch.sum(L * similarity) / (n1 * n2)
        return edge_loss

    def compute_neighborhood_loss(self, similarity):
        n1, n2 = similarity.shape
        neigh_similarity = self.W1.T @ similarity @ self.W2
        neigh_loss = torch.sum((neigh_similarity - similarity) ** 2) / (n1 * n2)
        return neigh_loss

    def compute_alignment_loss(self, similarity):
        n1, n2 = similarity.shape
        align_loss = torch.sum((self.H - similarity) ** 2) / (n1 * n2)
        return align_loss

    @staticmethod
    def pinv_diag(vec):
        inv = 1 / vec
        for i in range(len(inv)):
            if inv[i] == torch.inf:
                inv[i] = 0
        return torch.diag(inv)
