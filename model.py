import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import scipy


class PGNNLayer(torch.nn.Module):
    def __init__(self, input_dim, output_dim, dist_trainable=False, mcf_type='default', agg_type='mean', use_hidden=False):
        """
        One PGNN Layer
        :param input_dim: input feature dimension
        :param output_dim: output feature dimension
        :param dist_trainable: whether to use trainable distance metric scores
        :param mcf_type: type of message computation function (e.g. default, concat, mean, etc.)
        :param agg_type: type of message aggregation function (e.g. mean, sum, max, etc.)
        :param use_hidden: whether to use SLP after message computation function F
        """
        super(PGNNLayer, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = input_dim if mcf_type is not 'concat' else input_dim * 2
        self.output_dim = output_dim
        self.dist_trainable = dist_trainable
        self.mcf_type = mcf_type
        self.agg_type = agg_type
        self.use_hidden = use_hidden

        if self.dist_trainable:
            self.dist_compute = Nonlinear(1, output_dim, 1)

        self.linear_hidden = nn.Linear(self.hidden_dim, self.output_dim) if self.use_hidden else None
        self.linear_out_position = nn.Linear(self.output_dim, 1)
        self.act = nn.ReLU()

    def forward(self, G1_data, G2_data):
        x1, x2 = G1_data.x, G2_data.x
        dists_argmax_1, dists_argmax_2 = G1_data.dists_argmax, G2_data.dists_argmax
        dists_max_1, dists_max_2 = G1_data.dists_max, G2_data.dists_max

        anchor_features_1 = x1[dists_argmax_1, :]
        anchor_features_2 = x2[dists_argmax_2, :]
        self_features_1 = x1.unsqueeze(1).repeat(1, dists_max_1.shape[1], 1)
        self_features_2 = x2.unsqueeze(1).repeat(1, dists_max_2.shape[1], 1)

        messages_1 = self.mcf(self_features_1, anchor_features_1, dists_max_1)
        messages_2 = self.mcf(self_features_2, anchor_features_2, dists_max_2)

        if self.use_hidden:
            assert self.linear_hidden is not None, 'Hidden layer is not defined'
            messages_1 = self.linear_hidden(messages_1).squeeze()
            messages_1 = self.act(messages_1)
            messages_2 = self.linear_hidden(messages_2).squeeze()
            messages_2 = self.act(messages_2)

        out_position_1 = self.linear_out_position(messages_1).squeeze(-1)  # zv (output)
        out_structure_1 = self.agg(messages_1)  # hv (feed to the next layer)
        out_position_2 = self.linear_out_position(messages_2).squeeze(-1)
        out_structure_2 = self.agg(messages_2)

        return out_position_1, out_structure_1, out_position_2, out_structure_2

    def mcf(self, node_feat, anchor_feat, distances):
        """
        Message Computation Function F
        :param node_feat: node features (hv)
        :param anchor_feat: anchorset features (hu)
        :param distances: distances metric scores (s(v, u))
        :return:
            messages: messages F(v, u, hv, hu)
        """
        if self.mcf_type == 'default':
            return distances.unsqueeze(-1) * anchor_feat
        elif self.mcf_type == 'concat':
            return distances.unsqueeze(-1) * torch.cat((node_feat, anchor_feat), dim=-1)
        # TODO: Add more types of MCF here (e.g., mean, min, max, sum, etc.)

    def agg(self, messages):
        """
        Message Aggregation Function AGG
        :param messages: message matrix Mv
        :return:
            out: aggregated messages
        """
        if self.agg_type == 'mean':
            return torch.mean(messages, dim=1)
        # TODO: Add more types of aggregation functions here (e.g., sum, max, etc.)


class Nonlinear(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(Nonlinear, self).__init__()

        self.linear1 = nn.Linear(input_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, output_dim)

        self.act = nn.ReLU()

        for m in self.modules():
            if isinstance(m, nn.Linear):
                m.weight.data = nn.init.xavier_uniform_(m.weight.data, gain=nn.init.calculate_gain('relu'))
                if m.bias is not None:
                    m.bias.data = nn.init.constant_(m.bias.data, 0.0)

    def forward(self, x):
        x = self.linear1(x)
        x = self.act(x)
        x = self.linear2(x)
        return x


class RankingLossL1(torch.nn.Module):
    def __init__(self, k, gamma):
        """
        Marginal Ranking Loss with L1 distance
        :param k: number of negative samples
        :param gamma: margin
        """
        super().__init__()
        self.k = k
        self.gamma = gamma

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

        distances_1 = scipy.spatial.distance.cdist(anchor_embeddings_1, out2, metric='cityblock')
        ranks_1 = np.argsort(distances_1, axis=1)
        neg_samples_1 = ranks_1[:, :self.k]

        distances_2 = scipy.spatial.distance.cdist(anchor_embeddings_2, out1, metric='cityblock')
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

        A = torch.sum(torch.abs(anchor_embeddings_1 - anchor_embeddings_2), 1)
        D = A + self.gamma
        B1 = -torch.sum(torch.abs(anchor_embeddings_1.unsqueeze(1) - neg_embeddings_1), 2)
        L1 = torch.sum(F.relu(D.unsqueeze(-1) + B1))
        B2 = -torch.sum(torch.abs(anchor_embeddings_2.unsqueeze(1) - neg_embeddings_2), 2)
        L2 = torch.sum(F.relu(D.unsqueeze(-1) + B2))

        return (L1 + L2) / (anchor1.shape[0] * self.k)
