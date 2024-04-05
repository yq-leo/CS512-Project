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
        messages_2 = self.mcf(self_features_2, anchor_features_2, dists_max_1)

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
        super().__init__()
        self.k = k
        self.gamma = gamma

    def get_neg(self, out1, out2, anchor1, anchor2):
        neg1 = []
        neg2 = []
        t = len(anchor1)
        anchor1_vec = np.array(out1[anchor1])
        anchor2_vec = np.array(out2[anchor2])
        G1_vec = np.array(out1)
        G2_vec = np.array(out2)
        sim1 = scipy.spatial.distance.cdist(anchor1_vec, G2_vec, metric='cityblock')
        for i in range(t):
            rank = sim1[i, :].argsort()
            neg1.append(rank[0: self.k])
        neg1 = np.array(neg1)
        neg1 = neg1.reshape((t * self.k,))
        sim2 = scipy.spatial.distance.cdist(anchor2_vec, G1_vec, metric='cityblock')
        for i in range(t):
            rank = sim2[i, :].argsort()
            neg2.append(rank[0:self.k])
        anchor1 = np.repeat(anchor1, self.k)
        anchor2 = np.repeat(anchor2, self.k)
        neg2 = np.array(neg2)
        neg2 = neg2.reshape((t * self.k,))
        return anchor1, anchor2, neg1, neg2

    def forward(self, out1, out2, anchor1, anchor2):
        np_out1 = out1.detach().cpu().numpy()
        np_out2 = out2.detach().cpu().numpy()
        anchor1, anchor2, neg1, neg2 = self.get_neg(np_out1, np_out2, anchor1, anchor2)

        anchor1_vec = out1[anchor1]
        anchor2_vec = out2[anchor2]
        neg1_vec = out2[neg1]
        neg2_vec = out1[neg2]

        A = torch.sum(torch.abs(anchor1_vec - anchor2_vec), 1)
        D = A + self.gamma
        B1 = -torch.sum(torch.abs(anchor1_vec-neg1_vec), 1)
        L1 = torch.sum(F.relu(B1 + D))
        B2 = -torch.sum(torch.abs(anchor2_vec - neg2_vec), 1)
        L2 = torch.sum(F.relu(B2 + D))

        return (L1 + L2) / len(anchor1)
