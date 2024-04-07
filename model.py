import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import scipy


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

    def forward(self, x1, x2, dists_max_1, dists_max_2, dists_argmax_1, dists_argmax_2):
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
        # x1, x2 = F.relu(x1), F.relu(x2)  # Note: optional!
        if self.num_layers == 1:
            x1_position = F.normalize(x1_position, p=1, dim=-1)
            x2_position = F.normalize(x2_position, p=1, dim=-1)
            return x1_position, x2_position

        if self.use_dropout:
            x1 = F.dropout(x1, training=self.training)
            x2 = F.dropout(x2, training=self.training)

        for i in range(self.num_layers-2):
            _, x1, _, x2 = self.conv_hidden[i](x1, x2, dists_max_1, dists_max_2, dists_argmax_1, dists_argmax_2)
            # x1, x2 = F.relu(x1), F.relu(x2)  # Note: optional!
            if self.use_dropout:
                x1 = F.dropout(x1, training=self.training)
                x2 = F.dropout(x2, training=self.training)

        x1_position, x1, x2_position, x2 = self.conv_out(x1, x2, dists_max_1, dists_max_2, dists_argmax_1, dists_argmax_2)
        x1_position = F.normalize(x1_position, p=1, dim=-1)
        x2_position = F.normalize(x2_position, p=1, dim=-1)
        return x1_position, x2_position


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


class RankingLossL1(torch.nn.Module):
    def __init__(self, k, margin):
        """
        Marginal Ranking Loss with L1 distance
        :param k: number of negative samples
        :param margin: margin
        """
        super().__init__()
        self.k = k
        self.margin = margin

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
        D = A + self.margin
        B1 = -torch.sum(torch.abs(anchor_embeddings_1.unsqueeze(1) - neg_embeddings_1), 2)
        L1 = torch.sum(F.relu(D.unsqueeze(-1) + B1))
        B2 = -torch.sum(torch.abs(anchor_embeddings_2.unsqueeze(1) - neg_embeddings_2), 2)
        L2 = torch.sum(F.relu(D.unsqueeze(-1) + B2))

        return (L1 + L2) / (anchor1.shape[0] * self.k)
