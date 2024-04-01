import torch
import torch.nn.functional as F
import numpy as np
import scipy


class PGNNLayer(torch.nn.Module):
    def __init__(self, in_dim, out_dim):
        super(PGNNLayer, self).__init__()
        self.lin = torch.nn.Linear(in_dim, out_dim)

    def forward(self, x):
        return self.lin(x)


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
