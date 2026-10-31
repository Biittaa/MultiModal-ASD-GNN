"""fMRI branch: DHGFormer (Transformer -> dynamic graph -> hierarchical GCN), cc200 ROI time-series.
Ported from imoonlab/dhgformer (graph_generation='product', transformer extractor); no hard-coded .cuda()."""
import os
import pickle

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# ROI -> sub-network id, copied from imoonlab/dhgformer node_clus_map.pickle (order = DHGFormer ROI re-ordering)
NODE_CLUS_MAP_EMBEDDED = {3: 0, 8: 0, 9: 0, 14: 0, 17: 0, 29: 0, 35: 0, 36: 0, 40: 0, 44: 0, 46: 0, 47: 0, 51: 0, 66: 0, 67: 0, 76: 0, 79: 0, 83: 0, 85: 0, 91: 0, 93: 0, 102: 0, 109: 0, 111: 0, 117: 0, 119: 0, 125: 0, 129: 0, 134: 0, 144: 0, 147: 0, 151: 0, 154: 0, 159: 0, 161: 0, 175: 0, 177: 0, 189: 0, 191: 0, 193: 0, 198: 0, 0: 1, 2: 1, 25: 1, 43: 1, 53: 1, 61: 1, 69: 1, 80: 1, 84: 1, 88: 1, 96: 1, 101: 1, 104: 1, 107: 1, 121: 1, 130: 1, 137: 1, 141: 1, 146: 1, 149: 1, 157: 1, 158: 1, 169: 1, 171: 1, 174: 1, 176: 1, 178: 1, 188: 1, 194: 1, 7: 2, 20: 2, 23: 2, 27: 2, 34: 2, 59: 2, 64: 2, 65: 2, 87: 2, 89: 2, 95: 2, 97: 2, 110: 2, 120: 2, 122: 2, 133: 2, 145: 2, 153: 2, 156: 2, 184: 2, 199: 2, 15: 3, 16: 3, 30: 3, 49: 3, 62: 3, 63: 3, 72: 3, 92: 3, 99: 3, 113: 3, 115: 3, 131: 3, 135: 3, 142: 3, 155: 3, 162: 3, 170: 3, 187: 3, 196: 3, 5: 4, 12: 4, 19: 4, 32: 4, 54: 4, 58: 4, 68: 4, 75: 4, 78: 4, 82: 4, 118: 4, 124: 4, 127: 4, 136: 4, 160: 4, 164: 4, 167: 4, 179: 4, 181: 4, 183: 4, 26: 5, 31: 5, 42: 5, 56: 5, 77: 5, 86: 5, 197: 5, 6: 6, 11: 6, 22: 6, 24: 6, 33: 6, 37: 6, 38: 6, 39: 6, 41: 6, 74: 6, 94: 6, 98: 6, 112: 6, 114: 6, 123: 6, 126: 6, 148: 6, 150: 6, 163: 6, 168: 6, 182: 6, 1: 7, 4: 7, 10: 7, 13: 7, 18: 7, 21: 7, 28: 7, 45: 7, 48: 7, 50: 7, 52: 7, 55: 7, 57: 7, 60: 7, 70: 7, 71: 7, 73: 7, 81: 7, 90: 7, 100: 7, 103: 7, 105: 7, 106: 7, 108: 7, 116: 7, 128: 7, 132: 7, 138: 7, 139: 7, 140: 7, 143: 7, 152: 7, 165: 7, 166: 7, 172: 7, 173: 7, 180: 7, 185: 7, 186: 7, 190: 7, 192: 7, 195: 7}

SUBNET_ENDS = [41, 70, 91, 110, 130, 137, 158, 200]   # cc200 functional sub-networks after re-ordering


def load_cluster_order(path=""):
    if path and os.path.isfile(path):
        with open(path, "rb") as f:
            clus_map = pickle.load(f)
    else:
        clus_map = NODE_CLUS_MAP_EMBEDDED
    order = list(clus_map.keys())
    assert sorted(order) == list(range(200)), "node_clus_map must be a permutation of the 200 cc200 ROIs"
    sizes = [list(clus_map.values()).count(c) for c in sorted(set(clus_map.values()))]
    assert np.cumsum(sizes).tolist() == SUBNET_ENDS, f"sub-network sizes {sizes} do not match SUBNET_ENDS"
    return order


class _FCOut(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.fc = nn.Sequential(nn.Linear(embed_dim, 32), nn.LeakyReLU(0.2), nn.Dropout(0.1),
                                nn.Linear(32, embed_dim), nn.LeakyReLU(0.2), nn.Dropout(0.1))
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        return self.fc(self.norm(x))


def _fc_attention(Q, K, V, mask=None):
    l, num_head = Q.shape[2], Q.shape[1]
    score = torch.matmul(Q, K.permute(0, 1, 3, 2)) / (Q.shape[-1] ** 0.5)
    if mask is not None:                    # FC-inspired dynamic attention: |Pearson FC| rescales scores
        mask = torch.abs(mask).unsqueeze(1).expand(-1, num_head, -1, -1)
        score = score * mask
    score = torch.softmax(score, dim=-1)
    x = torch.matmul(score, V)
    return x.permute(0, 2, 1, 3).reshape(-1, l, num_head * Q.shape[3])


class _MultiHead(nn.Module):
    def __init__(self, input_dim, num_head, embed_dim):
        super().__init__()
        self.fc_Q, self.fc_K, self.fc_V = nn.Linear(input_dim, 32), nn.Linear(input_dim, 32), nn.Linear(input_dim, 32)
        self.num_head = num_head
        self.out_fc = nn.Linear(32, embed_dim)
        self.norm = nn.LayerNorm(input_dim)
        self.dropout = nn.Dropout(0.1)

    def forward(self, Q, K, V, mask=None):
        b, n = Q.shape[0], Q.shape[1]
        Q, K, V = self.norm(Q), self.norm(K), self.norm(V)
        Q, K, V = self.fc_Q(Q), self.fc_K(K), self.fc_V(V)
        Q = Q.reshape(b, n, self.num_head, -1).permute(0, 2, 1, 3)
        K = K.reshape(b, n, self.num_head, -1).permute(0, 2, 1, 3)
        V = V.reshape(b, n, self.num_head, -1).permute(0, 2, 1, 3)
        return self.dropout(self.out_fc(_fc_attention(Q, K, V, mask)))


class FCEncoder(nn.Module):
    def __init__(self, input_dim, num_head, embed_dim):
        super().__init__()
        self.mh = _MultiHead(input_dim, num_head, embed_dim)
        self.fc = _FCOut(embed_dim)

    def forward(self, x, mask=None):
        return self.fc(self.mh(x, x, x, mask))


def _subnetwork_matrix(adj, ends):
    starts = [0] + ends[:-1]
    m = torch.zeros((adj.shape[0], len(ends), len(ends)), device=adj.device)
    for i in range(len(ends)):
        for j in range(i, len(ends)):
            v = adj[:, starts[i]:ends[i], starts[j]:ends[j]].mean(dim=(1, 2))
            m[:, i, j] = v
            m[:, j, i] = v
    return m


class CrossGCNPredictor(nn.Module):
    """Hierarchical GCN: intra-subnetwork + inter-subnetwork propagation, 3 layers."""

    def __init__(self, node_input_dim, roi_num=200):
        super().__init__()
        self.roi_num, self.ends = roi_num, SUBNET_ENDS
        self.gcn = nn.Sequential(nn.Linear(node_input_dim, roi_num), nn.LeakyReLU(0.2), nn.Linear(roi_num, roi_num))
        self.bn1 = nn.BatchNorm1d(roi_num)
        self.gcn1 = nn.Sequential(nn.Linear(roi_num, roi_num), nn.LeakyReLU(0.2))
        self.bn2 = nn.BatchNorm1d(roi_num)
        self.gcn2 = nn.Sequential(nn.Linear(roi_num, 64), nn.LeakyReLU(0.2), nn.Linear(64, 8), nn.LeakyReLU(0.2))
        self.bn3 = nn.BatchNorm1d(roi_num)
        self.classifier = nn.Sequential(nn.Linear(8 * roi_num, 256), nn.LeakyReLU(0.2),
                                        nn.Linear(256, 32), nn.LeakyReLU(0.2), nn.Linear(32, 2))

    def _avg_sub(self, f):
        starts = [0] + self.ends[:-1]
        return torch.stack([f[:, s:e, :].mean(dim=1) for s, e in zip(starts, self.ends)], dim=1)

    def _propagate(self, sub_f, node_f):
        starts = [0] + self.ends[:-1]
        out = torch.zeros_like(node_f)
        for i, (s, e) in enumerate(zip(starts, self.ends)):
            out[:, s:e, :] = sub_f[:, i, :].unsqueeze(1).expand(-1, e - s, -1)
        return (node_f + out) / 2

    def _layer(self, x, intra, inter):
        intra_f = torch.einsum("ijk,ijp->ijp", intra, x)
        sub_f = torch.einsum("ijk,ijp->ijp", inter, self._avg_sub(x))
        return self._propagate(sub_f, intra_f)

    def _bn(self, bn, x):
        b = x.shape[0]
        return bn(x.reshape(b * self.roi_num, -1)).reshape(b, self.roi_num, -1)

    def features(self, intra, inter, node_features):
        x = self._bn(self.bn1, self.gcn(self._layer(node_features, intra, inter)))
        x = self._bn(self.bn2, self.gcn1(self._layer(x, intra, inter)))
        return self.bn3(self.gcn2(self._layer(x, intra, inter)))      # (B, 200, 8)


class DHGFormer(nn.Module):
    def __init__(self, embedding_size, roi_num, time_series_len, cluster_order):
        super().__init__()
        assert roi_num == 200, "DHGFormer sub-network boundaries are hard-coded for cc200"
        self.feature_extractor = FCEncoder(time_series_len, 4, embedding_size)
        self.predictor = CrossGCNPredictor(roi_num, roi_num=roi_num)
        self.cluster_order = list(cluster_order)
        self.ends = SUBNET_ENDS

    def encode(self, time_series, node_features):
        """Everything up to the classifier -> ROI features (B,200,8), learnable adjacency, edge variance."""
        time_series = time_series[:, self.cluster_order, :]
        node_features = node_features[:, self.cluster_order, :][:, :, self.cluster_order]
        emb = F.softmax(self.feature_extractor(time_series, node_features), dim=-1)
        adj = torch.einsum("ijk,ipk->ijp", emb, emb)                    # learnable (dynamic) FC
        n = adj.shape[1]
        intra_mask = torch.zeros((n, n), dtype=torch.bool, device=adj.device)
        s = 0
        for e in self.ends:
            intra_mask[s:e, s:e] = True
            s = e
        intra = adj * intra_mask.unsqueeze(0)
        inter = _subnetwork_matrix(adj, self.ends)
        edge_var = torch.mean(torch.var(adj.reshape(adj.shape[0], -1), dim=1))
        return self.predictor.features(intra, inter, node_features), adj, edge_var

    def forward(self, time_series, node_features):
        x, adj, edge_var = self.encode(time_series, node_features)
        return self.predictor.classifier(x.reshape(x.shape[0], -1)), adj, edge_var


class FMRIModel(nn.Module):
    """fMRI-only model with the common interface used by the trainer."""

    def __init__(self, cfg, fmri_len, cluster_order):
        super().__init__()
        self.dhg = DHGFormer(cfg.embedding_size, 200, fmri_len, cluster_order)

    def forward(self, batch):
        logits, adj, _ = self.dhg(batch["ts"], batch["corr"])
        return dict(logits=logits, aux=[], adj=adj)
