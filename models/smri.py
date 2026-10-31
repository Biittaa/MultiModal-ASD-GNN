"""sMRI branch: one brain-region graph per view (aseg / aparc / wmparc), mini-batch capable."""
import numpy as np
import torch
import torch.nn as nn
import torch_geometric as tg

from .layers import (CONV_TYPES, EDGE_SCRATCH_INIT_SCALE, LEARNABLE_GRAPH_MODES,
                     build_view_conv, run_view_conv)


class GraphBank:
    """Holds the single-copy ROI graph of every view and builds (and caches) the block-diagonal
    batched version for a mini-batch of m subjects: m copies of the graph, node ids offset by
    subject, so one conv call processes the whole batch."""

    def __init__(self, edge_index, edge_weight, n_nodes, device):
        self.ei, self.ew, self.n_nodes, self.device = edge_index, edge_weight, n_nodes, device
        self._cache = {}

    def get(self, m):
        if m not in self._cache:
            out = []
            for ei, ew, n in zip(self.ei, self.ew, self.n_nodes):
                offs = (np.arange(m) * n).reshape(-1, 1, 1)
                ei_b = (ei[None] + offs).transpose(1, 0, 2).reshape(2, -1).astype(np.int64)
                ew_b = np.tile(ew, m).astype(np.float32)
                bv = np.repeat(np.arange(m), n).astype(np.int64)
                out.append(tuple(torch.from_numpy(a).to(self.device) for a in (ei_b, ew_b, bv)))
            self._cache[m] = out
        return self._cache[m]


class SMRIEncoder(nn.Module):
    """x per view (m, n_nodes, F) -> one conv -> ReLU -> dropout -> mean-pool -> (m, n_views, hid)."""

    def __init__(self, in_channels, hid, K, dropout, conv_type="cheb",
                 graph_mode="static", base_edge_weights=None):
        super().__init__()
        if conv_type not in CONV_TYPES:
            raise ValueError(f"conv_type must be one of {CONV_TYPES}")
        self.conv_type, self.graph_mode = conv_type, graph_mode
        self.n_views = len(in_channels)
        self.convs = nn.ModuleList([build_view_conv(conv_type, c, hid, K) for c in in_channels])
        self.act = nn.ReLU()
        self.drop = nn.Dropout(dropout)
        if graph_mode in LEARNABLE_GRAPH_MODES:
            if base_edge_weights is None:
                raise ValueError("learnable graph modes need base_edge_weights")
            if graph_mode == "learnable":        # init from the partial-correlation prior
                self.edge_w = nn.ParameterList(
                    [nn.Parameter(torch.as_tensor(w, dtype=torch.float32).clone()) for w in base_edge_weights])
            else:                                # topology only; weights from scratch
                self.edge_w = nn.ParameterList(
                    [nn.Parameter(torch.empty(len(w)).uniform_(-EDGE_SCRATCH_INIT_SCALE, EDGE_SCRATCH_INIT_SCALE))
                     for w in base_edge_weights])
        elif graph_mode != "static":
            raise ValueError("graph_mode must be 'static', 'learnable' or 'learnable_scratch'")

    def forward(self, xs, bank, m):
        graphs = bank.get(m)
        embs = []
        for v, x in enumerate(xs):
            ei, ew, bv = graphs[v]
            if self.graph_mode in LEARNABLE_GRAPH_MODES:
                ew = self.edge_w[v].repeat(m)
            h = run_view_conv(self.convs[v], self.conv_type, x.reshape(-1, x.shape[-1]), ei, ew)
            h = self.drop(self.act(h))
            embs.append(tg.nn.global_mean_pool(h, bv, size=m))
        return torch.stack(embs, dim=1)

    def clamp_edges(self):
        if self.graph_mode in LEARNABLE_GRAPH_MODES:
            with torch.no_grad():
                for p in self.edge_w:
                    p.clamp_(-1.0, 1.0)


class SMRIClassifier(nn.Module):
    """sMRI-only model: concat of the view embeddings -> dropout -> linear."""

    def __init__(self, in_channels, cfg, base_edge_weights=None):
        super().__init__()
        self.enc = SMRIEncoder(in_channels, cfg.hidden_dim, cfg.cheb_K, cfg.dropout, cfg.conv_type,
                               cfg.graph_mode, base_edge_weights)
        self.drop = nn.Dropout(cfg.dropout)
        self.cls = nn.Linear(cfg.hidden_dim * len(in_channels), 2)

    def forward(self, batch):
        z = self.enc(batch["xs"], batch["bank"], batch["m"])
        return dict(logits=self.cls(self.drop(z.flatten(1))), aux=[], adj=None)

    def post_step(self):
        self.enc.clamp_edges()
