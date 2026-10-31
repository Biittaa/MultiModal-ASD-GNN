"""sMRI + fMRI fusion.
sMRI tokens : one per view (aseg / aparc / wmparc)                        = 3
fMRI tokens : DHGFormer ROI features averaged per functional sub-network  = 8
FUSION='concat'          : each modality -> one vector, concatenate -> classifier
FUSION='cross_attention' : sMRI attends to fMRI AND fMRI attends to sMRI (1 small layer), mean-pool, concat
Extras against overfitting: tiny shared dim, dropout, modality dropout, per-modality auxiliary heads."""
import torch
import torch.nn as nn

from .fmri import SUBNET_ENDS, DHGFormer
from .smri import SMRIEncoder


class FusionNet(nn.Module):
    def __init__(self, in_channels, cfg, base_edge_weights, cluster_order, fmri_len):
        super().__init__()
        d, p = cfg.fusion_dim, cfg.fusion_dropout
        self.fusion, self.mod_drop = cfg.fusion, cfg.modality_dropout
        self.smri = SMRIEncoder(in_channels, cfg.hidden_dim, cfg.cheb_K, cfg.dropout, cfg.conv_type,
                                cfg.graph_mode, base_edge_weights)
        self.dhg = DHGFormer(cfg.embedding_size, 200, fmri_len, cluster_order)
        n_f, n_s = len(SUBNET_ENDS), len(in_channels)
        self.s_norm, self.s_proj = nn.LayerNorm(cfg.hidden_dim), nn.Linear(cfg.hidden_dim, d)
        self.f_norm, self.f_proj = nn.LayerNorm(8), nn.Linear(8, d)
        self.s_type = nn.Parameter(torch.randn(1, 1, d) * 0.02)       # modality embeddings
        self.f_type = nn.Parameter(torch.randn(1, 1, d) * 0.02)
        if cfg.fusion == "concat":
            self.s_flat = nn.Sequential(nn.Flatten(), nn.Linear(n_s * d, d), nn.GELU(), nn.Dropout(p))
            self.f_flat = nn.Sequential(nn.Flatten(), nn.Linear(n_f * d, d), nn.GELU(), nn.Dropout(p))
        elif cfg.fusion == "cross_attention":
            self.attn_sf = nn.MultiheadAttention(d, cfg.fusion_heads, dropout=p, batch_first=True)
            self.attn_fs = nn.MultiheadAttention(d, cfg.fusion_heads, dropout=p, batch_first=True)
            self.ln_s, self.ln_f = nn.LayerNorm(d), nn.LayerNorm(d)
        else:
            raise ValueError("fusion must be 'concat' or 'cross_attention'")
        self.head = nn.Sequential(nn.Dropout(p), nn.Linear(2 * d, 2))
        self.aux_s, self.aux_f = nn.Linear(d, 2), nn.Linear(d, 2)

    def forward(self, batch):
        s = self.smri(batch["xs"], batch["bank"], batch["m"])            # (B, n_views, hid)
        fnode, adj, _ = self.dhg.encode(batch["ts"], batch["corr"])       # (B, 200, 8)
        f = self.dhg.predictor._avg_sub(fnode)                            # (B, 8, 8)
        s, f = self.s_proj(self.s_norm(s)), self.f_proj(self.f_norm(f))
        if self.training and self.mod_drop > 0:       # hide one whole modality for a random subset of subjects
            u = torch.rand(s.shape[0], device=s.device)
            keep_s = (~((u >= self.mod_drop) & (u < 2 * self.mod_drop))).float()[:, None, None]
            keep_f = (~(u < self.mod_drop)).float()[:, None, None]
            s, f = s * keep_s, f * keep_f
        s, f = s + self.s_type, f + self.f_type
        if self.fusion == "concat":
            zs, zf = self.s_flat(s), self.f_flat(f)
        else:
            s2 = self.ln_s(s + self.attn_sf(s, f, f, need_weights=False)[0])
            f2 = self.ln_f(f + self.attn_fs(f, s, s, need_weights=False)[0])
            zs, zf = s2.mean(1), f2.mean(1)
        logits = self.head(torch.cat([zs, zf], dim=1))
        return dict(logits=logits, aux=[self.aux_s(zs), self.aux_f(zf)], adj=adj)

    def post_step(self):
        self.smri.clamp_edges()
