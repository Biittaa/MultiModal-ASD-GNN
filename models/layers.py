"""Message-passing layers for the sMRI branch (NaN-safe ChebConv + selectable conv types)."""
import torch
import torch.nn as nn
import torch_geometric as tg
from torch_geometric.utils import add_self_loops, remove_self_loops
from torch_geometric.utils import scatter as pyg_scatter
try:
    from torch_geometric.utils import maybe_num_nodes
except ImportError:
    from torch_geometric.utils.num_nodes import maybe_num_nodes

CONV_TYPES = ("cheb", "gcn", "graph", "gat", "gin", "sage")
LEARNABLE_GRAPH_MODES = ("learnable", "learnable_scratch")
GAT_HEADS = 4
EDGE_DEGREE_EPS = 1e-6
EDGE_SCRATCH_INIT_SCALE = 0.1


def safe_get_laplacian(edge_index, edge_weight=None, normalization=None, dtype=None, num_nodes=None):
    """get_laplacian whose degree normalisation uses |deg| clamped away from 0, so signed /
    learnable edge weights can never give (negative) ** -0.5 = NaN."""
    edge_index, edge_weight = remove_self_loops(edge_index, edge_weight)
    if edge_weight is None:
        edge_weight = torch.ones(edge_index.size(1), dtype=dtype, device=edge_index.device)
    num_nodes = maybe_num_nodes(edge_index, num_nodes)
    row, col = edge_index[0], edge_index[1]
    deg = pyg_scatter(edge_weight, row, 0, dim_size=num_nodes, reduce="sum")
    deg_safe = deg.abs().clamp_min(EDGE_DEGREE_EPS)
    if normalization is None:
        edge_index, _ = add_self_loops(edge_index, num_nodes=num_nodes)
        edge_weight = torch.cat([-edge_weight, deg_safe], dim=0)
    elif normalization == "sym":
        deg_inv_sqrt = deg_safe.pow(-0.5)
        edge_weight = deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col]
        edge_index, edge_weight = add_self_loops(edge_index, -edge_weight, fill_value=1., num_nodes=num_nodes)
    else:  # 'rw'
        edge_weight = (1.0 / deg_safe)[row] * edge_weight
        edge_index, edge_weight = add_self_loops(edge_index, -edge_weight, fill_value=1., num_nodes=num_nodes)
    return edge_index, edge_weight


class SafeChebConv(tg.nn.ChebConv):
    def __norm__(self, edge_index, num_nodes, edge_weight, normalization,
                 lambda_max=None, dtype=None, batch=None):
        edge_index, edge_weight = safe_get_laplacian(edge_index, edge_weight, normalization, dtype, num_nodes)
        if lambda_max is None:
            lambda_max = 2.0 * edge_weight.max()
        elif not isinstance(lambda_max, torch.Tensor):
            lambda_max = torch.tensor(lambda_max, dtype=dtype, device=edge_index.device)
        if batch is not None and lambda_max.numel() > 1:
            lambda_max = lambda_max[batch[edge_index[0]]]
        edge_weight = (2.0 * edge_weight) / lambda_max
        edge_weight.masked_fill_(edge_weight == float("inf"), 0)
        loop_mask = edge_index[0] == edge_index[1]
        edge_weight[loop_mask] -= 1
        return edge_index, edge_weight


def build_view_conv(conv_type, in_c, hid_c, K):
    if conv_type == "cheb":
        return SafeChebConv(in_c, hid_c, K, normalization="sym", bias=True)
    if conv_type == "gcn":
        return tg.nn.GCNConv(in_c, hid_c, add_self_loops=True, normalize=True, bias=True)
    if conv_type == "graph":
        return tg.nn.GraphConv(in_c, hid_c, aggr="add", bias=True)
    if conv_type == "gat":
        return tg.nn.GATConv(in_c, hid_c, heads=GAT_HEADS, concat=False, edge_dim=1, dropout=0.0, bias=True)
    if conv_type == "gin":
        mlp = nn.Sequential(nn.Linear(in_c, hid_c), nn.ReLU(), nn.Linear(hid_c, hid_c))
        return tg.nn.GINEConv(mlp, edge_dim=1)
    if conv_type == "sage":
        return tg.nn.SAGEConv(in_c, hid_c, aggr="mean", bias=True)
    raise ValueError(f"conv_type must be one of {CONV_TYPES}, got {conv_type!r}")


def run_view_conv(conv, conv_type, x, edge_index, edge_weight):
    if conv_type in ("cheb", "gcn", "graph"):
        return conv(x, edge_index, edge_weight)
    if conv_type in ("gat", "gin"):
        return conv(x, edge_index, edge_attr=edge_weight.unsqueeze(-1))
    if conv_type == "sage":
        return conv(x, edge_index)
    raise ValueError(f"conv_type must be one of {CONV_TYPES}, got {conv_type!r}")
