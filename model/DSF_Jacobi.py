# developed based on https://github.com/GraphPKU/JacobiConv
from functools import partial
from torch_geometric.nn.conv.gcn_conv import gcn_norm
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import degree
from torch_sparse import SparseTensor
import torch.nn.functional as F
from torch import Tensor
import torch.nn as nn
import torch


def DSF_Jacobi_R(alpha, dpb, dpt, in_dim, out_dim, conv_layer=10, sole=False,
                 PE_dropout=0, PE_in_dim=None, PE_hid_dim=None, PE_alpha=None, **kwargs):
    emb = nn.Sequential(nn.Dropout(p=dpb), nn.Linear(in_dim, out_dim), nn.Dropout(dpt, inplace=True))
    frame_fn = PolyConvFrame
    conv_fn = partial(JacobiConv, **kwargs)
    conv = frame_fn(conv_fn, depth=conv_layer, alpha=alpha)
    comb = Combination(out_dim, conv_layer + 1, sole=sole)
    pe_agg = PE_aggregation(conv_layer + 1, PE_dropout, PE_in_dim, PE_hid_dim, PE_alpha)
    gnn = Gmodel(emb, conv, comb, pe_agg=pe_agg)
    return gnn


def DSF_Jacobi_I(alpha, dpb, dpt, in_dim, out_dim, conv_layer=10, sole=False,
                 PE_dropout=0, PE_in_dim=None, PE_hid_dim=None, PE_alpha=None, PE_beta=None, **kwargs):
    emb = nn.Sequential(nn.Dropout(p=dpb), nn.Linear(in_dim, out_dim), nn.Dropout(dpt, inplace=True))
    conv_fn = partial(JacobiConv, **kwargs)
    conv = PolyConvFrame(conv_fn, depth=conv_layer, alpha=alpha)
    comb = Combination(out_dim, conv_layer + 1, sole=sole)
    pe_agg = Orth_PE_aggregation(conv_layer + 1, PE_dropout, PE_in_dim, PE_hid_dim, PE_alpha, PE_beta)
    gnn = Gmodel(emb, conv, comb, pe_agg=pe_agg)
    return gnn


class PolyConvFrame(nn.Module):
    '''
    A framework for polynomial graph signal filter.
    Args:
        conv_fn: the filter function, like PowerConv, LegendreConv,...
        depth (int): the order of polynomial.
        cached (bool): whether or not to cache the adjacency matrix.
        alpha (float):  the parameter to initialize polynomial coefficients.
    '''

    def __init__(self,
                 conv_fn,
                 depth: int = 10,
                 aggr: int = "gcn",
                 cached: bool = True,
                 alpha: float = 1.0
                 ):
        super().__init__()
        self.depth = depth
        self.basealpha = alpha
        self.cached = cached
        self.aggr = aggr
        self.adj = None
        self.conv_fn = conv_fn

    def forward(self, x: Tensor, edge_index: Tensor, edge_attr: Tensor, pe_alphas=None):
        self.alphas = pe_alphas
        if self.adj is None or not self.cached:
            n_node = x.shape[0]
            self.adj = buildAdj(edge_index, edge_attr, n_node, self.aggr)
        alphas = [self.basealpha * torch.tanh(_) for _ in self.alphas]
        xs = [self.conv_fn(0, [x], self.adj, alphas)]
        for L in range(1, self.depth + 1):
            tx = self.conv_fn(L, xs, self.adj, alphas)
            xs.append(tx)
        xs = [x.unsqueeze(1) for x in xs]
        x = torch.cat(xs, dim=1)
        return x


def JacobiConv(L, xs, adj, alphas, a=1.0, b=1.0, l=-1.0, r=1.0):
    '''
    Jacobi Bases. Please refer to our paper for the form of the bases.
    '''
    if L == 0: return xs[0]
    if L == 1:
        coef1 = (a - b) / 2 - (a + b + 2) / 2 * (l + r) / (r - l)
        coef1 *= alphas[0]
        coef2 = (a + b + 2) / (r - l)
        coef2 *= alphas[0]
        return coef1 * xs[-1] + coef2 * (adj @ xs[-1])
    coef_l = 2 * L * (L + a + b) * (2 * L - 2 + a + b)
    coef_lm1_1 = (2 * L + a + b - 1) * (2 * L + a + b) * (2 * L + a + b - 2)
    coef_lm1_2 = (2 * L + a + b - 1) * (a ** 2 - b ** 2)
    coef_lm2 = 2 * (L - 1 + a) * (L - 1 + b) * (2 * L + a + b)
    tmp1 = alphas[L - 1] * (coef_lm1_1 / coef_l)
    tmp2 = alphas[L - 1] * (coef_lm1_2 / coef_l)
    tmp3 = alphas[L - 1] * alphas[L - 2] * (coef_lm2 / coef_l)
    tmp1_2 = tmp1 * (2 / (r - l))
    tmp2_2 = tmp1 * ((r + l) / (r - l)) + tmp2
    nx = tmp1_2 * (adj @ xs[-1]) - tmp2_2 * xs[-1]
    nx -= tmp3 * xs[-2]
    return nx


def buildAdj(edge_index: Tensor, edge_weight: Tensor, n_node: int, aggr: str):
    '''
    convert edge_index and edge_weight to the sparse adjacency matrix.
    Args:
        edge_index (Tensor): shape (2, number of edges).
        edge_attr (Tensor): shape (number of edges).
        n_node (int): number of nodes in the graph.
        aggr (str): how adjacency matrix is normalized. choice: ["mean", "sum", "gcn"]
    '''
    deg = degree(edge_index[0], n_node)
    deg[deg < 0.5] += 1.0
    ret = None
    if aggr == "mean":
        val = (1.0 / deg)[edge_index[0]] * edge_weight
    elif aggr == "sum":
        val = edge_weight
    elif aggr == "gcn":
        deg = torch.pow(deg, -0.5)
        val = deg[edge_index[0]] * edge_weight * deg[edge_index[1]]
    else:
        raise NotImplementedError
    ret = SparseTensor(row=edge_index[0],
                       col=edge_index[1],
                       value=val,
                       sparse_sizes=(n_node, n_node)).coalesce()
    ret = ret.cuda() if edge_index.is_cuda else ret
    return ret


class Orth_PE_aggregation(MessagePassing):
    def __init__(self, num_lin, PE_dropout, PE_in_dim, PE_hid_dim, PE_alpha, PE_beta, **kwargs):
        super(Orth_PE_aggregation, self).__init__(aggr='add', **kwargs)
        self.PE_dropout = PE_dropout
        self.pe_linear = nn.Linear(PE_in_dim, PE_hid_dim)
        self.PE_alpha = PE_alpha
        self.num_lin = num_lin
        pe_to_coeff = []
        for _ in range(num_lin):
            pe_to_coeff.append(nn.Linear(PE_hid_dim, 1))
        self.pe_to_coeff = nn.ModuleList(pe_to_coeff)
        self.PE_beta = PE_beta
        self.cor_lin = nn.Linear(PE_hid_dim, PE_hid_dim)

    def PE2Coeff(self, pe, depth_idx):
        return self.pe_to_coeff[depth_idx](pe)

    def corr_node_propagate(self, pe):
        pe_adj = torch.matmul(self.cor_lin(pe), self.cor_lin(pe).T)
        pe_adj = torch.sigmoid(pe_adj)
        pe_corr = torch.matmul(pe_adj, pe)
        return pe_corr

    def forward(self, pe, edge_index):
        gamma_ls = []
        pe = self.pe_linear(pe)
        pe = torch.tanh(pe)
        if self.PE_dropout != 0.0:
            pe = F.dropout(pe, p=self.PE_dropout, training=self.training)
        gamma_ls.append(self.PE2Coeff(pe, 0))
        edge_index__appnp, norm__appnp = gcn_norm(edge_index, num_nodes=pe.size(0), dtype=pe.dtype)
        raw_pe = pe
        for i in range(1, self.num_lin):
            # Top-Cor Combination
            pe_tpo = self.propagate(edge_index__appnp, x=pe, norm=norm__appnp)  # topology mining
            pe_corr = self.corr_node_propagate(pe)  # correlation mining
            pe = (1 + self.PE_beta) * pe_tpo - self.PE_beta * pe_corr  # PE_beta in [1e-3, 10]
            pe = self.PE_alpha * raw_pe + (1 - self.PE_alpha) * pe
            pe = torch.tanh(pe)
            gamma_ls.append(self.PE2Coeff(pe, i))
        return gamma_ls, pe, None


class PE_aggregation(MessagePassing):
    def __init__(self, num_lin, PE_dropout, PE_in_dim, PE_hid_dim, PE_alpha, **kwargs):
        super(PE_aggregation, self).__init__(aggr='add', **kwargs)
        self.PE_dropout = PE_dropout
        self.pe_linear = nn.Linear(PE_in_dim, PE_hid_dim)
        self.PE_alpha = PE_alpha
        self.num_lin = num_lin
        pe_to_coeff = []
        for _ in range(num_lin):
            pe_to_coeff.append(nn.Linear(PE_hid_dim, 1))
        self.pe_to_coeff = nn.ModuleList(pe_to_coeff)

    def PE2Coeff(self, pe, depth_idx):
        return self.pe_to_coeff[depth_idx](pe)

    def forward(self, pe, edge_index):
        gamma_ls = []
        pos_emb_ls = []
        pe = self.pe_linear(pe)
        pe = torch.tanh(pe)
        if self.PE_dropout != 0.0:
            pe = F.dropout(pe, p=self.PE_dropout, training=self.training)
        gamma_ls.append(self.PE2Coeff(pe, 0))
        pos_emb_ls.append(pe)
        edge_index__appnp, norm__appnp = gcn_norm(edge_index, num_nodes=pe.size(0), dtype=pe.dtype)
        raw_pe = pe
        for i in range(1, self.num_lin):
            pe = self.PE_alpha * raw_pe + (1 - self.PE_alpha) * self.propagate(edge_index__appnp, x=pe, norm=norm__appnp)
            pe = torch.tanh(pe)
            gamma_ls.append(self.PE2Coeff(pe, i))
            pos_emb_ls.append(pe)
        return gamma_ls, pe, pos_emb_ls


class Combination(nn.Module):
    def __init__(self, channels: int, depth: int, sole=False):
        super().__init__()
        if sole:
            self.comb_weight = nn.Parameter(torch.ones((1, depth, 1)))
        else:
            self.comb_weight = nn.Parameter(torch.ones((1, depth, channels)))

    def forward(self, x):
        x = x * self.comb_weight
        x = torch.sum(x, dim=1)
        return x


class Gmodel(nn.Module):
    def __init__(self, emb: nn.Module, conv: nn.Module, comb: nn.Module, pe_agg=None):
        super().__init__()
        self.emb = emb
        self.conv = conv
        self.comb = comb
        self.pe_agg = pe_agg

    def forward(self, x, edge_index: Tensor, edge_weight: Tensor, pe=None):
        pe_alphas, pe, _ = self.pe_agg(pe, edge_index)
        x = self.conv(self.emb(x), edge_index, edge_weight, pe_alphas=pe_alphas)
        nemb = self.comb(x)
        return nemb, pe
