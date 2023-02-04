# developed based on https://github.com/ivam-he/BernNet
from torch_geometric.nn.conv.gcn_conv import gcn_norm
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import get_laplacian, add_self_loops
from scipy.special import comb
import torch.nn.functional as F
import torch.nn as nn
import torch


class DSF_Bern_I(torch.nn.Module):
    def __init__(self, in_dim, hid_dim, out_dim, args, PE_in_dim, PE_hid_dim, PE_alpha=0.1, PE_beta=0.5, K=10):
        super(DSF_Bern_I, self).__init__()
        args.K = K
        self.dprate = args.dprate
        self.dropout = args.dropout
        self.PE_dropout = args.PE_dropout
        self.lin1 = nn.Linear(in_dim, hid_dim)
        self.lin2 = nn.Linear(hid_dim, out_dim)
        self.pe_linear = nn.Linear(PE_in_dim, PE_hid_dim)
        self.prop1 = _prop(args.K, PE_hid_dim, PE_alpha, PE_beta)

    def forward(self, data):
        x, edge_index = data.graph['node_feat'], data.graph['edge_index']
        pe = data.graph['pos_enc']

        pe = self.pe_linear(pe)
        pe = torch.tanh(pe)
        if self.PE_dropout != 0.0:
            pe = F.dropout(pe, p=self.PE_dropout, training=self.training)

        x = F.dropout(x, p=self.dropout, training=self.training)
        x = torch.relu(self.lin1(x))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lin2(x)

        if self.dprate == 0.0:
            x, pe = self.prop1(x, edge_index, pe=pe)
            return x, pe
        else:
            x = F.dropout(x, p=self.dprate, training=self.training)
            x, pe = self.prop1(x, edge_index, pe=pe)
            return x, pe


class _prop(MessagePassing):
    def __init__(self, K, PE_hid_dim, PE_alpha, PE_beta, **kwargs):
        super(_prop, self).__init__(aggr='add', **kwargs)
        self.K = K
        pe_to_coeff = []
        for _ in range(K + 1):
            pe_to_coeff.append(nn.Sequential(nn.Linear(PE_hid_dim, 1), nn.Sigmoid()))
        self.pe_to_coeff = nn.ModuleList(pe_to_coeff)
        self.PE_alpha = PE_alpha
        self.PE_beta = PE_beta
        self.cor_lin = nn.Linear(PE_hid_dim, PE_hid_dim)

        self.temp = nn.Parameter(torch.Tensor(self.K + 1))
        self.reset_parameters()

    def reset_parameters(self):
        self.temp.data.fill_(1)

    def corr_node_propagate(self, pe):
        pe_adj = torch.matmul(self.cor_lin(pe), self.cor_lin(pe).T)
        pe_adj = torch.sigmoid(pe_adj)
        pe_corr = torch.matmul(pe_adj, pe)
        return pe_corr

    def forward(self, x, edge_index, edge_weight=None, pe=None):
        TEMP = F.relu(self.temp)
        edge_index1, norm1 = get_laplacian(edge_index, edge_weight, normalization='sym', dtype=x.dtype, num_nodes=x.size(self.node_dim))  # L=I-D^(-0.5)AD^(-0.5)
        edge_index2, norm2 = add_self_loops(edge_index1, -norm1, fill_value=2., num_nodes=x.size(self.node_dim))  # 2I-L
        tmp = []
        tmp.append(x)
        for i in range(self.K):
            x = self.propagate(edge_index2, x=x, norm=norm2, size=None)
            tmp.append(x)

        glob_gamma = TEMP[0]
        local_gamma = self.pe_to_coeff[0](pe)
        gamma = glob_gamma * local_gamma
        out = (comb(self.K, 0) / (2 ** self.K)) * gamma * tmp[self.K]
        edge_index__appnp, norm__appnp = gcn_norm(edge_index, edge_weight, num_nodes=x.size(0), dtype=x.dtype)
        raw_pe = pe

        for i in range(self.K):
            x = tmp[self.K - i - 1]
            x = self.propagate(edge_index1, x=x, norm=norm1, size=None)
            for j in range(i):
                x = self.propagate(edge_index1, x=x, norm=norm1, size=None)
            # Top-Cor Combination
            pe_tpo = self.propagate(edge_index__appnp, x=pe, norm=norm__appnp)
            pe_corr = self.corr_node_propagate(pe)  # correlation mining
            pe = (1 + self.PE_beta) * pe_tpo - self.PE_beta * pe_corr  # PE_beta in {0, 0.1, ..., 1}
            # Final-Update
            pe = self.PE_alpha * raw_pe + (1 - self.PE_alpha) * pe
            pe = torch.tanh(pe)
            glob_gamma = TEMP[i + 1]
            local_gamma = self.pe_to_coeff[i + 1](pe)
            gamma = glob_gamma * local_gamma
            out = out + (comb(self.K, i + 1) / (2 ** self.K)) * gamma * x
        return out, pe

    def message(self, x_j, norm):
        return norm.view(-1, 1) * x_j

    def __repr__(self):
        return '{}(K={}, temp={})'.format(self.__class__.__name__, self.K,
                                          self.temp)
