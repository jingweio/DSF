# developed based on https://github.com/jianhao2016/GPRGNN
from torch_geometric.nn import MessagePassing
from torch_geometric.nn.conv.gcn_conv import gcn_norm
import torch.nn.functional as F
from torch.nn import Parameter, Linear
import torch.nn as nn
import torch
import numpy as np


class DSF_GPR_I(torch.nn.Module):
    def __init__(self, in_dim, hid_dim, out_dim, args, PE_in_dim, PE_hid_dim, PE_alpha, PE_beta, K=10):
        super(DSF_GPR_I, self).__init__()
        self.dprate = args.dprate
        self.dropout = args.dropout
        self.PE_dropout = args.PE_dropout
        self.lin1 = Linear(in_dim, hid_dim)
        self.lin2 = Linear(hid_dim, out_dim)
        self.pe_linear = nn.Linear(PE_in_dim, PE_hid_dim)
        self.prop1 = _prop(K, args.alpha, args.Init, PE_hid_dim=PE_hid_dim, PE_alpha=PE_alpha, PE_beta=PE_beta)

    def forward(self, data):
        x, edge_index = data.graph['node_feat'], data.graph['edge_index']
        pe = data.graph['pos_enc']

        pe = self.pe_linear(pe)
        pe = torch.tanh(pe)
        if self.PE_dropout != 0.0:
            pe = F.dropout(pe, p=self.PE_dropout, training=self.training)

        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.relu(self.lin1(x))
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
    def __init__(self, K, alpha, Init, PE_hid_dim, PE_alpha, PE_beta, **kwargs):
        super(_prop, self).__init__(aggr='add', **kwargs)
        self.K = K

        assert Init in ['PPR', 'NPPR', 'Random']
        if Init == 'PPR':
            # PPR-like
            TEMP = alpha * (1 - alpha) ** np.arange(K + 1)
            TEMP[-1] = (1 - alpha) ** K
        elif Init == 'NPPR':
            # Negative PPR
            TEMP = (alpha) ** np.arange(K + 1)
            TEMP = TEMP / np.sum(np.abs(TEMP))
        elif Init == 'Random':
            # Random
            bound = np.sqrt(3 / (K + 1))
            TEMP = np.random.uniform(-bound, bound, K + 1)
            TEMP = TEMP / np.sum(np.abs(TEMP))

        self.temp = Parameter(torch.tensor(TEMP))

        pe_to_coeff = []
        for _ in range(K + 1):
            pe_to_coeff.append(nn.Linear(PE_hid_dim, 1))
        self.pe_to_coeff = nn.ModuleList(pe_to_coeff)
        self.PE_alpha = PE_alpha
        self.PE_beta = PE_beta
        self.cor_lin = nn.Linear(PE_hid_dim, PE_hid_dim)

    def corr_node_propagate(self, pe):
        pe_adj = torch.matmul(self.cor_lin(pe), self.cor_lin(pe).T)
        pe_adj = torch.sigmoid(pe_adj)
        pe_corr = torch.matmul(pe_adj, pe)
        return pe_corr

    def forward(self, x, edge_index, edge_weight=None, pe=None):
        edge_index, norm = gcn_norm(edge_index, edge_weight, num_nodes=x.size(0), dtype=x.dtype)

        glob_gamma = self.temp[0]
        local_gamma = self.pe_to_coeff[0](pe)
        gamma = glob_gamma * torch.tanh(local_gamma)
        hidden = x * gamma
        raw_pe = pe
        for k in range(self.K):
            x = self.propagate(edge_index, x=x, norm=norm)
            # Top-Cor Combination
            pe_tpo = self.propagate(edge_index, x=pe, norm=norm)  # topology mining
            pe_corr = self.corr_node_propagate(pe)  # correlation mining
            pe = (1 + self.PE_beta) * pe_tpo - self.PE_beta * pe_corr  # PE_beta in {0,0.1,...,1}
            # Final-Update
            pe = self.PE_alpha * raw_pe + (1 - self.PE_alpha) * pe
            pe = torch.tanh(pe)
            # Local-Global Parameter
            glob_gamma = self.temp[k + 1]
            local_gamma = self.pe_to_coeff[k + 1](pe)
            gamma = glob_gamma * torch.tanh(local_gamma)
            hidden = hidden + gamma * x
        return hidden, pe

    def message(self, x_j, norm):
        return norm.view(-1, 1) * x_j

    def __repr__(self):
        return '{}(K={}, temp={})'.format(self.__class__.__name__, self.K,
                                          self.temp)
