from torch_geometric.utils import to_undirected, remove_self_loops
from dataset import load_nc_dataset
from scipy import sparse as sp
import dgl
import torch
import numpy as np


def lap_positional_encoding(g, pos_enc_dim):
    """
        Graph positional encoding v/ Laplacian eigenvectors
        # downloaded from https://github.com/vijaydwivedi75/gnn-lspe
    """

    # Laplacian
    A = g.adjacency_matrix_scipy(return_edge_ids=False).astype(float)
    N = sp.diags(dgl.backend.asnumpy(g.in_degrees()).clip(1) ** -0.5, dtype=float)
    L = sp.eye(g.number_of_nodes()) - N * A * N

    # Eigenvectors with numpy
    EigVal, EigVec = np.linalg.eig(L.toarray())
    idx = EigVal.argsort()  # increasing order
    EigVal, EigVec = EigVal[idx], np.real(EigVec[:, idx])
    pos_enc = torch.from_numpy(EigVec[:, 1:pos_enc_dim + 1]).float()
    return pos_enc


def init_positional_encoding(g, pos_enc_dim):
    """
        Initializing positional encoding with RWPE
        # downloaded from https://github.com/vijaydwivedi75/gnn-lspe
    """

    # n = g.number_of_nodes()

    # Geometric diffusion features with Random Walk
    A = g.adjacency_matrix(scipy_fmt="csr")
    Dinv = sp.diags(dgl.backend.asnumpy(g.in_degrees()).clip(1) ** -1.0, dtype=float)  # D^-1
    RW = A * Dinv
    M = RW

    # Iterate
    nb_pos_enc = pos_enc_dim
    PE = [torch.from_numpy(M.diagonal()).float()]
    M_power = M
    for _ in range(nb_pos_enc - 1):
        M_power = M_power * M
        PE.append(torch.from_numpy(M_power.diagonal()).float())
    PE = torch.stack(PE, dim=-1)
    return PE


def generate_pos_enc(datname, sub_datname="", pe_indim=None):
    dataset = load_nc_dataset(datname, sub_datname)
    dataset.graph["edge_index"] = remove_self_loops(dataset.graph["edge_index"])[0]
    dataset.graph["edge_index"] = to_undirected(dataset.graph["edge_index"])
    # compute node positional embedding
    g = dgl.graph((dataset.graph["edge_index"][0], dataset.graph["edge_index"][1]))
    RW_PE = init_positional_encoding(g, pos_enc_dim=pe_indim)
    LAP_PE = lap_positional_encoding(g, pos_enc_dim=pe_indim)
    return RW_PE, LAP_PE


def main():
    datname, sub_datname = "chameleon", ""
    sav_path = "data/node_pos_enc/"
    for pe_indim in np.arange(2, 34, 2):  # 2, 4, ..., 32
        pe_indim = int(pe_indim)
        print(f"{datname}_{sub_datname}_{pe_indim}")
        RW_PE, LAP_PE = generate_pos_enc(datname, sub_datname, pe_indim=pe_indim)
        torch.save(RW_PE, f"{sav_path}RW_PE_{datname}_indim_{pe_indim}.pt")
        torch.save(LAP_PE, f"{sav_path}LAP_PE_{datname}_indim_{pe_indim}.pt")


if __name__ == '__main__':
    main()
