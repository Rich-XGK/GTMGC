"""
    Collator for Conformer.
"""

import torch
import numpy as np
from typing import Dict, List, Sequence


def valid_length_to_mask(valid_length: List[int], max_length: int = None) -> np.ndarray:
    max_length = max(valid_length) if max_length is None else max_length
    mask = np.zeros((len(valid_length), max_length), dtype=np.int64)
    for i, length in enumerate(valid_length):
        mask[i, :length] = 1
    return mask


def get_adjacency(num_atoms: int, edge_indexes: List[List[int]]) -> np.ndarray:
    assert len(edge_indexes) == 2 and len(edge_indexes[0]) == len(edge_indexes[1])
    adjacency = np.zeros((num_atoms, num_atoms))
    adjacency[edge_indexes[0], edge_indexes[1]] = 1
    return adjacency


def get_k_hop_adjacency(A: np.ndarray, k: int = 1, current_node: bool = False) -> np.ndarray:
    """Get k-hop adjacency matrix from adjacency matrix A.
    Args:
        A (np.ndarray): adjacency matrix, shape (N, N)
        k (int, optional): k-hop. Defaults to 1.
        current_node (bool, optional): whether to include the current node. Defaults to False.
    Returns:
        np.ndarray: k-hop adjacency matrix, shape (N, N)
    """
    assert k >= 1, "k must be greater than or equal to 1."
    if k == 1:
        return A if not current_node else A + np.eye(A.shape[0])
    A_k_hop = np.asarray(A, dtype=np.float32)
    for i in range(2, k + 1):
        A_K = np.linalg.matrix_power(A, i)
        A_k_hop += A_K
    A_k_hop = (A_k_hop > 0).astype(np.int64)
    if not current_node:
        A_k_hop -= np.eye(A.shape[0], dtype=np.int64)
    return A_k_hop


def get_laplacian_eigenvectors(adjacency: np.ndarray) -> np.ndarray:
    A = adjacency
    l, _ = A.shape
    epsilon = 1e-8
    D = np.diag(1 / np.sqrt(A.sum(axis=1) + epsilon))
    L = np.eye(l) - D @ A @ D
    w, v = np.linalg.eigh(L)
    return v


class GTMGCCollator:
    def __init__(self, max_nodes: int = None, max_edges: int = None, max_lap_eigenvalues: int = 10) -> None:
        self.max_nodes = max_nodes
        self.max_edges = max_edges
        self.max_lap_eigenvalues = max_lap_eigenvalues

    def __call__(self, mol_sq: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        # add num_nodes for each mol
        keys = mol_sq[0].keys()
        num_mol = len(mol_sq)
        num_nodes = [len(mol["node_type"]) for mol in mol_sq]
        max_nodes = max(num_nodes) if self.max_nodes is None else self.max_nodes
        node_type = np.zeros((num_mol, max_nodes), dtype=np.int64)
        lap_eigenvectors = np.zeros((num_mol, max_nodes, max_nodes), dtype=np.float32)
        adjacency = np.zeros((num_mol, max_nodes, max_nodes), dtype=np.int64)
        in_degree = np.zeros((num_mol, max_nodes), dtype=np.int64)
        out_degree = np.zeros((num_mol, max_nodes), dtype=np.int64)
        node_input_ids = np.zeros((num_mol, max_nodes), dtype=np.int64) if "input_ids" in keys else None
        conformer = np.zeros((num_mol, max_nodes, 3), dtype=np.float32) if "conformer" in keys else None
        labels = np.array([mol["labels"] for mol in mol_sq], dtype=np.float32) if "labels" in keys else None
        node_attr = np.zeros((num_mol, max_nodes, 9), dtype=np.int64) if "node_attr" in keys else None
        for i, mol in enumerate(mol_sq):
            adj = get_adjacency(num_nodes[i], mol["edge_index"])
            lap_eigenvectors[i, : num_nodes[i], : num_nodes[i]] = get_laplacian_eigenvectors(adj)
            # adj = get_k_hop_adjacency(A=adj, k=3, current_node=True)
            adjacency[i, : num_nodes[i], : num_nodes[i]] = adj
            node_type[i, : num_nodes[i]] = np.array(mol["node_type"]) + 1  # 0 for padding
            in_degree[i, : num_nodes[i]] = np.sum(adj, axis=0, dtype=np.int64) + 1  # 0 for padding
            out_degree[i, : num_nodes[i]] = np.sum(adj, axis=1, dtype=np.int64) + 1  # 0 for padding
            if "input_ids" in keys:
                node_input_ids[i, : num_nodes[i]] = np.array(mol["input_ids"]) + 1  # 0 for padding
            if "conformer" in keys:
                conformer[i, : num_nodes[i]] = mol["conformer"]
            if "node_attr" in keys:
                node_attr[i, : num_nodes[i]] = np.array(mol["node_attr"]) + 1  # 0 for padding
        res_dic = {
            "node_type": torch.from_numpy(node_type),
            "node_mask": torch.from_numpy(valid_length_to_mask(num_nodes, max_nodes)),
            "adjacency": torch.from_numpy(adjacency),
            "lap_eigenvectors": torch.from_numpy(lap_eigenvectors),
            "in_degree": torch.from_numpy(in_degree),
            "out_degree": torch.from_numpy(out_degree),
        }
        if node_input_ids is not None:
            res_dic["node_input_ids"] = torch.from_numpy(node_input_ids)
        if labels is not None:
            res_dic["labels"] = torch.from_numpy(labels)
        if conformer is not None:
            res_dic["conformer"] = torch.from_numpy(conformer)
        if node_attr is not None:
            res_dic["node_attr"] = torch.from_numpy(node_attr)
        return res_dic
