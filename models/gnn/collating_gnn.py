"""
    Collator for GNNs
"""
import torch
from typing import Any, Dict, Sequence
from torch_geometric.data import Data, Batch


class GNNCollator:
    def __init__(self) -> None:
        pass

    @torch.no_grad()
    def __call__(self, mol_sq: Sequence[Dict]) -> Any:
        mol_data_seq = [self._get_pyg_data(mol) for mol in mol_sq]
        batch = Batch().from_data_list(mol_data_seq)
        return {
            "node_type": batch.node_type,
            "edge_index": batch.edge_index,
            "node_attr": batch.node_attr,
            "edge_attr": batch.edge_attr,
            "conformer": batch.conformer,
            "batch": batch.batch,
        }

    def _get_pyg_data(self, mol: Dict) -> Data:
        return Data(
            node_type=torch.tensor(mol["node_type"], dtype=torch.long),
            edge_index=torch.tensor(mol["edge_index"], dtype=torch.long),
            node_attr=torch.tensor(mol["node_attr"], dtype=torch.long),
            edge_attr=torch.tensor(mol["edge_attr"], dtype=torch.long),
            conformer=torch.tensor(mol["conformer"], dtype=torch.float32),
        )
