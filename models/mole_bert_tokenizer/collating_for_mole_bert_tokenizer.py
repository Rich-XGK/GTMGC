"""
    Collator for MoleBERTTokenizer
"""
import torch
from typing import Any, Dict, Sequence
from torch_geometric.data import Data, Batch


class MoleBERTTokenizerCollator:
    """
    Collator for MoleBERTTokenizer
    """

    def __init__(self) -> None:
        pass

    @torch.no_grad()
    def __call__(self, mol_sq: Sequence[Dict]) -> Any:
        mol_data_seq = [self._get_pyg_data(mol) for mol in mol_sq]
        batch = Batch().from_data_list(mol_data_seq)
        return {
            # "node_attr": batch.node_attr,
            # "edge_attr": batch.edge_attr,
            "node_type": batch.node_type,
            "node_chiral_type": batch.node_chiral_type,
            "edge_type": batch.edge_type,
            "edge_dire_type": batch.edge_dire_type,
            "edge_index": batch.edge_index,
        }

    def _get_pyg_data(self, mol: Dict) -> Data:
        return Data(
            # node_attr=torch.tensor(mol["node_attr"], dtype=torch.long),
            # edge_attr=torch.tensor(mol["edge_attr"], dtype=torch.long),
            node_type=torch.tensor(mol["node_type"], dtype=torch.long),
            node_chiral_type=torch.tensor(mol["node_chiral_type"], dtype=torch.long),
            edge_type=torch.tensor(mol["edge_type"], dtype=torch.long),
            edge_dire_type=torch.tensor(mol["edge_dire_type"], dtype=torch.long),
            edge_index=torch.tensor(mol["edge_index"], dtype=torch.long),
        )
