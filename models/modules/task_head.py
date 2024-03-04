import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict

from torchmetrics import AUROC
from .gnn import GNNDecoder
from .utils import make_cdist_mask, mask_hidden_state, align_conformer_hat_to_conformer, make_mask_for_pyd_batch_graph


class ConformerPredictionHead(nn.Module):
    def __init__(self, hidden_X_dim: int = 256) -> None:
        super().__init__()
        self.head = nn.Sequential(nn.Linear(hidden_X_dim, hidden_X_dim * 3), nn.GELU(), nn.Linear(hidden_X_dim * 3, 3))
        # self.criterion = MultiTaskLearnableWeightLoss(n_task=2)

    def forward(self, conformer: torch.Tensor, hidden_X: torch.Tensor, padding_mask: torch.Tensor = None, compute_loss: bool = True) -> torch.Tensor:
        # get conformer_hat
        conformer_hat = self.head(hidden_X)
        # mask padding atoms
        conformer_hat = mask_hidden_state(conformer_hat, padding_mask)
        # align conformer_hat to conformer
        conformer_hat = align_conformer_hat_to_conformer(conformer_hat, conformer)
        conformer_hat = mask_hidden_state(conformer_hat, padding_mask)
        if not compute_loss:
            return conformer_hat
        return self._compute_loss(conformer, conformer_hat, padding_mask)

    def _compute_loss(self, conformer: torch.Tensor, conformer_hat: torch.Tensor, padding_mask: torch.Tensor = None) -> torch.Tensor:
        # Convenient for design loss function in the future.
        cdist, cdist_hat = torch.cdist(conformer, conformer), torch.cdist(conformer_hat, conformer_hat)
        c_dist_mask = make_cdist_mask(padding_mask)
        cdist, cdist_hat = cdist * c_dist_mask, cdist_hat * c_dist_mask
        cdist_mae = self._compute_cdist_mae(cdist, cdist_hat, c_dist_mask)
        cdist_mse = self._compute_cdist_mse(cdist, cdist_hat, c_dist_mask)
        coord_rmsd = self._compute_conformer_rmsd(conformer, conformer_hat, padding_mask)
        # compute learnable weighted loss
        # loss = self.criterion([cdist_mae, cdist_rmse])
        loss = cdist_mae
        return {
            "loss": loss,
            "cdist_mae": cdist_mae.detach(),
            "cdist_mse": cdist_mse.detach(),
            "coord_rmsd": coord_rmsd.detach(),
            "conformer": conformer.detach(),
            "conformer_hat": conformer_hat.detach(),
        }

    @staticmethod
    def _compute_cdist_mae(masked_cdist: torch.Tensor, masked_cdist_hat: torch.Tensor, cdist_mask: torch.Tensor) -> torch.Tensor:
        """Compute mean absolute error of conformer and conformer_hat.

        Args:
            - masked_cdist (torch.Tensor): A torch tensor of shape (b, l, l), which denotes the pairwise distance matrix of the conformer.
            - masked_cdist_hat (torch.Tensor): A torch tensor of shape (b, l, l), which denotes the pairwise distance matrix of the conformer_hat.
            - cdist_mask (torch.Tensor): A torch tensor of shape (b, l, l), which denotes the mask of the pairwise distance matrix.

        Returns:
            torch.Tensor: The mean absolute error of conformer and conformer_hat.
        """
        mae = F.l1_loss(masked_cdist, masked_cdist_hat, reduction="sum") / cdist_mask.sum()  # exclude padding atoms
        return mae

    @staticmethod
    def _compute_cdist_mse(masked_cdist: torch.Tensor, masked_cdist_hat: torch.Tensor, cdist_mask: torch.Tensor) -> torch.Tensor:
        """Compute root mean squared error of conformer and conformer_hat.

        Args:
            - masked_cdist (torch.Tensor): A torch tensor of shape (b, l, l), which denotes the pairwise distance matrix of the conformer.
            - masked_cdist_hat (torch.Tensor): A torch tensor of shape (b, l, l), which denotes the pairwise distance matrix of the conformer_hat.
            - cdist_mask (torch.Tensor): A torch tensor of shape (b, l, l), which denotes the mask of the pairwise distance matrix.

        Returns:
            torch.Tensor: The root mean squared error of conformer and conformer_hat.
        """
        mse = F.mse_loss(masked_cdist, masked_cdist_hat, reduction="sum") / cdist_mask.sum()  # exclude padding atoms
        return mse

    @staticmethod
    def _compute_conformer_rmsd(masked_conformer: torch.Tensor, masked_conformer_hat: torch.Tensor, padding_mask: torch.Tensor) -> torch.Tensor:
        """Compute root mean squared deviation of conformer and conformer_hat.

        Args:
            - masked_conformer (torch.Tensor): A torch tensor of shape (b, l, 3), which denotes the coordinate of the conformer.
            - masked_conformer_hat (torch.Tensor): A torch tensor of shape (b, l, 3), which denotes the coordinate of the conformer_hat.
            - padding_mask (torch.Tensor): A torch tensor of shape (b, l), which denotes the mask of the conformer.

        Returns:
            torch.Tensor: The root mean squared deviation of conformer and conformer_hat.
        """
        R, R_h, M = masked_conformer, masked_conformer_hat, padding_mask
        delta = (R - R_h).to(torch.float32)
        point_2_norm = torch.norm(delta, p=2, dim=-1)
        MSD = torch.sum(point_2_norm**2, dim=-1) / torch.sum(M, dim=-1)
        RMSD = torch.sqrt(MSD)
        return RMSD.mean()


class GNNConformerPredictionHead(nn.Module):
    def __init__(self, hidden_X_dim: int = 300) -> None:
        super().__init__()
        self.head = nn.Sequential(nn.Linear(hidden_X_dim, hidden_X_dim * 3), nn.ReLU(), nn.Linear(hidden_X_dim * 3, 3))

    def forward(self, conformer: torch.Tensor, hidden_X: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
        conformer_hat = self.head(hidden_X)
        return self._compute_loss(conformer, conformer_hat, batch)

    def _compute_loss(self, conformer: torch.Tensor, conformer_hat: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
        cdist = torch.cdist(conformer, conformer)
        cdist_hat = torch.cdist(conformer_hat, conformer_hat)
        M = make_mask_for_pyd_batch_graph(batch)
        masked_cdist, masked_cdist_hat = cdist * M, cdist_hat * M
        mae = self._compute_cdist_mae(masked_cdist, masked_cdist_hat, M)
        mse = self._compute_cdist_mse(masked_cdist, masked_cdist_hat, M)
        return {
            "loss": mae,
            "cdist_mae": mae,
            "cdist_mse": mse,
            "conformer": conformer,
            "conformer_hat": conformer_hat,
        }

    @staticmethod
    def _compute_cdist_mae(masked_cdist: torch.Tensor, masked_cdist_hat: torch.Tensor, cdist_mask: torch.Tensor) -> torch.Tensor:
        D, D_h, M = masked_cdist, masked_cdist_hat, cdist_mask
        mae = F.l1_loss(D, D_h, reduction="sum") / M.sum()
        return mae

    @staticmethod
    def _compute_cdist_mse(masked_cdist: torch.Tensor, masked_cdist_hat: torch.Tensor, cdist_mask: torch.Tensor) -> torch.Tensor:
        D, D_h, M = masked_cdist, masked_cdist_hat, cdist_mask
        mse = F.mse_loss(D, D_h, reduction="sum") / M.sum()
        return mse


class GraphReConstructionHead(nn.Module):
    num_node_type = 119
    num_node_chiral = 4
    num_edge_type = 4

    def __init__(self, in_dim: int = 300, hidden_dim: int = 600, dropout: float = 0.0, re_build_edge: bool = False) -> None:
        super().__init__()
        self.re_build_edge = re_build_edge
        self.atom_re_constructor = GNNDecoder(
            in_dim=in_dim,
            hidden_dim=hidden_dim,
            out_dim=self.num_node_type,
            dropout=dropout,
        )
        self.atom_chiral_re_constructor = GNNDecoder(
            in_dim=in_dim,
            hidden_dim=hidden_dim,
            out_dim=self.num_node_chiral,
            dropout=dropout,
        )
        if re_build_edge:
            self.edge_re_constructor = GNNDecoder(
                in_dim=in_dim,
                hidden_dim=hidden_dim,
                out_dim=self.num_edge_type,
                dropout=dropout,
            )

    def forward(self, **inputs) -> Dict[str, torch.Tensor]:
        node_representation = inputs.get("node_representation")
        node_type = inputs.get("node_type")
        node_chiral_type = inputs.get("node_chiral_type")
        edge_type = inputs.get("edge_type")
        edge_dire_type = inputs.get("edge_dire_type")
        edge_index = inputs.get("edge_index")

        node_logits = self.atom_re_constructor(
            node_attr=node_representation,
            edge_index=edge_index,
            edge_type=edge_type,
            edge_dire_type=edge_dire_type,
        )
        node_chiral_logits = self.atom_chiral_re_constructor(
            node_attr=node_representation, edge_index=edge_index, edge_type=edge_type, edge_dire_type=edge_dire_type
        )
        edge_logits = None
        if self.re_build_edge:
            edge_representation = node_representation[edge_index[0]] + node_representation[edge_index[1]]
            edge_logits = self.edge_re_constructor(node_attr=edge_representation, edge_index=edge_index, edge_type=edge_type, edge_dire_type=edge_dire_type)
        reconstruction_loss = self._compute_loss(
            node_logits,
            node_chiral_logits,
            edge_logits,
            node_type,
            node_chiral_type,
            edge_type,
        )
        reconstruction_accuracy = self._compute_accuracy(node_logits, node_chiral_logits, edge_logits, node_type, node_chiral_type, edge_type)
        return {"reconstruction_loss": reconstruction_loss, "reconstruction_accuracy": reconstruction_accuracy.detach()}

    def _compute_loss(self, node_logits, node_chiral_logits, edge_logits, node_type, node_chiral_type, edge_type):
        node_type_loss = F.cross_entropy(node_logits, node_type)
        node_chiral_loss = F.cross_entropy(node_chiral_logits, node_chiral_type)
        if self.re_build_edge:
            edge_type_loss = F.cross_entropy(edge_logits, edge_type)
            return node_type_loss + node_chiral_loss + edge_type_loss
        return node_type_loss + node_chiral_loss

    def _compute_accuracy(self, node_logits, node_chiral_logits, edge_logits, node_type, node_chiral_type, edge_type):
        node_type_preds, node_chiral_preds = node_logits.detach().argmax(dim=-1), node_chiral_logits.detach().argmax(dim=-1)
        node_type_accuracy = (node_type_preds == node_type.detach()).float().mean()
        node_chiral_accuracy = (node_chiral_preds == node_chiral_type.detach()).float().mean()
        if self.re_build_edge:
            edge_type_preds = edge_logits.detach().argmax(dim=-1)
            edge_type_accuracy = (edge_type_preds == edge_type.detach()).float().mean()
            # return (node_type_accuracy + node_chiral_accuracy + edge_type_accuracy) / 3
        # return (node_type_accuracy + node_chiral_accuracy) / 2
        return node_type_accuracy  # cause the node_type_accuracy and node_chiral_accuracy grow fast, so we only use node_type_accuracy


class GraphRegressionHead(nn.Module):
    def __init__(self, hidden_X_dim: int = 256) -> None:
        super().__init__()
        self.head = nn.Sequential(nn.Linear(hidden_X_dim, hidden_X_dim * 3), nn.GELU(), nn.Linear(hidden_X_dim * 3, 1))

    def forward(self, hidden_X: torch.tensor, labels: torch.Tensor) -> torch.Tensor:
        logits = self.head(hidden_X).squeeze(-1)
        loss = F.l1_loss(logits, labels)
        mse = F.mse_loss(logits, labels)
        return {
            "loss": loss,
            "mae": loss.detach(),
            "mse": mse.detach(),
            "logits": logits.detach(),
            "labels": labels.detach(),
        }
