"""
    This module contains basic GNN modules used in MoleBert.
"""

from typing import Any
import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, degree

NUM_ATOM_TYPE = 120  # Atom numbers 1~118 + extra mask tokens   TODO: What is the extra mask token?
NUM_CHIRALITY_TAG = 3  # {UNSPECIFIED, TETRAHEDRAL_CW, CHI_TETRAHEDRAL_CCW, CHI_OTHER}     TODO: Why 3?
NUM_BOND_TYPE = 6  # {SINGLE, DOUBLE, TRIPLE, AROMATIC, SELF_LOOP} + masked tokens    TODO: What is masked tokens?
NUM_BOND_DIRECTION = 3  # {NONE, ENDUPRIGHT, ENDDOWNRIGHT}


class GINConv(MessagePassing):
    """
    Extension of GIN aggregation to incorporate edge information by concatenation.
    """

    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int, aggr: str = "add", **kwargs):
        kwargs.setdefault("aggr", aggr)
        super().__init__(**kwargs)
        self.aggr = aggr
        self.mlp = nn.Sequential(nn.Linear(in_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, out_dim))
        self.edge_embedding01 = nn.Embedding(NUM_BOND_TYPE, in_dim)
        self.edge_embedding02 = nn.Embedding(NUM_BOND_DIRECTION, in_dim)

    def forward(self, **inputs) -> Any:
        node_attr = inputs.get("node_attr")
        edge_index = inputs.get("edge_index")
        edge_type = inputs.get("edge_type")
        edge_dire_type = inputs.get("edge_dire_type")
        # add self loops in the edge space
        edge_index, _ = add_self_loops(edge_index, num_nodes=node_attr.shape[0])
        # edge_index.shape = [2, num_edges + num_nodes]
        # [
        #   [0, 1, 2, 3, 4, 0, 1, 2, 3, 4],
        #   [1, 2, 3, 4, 0, 0, 1, 2, 3, 4]
        # ]

        # add features corresponding to self-loop edges
        # bond direction : NONE   bond type : SELF_LOOP
        self_loop_attr = torch.zeros(node_attr.size(0), 2, dtype=torch.long, device=node_attr.device)
        self_loop_attr[:, 0] = 4
        # computing edge embedding: bond type embedded feature + bond direction embedded feature
        edge_attr = self.edge_embedding01(edge_type) + self.edge_embedding02(edge_dire_type)
        self_loop_attr = self.edge_embedding01(self_loop_attr[:, 0]) + self.edge_embedding02(self_loop_attr[:, 1])
        edge_attr = torch.cat([edge_attr, self_loop_attr], dim=0)
        # edge_attr.shape = [num_edges + num_nodes, emb_dim]

        # propagate():
        # 1. message(): aggregate messages from neighbors;
        # 2. aggregate(): combine messages with self embeddings;
        # 3. update(): update self embeddings with aggregated messages.
        h = self.propagate(edge_index, x=node_attr, edge_attr=edge_attr)
        return h

    def message(self, x_j: torch.Tensor, edge_attr: torch.Tensor) -> torch.Tensor:
        # x_j.shape = [num_edges, emb_dim]  TODO: What is x_j and How to get it?
        # TODO: Is this edge_attr the same as the edge_attr in forward()?
        return x_j + edge_attr

    def update(self, aggr_out: torch.Tensor) -> torch.Tensor:
        # aggr_out.shape = [num_nodes, out_dim]
        return self.mlp(aggr_out)


class GNNEncoder(nn.Module):
    def __init__(self, num_layers: int = 5, embedding_dim=300, layer_hidden_dim: int = 600, jk="last", dropout: float = 0.0, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.num_layers = num_layers
        self.embedding_dim = embedding_dim
        self.jk = jk
        self.dropout = dropout

        if self.num_layers < 2:
            raise ValueError("Number of GNN layers must be greater than 1.")

        self.node_embedding01 = nn.Embedding(NUM_ATOM_TYPE, embedding_dim)
        self.node_embedding02 = nn.Embedding(NUM_CHIRALITY_TAG, embedding_dim)

        self.gnn_layers = nn.ModuleList()
        for _ in range(self.num_layers):
            self.gnn_layers.append(GINConv(embedding_dim, layer_hidden_dim, embedding_dim))

        self.batch_norms = nn.ModuleList()
        for _ in range(self.num_layers):
            self.batch_norms.append(nn.BatchNorm1d(embedding_dim))

    def forward(self, **inputs):
        node_type = inputs.get("node_type")
        node_chiral_type = inputs.get("node_chiral_type")
        edge_type = inputs.get("edge_type")
        edge_dire_type = inputs.get("edge_dire_type")
        edge_index = inputs.get("edge_index")

        # computing input node embedding: atom type embedded feature + chirality embedded feature
        node_attr = self.node_embedding01(node_type) + self.node_embedding02(node_chiral_type)
        h_list = [node_attr]  # list of hidden representation at each layer (including input)

        for i in range(self.num_layers):
            h = self.gnn_layers[i](node_attr=h_list[i], edge_index=edge_index, edge_type=edge_type, edge_dire_type=edge_dire_type)
            h = self.batch_norms[i](h)
            # h.shape = (num_nodes, emb_dim)
            if i == self.num_layers - 1:
                # remove relu activation for the last layer
                h = F.dropout(h, self.dropout, training=self.training)
            else:
                h = F.dropout(F.relu(h), self.dropout, training=self.training)
            h_list.append(h)

        # different implementations of Jk-concat
        if self.jk == "concat":
            node_representation = torch.cat(h_list, dim=1)
            # node_representation.shape = (num_nodes, emb_dim * num_layer)
        elif self.jk == "last":
            node_representation = h_list[-1]
            # node_representation.shape = (num_nodes, emb_dim)
        elif self.jk == "max":  # position-wise max
            h_list = [h.unsqueeze(0) for h in h_list]  # h.shape = (1, num_nodes, emb_dim)
            node_representation = torch.max(torch.cat(h_list, dim=0), dim=0)[0]
            # node_representation.shape = (num_nodes, emb_dim)
        elif self.jk == "sum":  # position-wise sum
            h_list = [h.unsqueeze(0) for h in h_list]
            node_representation = torch.sum(torch.cat(h_list, dim=0), dim=0)[0]
            # node_representation.shape = (num_nodes, emb_dim)

        return node_representation  # [num_nodes, emb_dim] / [num_nodes, emb_dim * num_layer]

    def __init__weights__(self):
        for m in self.modules():
            if isinstance(m, nn.Embedding):
                nn.init.xavier_uniform_(m.weight.data)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight.data)
                if m.bias is not None:
                    nn.init.zeros_(m.bias.data)


class GNNDecoder(nn.Module):
    def __init__(self, in_dim: int = 300, hidden_dim: int = 600, out_dim: int = 300, dropout: float = 0.0):
        super().__init__()
        self.dropout = dropout
        self.conv = GINConv(in_dim, hidden_dim, out_dim)
        self.act = nn.PReLU()
        self.enc_to_dec = nn.Linear(in_dim, in_dim, bias=False)

    def forward(self, **inputs):
        node_attr = inputs.get("node_attr")
        edge_index = inputs.get("edge_index")
        edge_type = inputs.get("edge_type")
        edge_dire_type = inputs.get("edge_dire_type")
        node_attr = self.act(node_attr)
        node_attr = self.enc_to_dec(node_attr)
        node_attr = F.dropout(node_attr, self.dropout, training=self.training)
        out = self.conv(node_attr=node_attr, edge_index=edge_index, edge_type=edge_type, edge_dire_type=edge_dire_type)
        return out
