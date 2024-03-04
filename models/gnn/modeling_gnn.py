import torch
import torch.nn as nn
import torch_geometric.nn as gnn
import torch.nn.functional as F
from transformers.configuration_utils import PretrainedConfig

from .configuration_gnn import GNNConfig

from ..modules import AtomEmbedding, BondEmbedding
from ..modules import GNNConformerPredictionHead
from ..modules import ConformerPredictionOutput

from transformers import PretrainedConfig, PreTrainedModel


class GNNPretrainedModel(PreTrainedModel):
    config_class = GNNConfig
    base_model_prefix = "Gnn"
    is_parallelizable = False
    main_input_name = "node_attr"

    def __init_weights__(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Embedding):
                nn.init.xavier_normal_(m.weight)


class GNN(GNNPretrainedModel):
    def __init__(self, config: PretrainedConfig, *inputs, **kwargs):
        super().__init__(config, *inputs, **kwargs)
        assert config is not None, f"config must be specified to build {self.__class__.__name__}"
        self.d_embed = getattr(config, "d_embed", 300)
        self.gnn_type = getattr(config, "gnn_type", "gine")
        self.num_layer = getattr(config, "num_layers", 6)
        self.dropout = getattr(config, "dropout", 0.0)
        self.JK = getattr(config, "JK", "last")
        self.gat_num_heads = getattr(config, "gat_num_heads", 4)
        assert self.num_layer >= 2, "Number of GNN layers must be greater than 1."

        self.node_embedding = AtomEmbedding(atom_embedding_dim=self.d_embed, attr_reduction="sum")
        self.edge_embedding = BondEmbedding(bond_embedding_dim=self.d_embed, attr_reduction="sum")

        self.gnns = nn.ModuleList()
        self.batch_norms = nn.ModuleList()

        for _ in range(self.num_layer):
            if self.gnn_type == "gine":
                self.gnns.append(
                    gnn.GINEConv(
                        nn=nn.Sequential(nn.Linear(self.d_embed, 2 * self.d_embed), nn.ReLU(), nn.Linear(self.d_embed * 2, self.d_embed)),
                        train_eps=True,
                        edge_dim=self.d_embed,
                    )
                )
            if self.gnn_type == "gatv2":
                self.gnns.append(
                    gnn.GATv2Conv(in_channels=self.d_embed, out_channels=self.d_embed, heads=self.gat_num_heads, edge_dim=self.d_embed, concat=False)
                )
            self.batch_norms.append(nn.BatchNorm1d(self.d_embed))
        self.__init_weights__()

    def forward(self, **inputs):
        node_attr, edge_attr, edge_index = inputs["node_attr"], inputs["edge_attr"], inputs["edge_index"]
        node_rep = self.node_embedding(node_attr)
        edge_rep = self.edge_embedding(edge_attr)
        h_list = [node_rep]
        for L in range(self.num_layer):
            h = self.gnns[L](x=h_list[L], edge_index=edge_index, edge_attr=edge_rep)
            h = self.batch_norms[L](h)
            if L == self.num_layer - 1:
                h = F.dropout(h, self.dropout, training=self.training)
            else:
                h = F.dropout(F.relu(h), self.dropout, training=self.training)
            h_list.append(h)
        if self.JK == "concat":
            node_rep = torch.cat(h_list, dim=1)
        elif self.JK == "last":
            node_rep = h_list[-1]
        elif self.JK == "max":
            h_list = [h.unsqueeze_(0) for h in h_list]
            node_rep = torch.max(torch.cat(h_list, dim=0), dim=0)[0]
        elif self.JK == "sum":
            h_list = [h.unsqueeze_(0) for h in h_list]
            node_rep = torch.sum(torch.cat(h_list, dim=0), dim=0)[0]
        return node_rep


class GNNForConformerPrediction(GNNPretrainedModel):
    def __init__(self, config: PretrainedConfig, *inputs, **kwargs):
        super().__init__(config, *inputs, **kwargs)
        self.gnn = GNN(config)
        self.conformer_head = GNNConformerPredictionHead(hidden_X_dim=getattr(config, "d_embed", 300))
        self.__init_weights__()

    def forward(self, **inputs):
        conformer = inputs["conformer"]
        batch = inputs["batch"]
        node_rep = self.gnn(**inputs)
        out = self.conformer_head(conformer=conformer, hidden_X=node_rep, batch=batch)
        return ConformerPredictionOutput(
            loss=out["loss"], cdist_mae=out["cdist_mae"], cdist_mse=out["cdist_mse"], conformer=out["conformer"], conformer_hat=out["conformer_hat"]
        )
