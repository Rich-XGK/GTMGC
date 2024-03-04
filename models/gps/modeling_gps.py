import torch
import torch.nn as nn
import torch_geometric.nn as gnn
import torch.nn.functional as F
from transformers.configuration_utils import PretrainedConfig

from .configuration_gps import GPSConfig
from ..modules import AtomEmbedding, BondEmbedding
from ..modules import GNNConformerPredictionHead
from ..modules import ConformerPredictionOutput

from transformers import PretrainedConfig, PreTrainedModel


class GPSPretrainedModel(PreTrainedModel):
    config_class = GPSConfig
    base_model_prefix = "GPS"
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


class GPS(GPSPretrainedModel):
    def __init__(self, config: PretrainedConfig, *inputs, **kwargs):
        super().__init__(config, *inputs, **kwargs)
        assert config is not None, f"config must be specified to build {self.__class__.__name__}"

        self.d_embed = getattr(config, "d_embed", 256)
        self.d_pe = getattr(config, "d_pe", 64)
        self.pe_length = getattr(config, "pe_length", 20)
        self.num_layer = getattr(config, "num_layer", 6)
        self.num_head = getattr(config, "num_head", 8)
        self.attn_type = getattr(config, "attn_type", "multihead")
        self.attn_dropout = getattr(config, "attn_dropout", 0.0)

        self.node_embedding = AtomEmbedding(atom_embedding_dim=self.d_embed - self.d_pe, attr_reduction="sum")
        self.pe_linear = nn.Linear(self.pe_length, self.d_pe)
        self.pe_norm = nn.BatchNorm1d(self.pe_length)
        self.edge_embedding = BondEmbedding(bond_embedding_dim=self.d_embed, attr_reduction="sum")

        self.convs = nn.ModuleList()
        for _ in range(self.num_layer):
            mlp = nn.Sequential(nn.Linear(self.d_embed, self.d_embed), nn.ReLU(), nn.Linear(self.d_embed, self.d_embed))
            conv = gnn.GPSConv(
                channels=self.d_embed,
                conv=gnn.GINEConv(nn=mlp),
                heads=self.num_head,
                # attn_type=self.attn_type,
                dropout=self.attn_dropout,
            )
            self.convs.append(conv)

    def forward(self, **inputs):
        node_attr, edge_attr, edge_index = inputs["node_attr"], inputs["edge_attr"], inputs["edge_index"]
        pe, batch = inputs["pe"], inputs["batch"]
        x_pe = self.pe_norm(pe)
        node_rep, pe_rep = self.node_embedding(node_attr), self.pe_linear(x_pe)
        node_rep = torch.cat((node_rep, pe_rep), dim=-1)
        edge_rep = self.edge_embedding(edge_attr)
        for conv in self.convs:
            node_rep = conv(x=node_rep, edge_attr=edge_rep, edge_index=edge_index, batch=batch)
        return node_rep


class GPSForConformerPrediction(GPSPretrainedModel):
    def __init__(self, config: PretrainedConfig, *inputs, **kwargs):
        super().__init__(config, *inputs, **kwargs)
        self.gps = GPS(config)
        self.conformer_head = GNNConformerPredictionHead(hidden_X_dim=getattr(config, "d_embed", 256))
        self.__init_weights__()

    def forward(self, **inputs):
        conformer = inputs["conformer"]
        batch = inputs["batch"]
        node_rep = self.gps(**inputs)
        out = self.conformer_head(conformer=conformer, hidden_X=node_rep, batch=batch)
        return ConformerPredictionOutput(
            loss=out["loss"], cdist_mae=out["cdist_mae"], cdist_mse=out["cdist_mse"], conformer=out["conformer"], conformer_hat=out["conformer_hat"]
        )
