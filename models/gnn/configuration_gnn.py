"""
    This module contains self-defined configuration class for Gnn.
"""
from transformers import PretrainedConfig
from typing import Literal


class GNNConfig(PretrainedConfig):
    def __init__(
        self,
        num_layers: int = 6,
        d_embed: int = 300,
        dropout: float = 0.0,
        JK: Literal["last", "concat", "max", "sum"] = "last",
        gnn_type: Literal["gine", "gatv2"] = "gine",
        gat_num_heads: int = 4,
        # gap_prob_dropout: float = 0.0,
        **kwargs
    ):
        self.num_layers = num_layers
        self.d_embed = d_embed
        self.dropout = dropout
        self.JK = JK
        self.gnn_type = gnn_type
        self.gat_num_heads = gat_num_heads
        # self.gap_prob_dropout =  gap_prob_dropout
        super().__init__(**kwargs)
