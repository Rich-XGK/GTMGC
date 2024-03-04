"""
    This module contains self-defined configuration class for VQ_VAE.
"""

from typing import Literal
from transformers import PretrainedConfig


class MoleBERTTokenizerConfig(PretrainedConfig):
    def __init__(
        self,
        gnn_encoder_num_layers: int = 5,
        gnn_encoder_embedding_dim: int = 300,
        gnn_encoder_layer_hidden_dim: int = 600,
        gnn_encoder_jk: Literal["concat", "last", "max", "sum"] = "last",
        gnn_encoder_dropout: float = 0.0,
        atom_vocab_size: int = 512,
        vq_commitment_cost: float = 0.25,
        graph_reconstruct_hidden_dim: int = 600,
        graph_reconstruct_dropout: float = 0.0,
        re_build_edge: bool = True,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.gnn_encoder_num_layers = gnn_encoder_num_layers
        self.gnn_encoder_embedding_dim = gnn_encoder_embedding_dim
        self.gnn_encoder_layer_hidden_dim = gnn_encoder_layer_hidden_dim
        self.gnn_encoder_jk = gnn_encoder_jk
        self.gnn_encoder_dropout = gnn_encoder_dropout
        self.atom_vocab_size = atom_vocab_size
        self.vq_commitment_cost = vq_commitment_cost
        self.graph_reconstruct_hidden_dim = graph_reconstruct_hidden_dim
        self.graph_reconstruct_dropout = graph_reconstruct_dropout
        self.re_build_edge = re_build_edge
