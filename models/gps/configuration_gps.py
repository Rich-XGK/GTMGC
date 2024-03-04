"""
    This module contains self-defined configuration class for GPS.
"""
from transformers import PretrainedConfig
from typing import Literal


class GPSConfig(PretrainedConfig):
    def __init__(
        self,
        d_embed: int = 256,
        d_pe: int = 64,
        pe_length: int = 20,
        num_layer: int = 6,
        num_head: int = 8,
        attn_type: Literal["multihead", "performer"] = "multihead",
        attn_dropout: float = 0.0,
        **kwargs
    ):
        self.d_embed = d_embed
        self.d_pe = d_pe
        self.pe_length = pe_length
        self.num_layer = num_layer
        self.num_head = num_head
        self.attn_type = attn_type
        self.attn_dropout = attn_dropout
        super().__init__(**kwargs)
