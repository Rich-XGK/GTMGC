"""
    This module contains self-defined GraphBert model.
"""

import torch
import torch.nn as nn
from .configuration_gtmgc import GTMGCConfig

from ..modules import ConformerPredictionHead, GraphRegressionHead
from ..modules import ConformerPredictionOutput, GraphRegressionOutput
from ..modules import MultiHeadAttention, AddNorm, PositionWiseFFN, Residual
from ..modules.utils import make_cdist_mask, compute_distance_residual_bias
from ..modules import AtomEmbedding, NodeEmbedding  # for Ogb embedding ablation
from transformers import PretrainedConfig, PreTrainedModel


class GTMGCBlock(nn.Module):
    def __init__(self, config: PretrainedConfig = None, encoder: bool = True) -> None:
        super().__init__()
        self.config = config
        self.encoder = encoder
        assert config is not None, f"config must be specified to build {self.__class__.__name__}"
        self.use_A_in_attn = getattr(config, "encoder_use_A_in_attn", False) if encoder else getattr(config, "decoder_use_A_in_attn", False)
        self.use_D_in_attn = getattr(config, "encoder_use_D_in_attn", False) if encoder else getattr(config, "decoder_use_D_in_attn", False)
        self.multi_attention = MultiHeadAttention(
            d_q=getattr(config, "d_q", 256),
            d_k=getattr(config, "d_k", 256),
            d_v=getattr(config, "d_v", 256),
            d_model=getattr(config, "d_model", 256),
            n_head=getattr(config, "n_head", 8),
            qkv_bias=getattr(config, "qkv_bias", True),
            attn_drop=getattr(config, "attn_drop", 0.1),
            use_adjacency=self.use_A_in_attn,
            use_distance=self.use_D_in_attn,
        )
        self.add_norm01 = AddNorm(norm_shape=getattr(config, "d_model", 256), dropout=getattr(config, "norm_drop", 0.1), pre_ln=getattr(config, "pre_ln", True))
        self.position_wise_ffn = PositionWiseFFN(
            d_in=getattr(config, "d_model", 256),
            d_hidden=getattr(config, "d_ffn", 1024),
            d_out=getattr(config, "d_model", 256),
            dropout=getattr(config, "ffn_drop", 0.1),
        )
        self.add_norm02 = AddNorm(norm_shape=getattr(config, "d_model", 256), dropout=getattr(config, "norm_drop", 0.1), pre_ln=getattr(config, "pre_ln", True))

    def forward(self, **inputs):
        # Using kwargs to make getting inputs more flexible
        X, M = inputs.get("node_embedding"), inputs.get("node_mask")
        A = inputs.get("adjacency") if self.use_A_in_attn else None
        D = inputs.get("distance") if self.use_D_in_attn else None
        attn_out = self.multi_attention(X, X, X, attention_mask=M, adjacency_matrix=A, distance_matrix=D)
        Y, attn_weight = attn_out["out"], attn_out["attn_weight"]
        X = self.add_norm01(X, Y)
        Y = self.position_wise_ffn(X)
        X = self.add_norm02(X, Y)
        return {
            "out": X,
            "attn_weight": attn_weight,
        }


class GTMGCPretrainedModel(PreTrainedModel):
    """
        An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    config_class = GTMGCConfig
    base_model_prefix = "Conformer"
    is_parallelizable = False
    # NOTE: version 1.0.0
    # main_input_name = "node_attr"
    # NOTE: version 2.0.0
    main_input_name = "node_input_ids"

    def __init_weights__(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Embedding):
                nn.init.xavier_normal_(m.weight)


class GTMGCEncoder(GTMGCPretrainedModel):
    def __init__(self, config: PretrainedConfig = None, *inputs, **kwargs):
        super().__init__(config, *inputs, **kwargs)
        assert config is not None, f"config must be specified to build {self.__class__.__name__}"
        self.embed_style = getattr(config, "embed_style", "atom_tokenized_ids")
        if self.embed_style == "atom_tokenized_ids":
            self.node_embedding = nn.Embedding(getattr(config, "atom_vocab_size", 513), getattr(config, "d_embed", 256), padding_idx=0)
        elif self.embed_style == "atom_type_ids":
            self.node_embedding = nn.Embedding(getattr(config, "atom_vocab_size", 119), getattr(config, "d_embed", 256), padding_idx=0)
        elif self.embed_style == "ogb":
            self.ogb_node_embedding = NodeEmbedding(atom_embedding_dim=getattr(config, "d_embed", 256), attr_reduction="sum")  # for Ogb embedding ablation
        self.encoder_blocks = nn.ModuleList([GTMGCBlock(config, encoder=True) for _ in range(getattr(config, "n_encode_layers", 12))])
        self.__init_weights__()

    def forward(self, **inputs):
        # Using kwargs to make getting inputs more flexible
        if self.embed_style == "atom_tokenized_ids":
            node_input_ids = inputs.get("node_input_ids")
            node_embedding = self.node_embedding(node_input_ids)
        elif self.embed_style == "atom_type_ids":
            node_input_ids = inputs.get("node_type")  # for node type id
            node_embedding = self.node_embedding(node_input_ids)
        elif self.embed_style == "ogb":
            node_embedding = self.ogb_node_embedding(inputs["node_attr"])  # for Ogb embedding ablation
        # laplacian positional embedding
        lap = inputs.get("lap_eigenvectors")
        node_embedding[:, :, : lap.shape[-1]] = node_embedding[:, :, : lap.shape[-1]] + lap
        inputs["node_embedding"] = node_embedding
        if self.config.encoder_use_D_in_attn:
            C = inputs.get("conformer")
            D, D_M = torch.cdist(C, C), make_cdist_mask(inputs.get("node_mask"))
            D = compute_distance_residual_bias(cdist=D, cdist_mask=D_M)  # masked and row-max subtracted distance matrix
            # D = D * D_M  # for ablation study
            inputs["distance"] = D
        attn_weight_dict = {}
        for i, encoder_block in enumerate(self.encoder_blocks):
            block_out = encoder_block(**inputs)
            node_embedding, attn_weight = block_out["out"], block_out["attn_weight"]
            inputs["node_embedding"] = node_embedding
            attn_weight_dict[f"encoder_block_{i}"] = attn_weight
        return {"node_embedding": node_embedding, "attn_weight_dict": attn_weight_dict}


class GTMGCDecoder(GTMGCPretrainedModel):
    def __init__(self, config: PretrainedConfig = None, *inputs, **kwargs):
        super().__init__(config, *inputs, **kwargs)
        assert config is not None, f"config must be specified to build {self.__class__.__name__}"
        self.decoder_blocks = nn.ModuleList([GTMGCBlock(config, encoder=False) for _ in range(getattr(config, "n_decode_layers", 6))])
        self.__init_weights__()

    def forward(self, **inputs):
        node_embedding, lap = inputs.get("node_embedding"), inputs.get("lap_eigenvectors")
        # laplacian positional embedding
        node_embedding[:, :, : lap.shape[-1]] = node_embedding[:, :, : lap.shape[-1]] + lap
        inputs["node_embedding"] = node_embedding
        attn_weight_dict = {}
        for i, decoder_block in enumerate(self.decoder_blocks):
            block_out = decoder_block(**inputs)
            node_embedding, attn_weight = block_out["out"], block_out["attn_weight"]
            inputs["node_embedding"] = node_embedding
            attn_weight_dict[f"decoder_block_{i}"] = attn_weight
        return {"node_embedding": node_embedding, "attn_weight_dict": attn_weight_dict}


class GTMGCForConformerPrediction(GTMGCPretrainedModel):
    def __init__(self, config: PretrainedConfig, *inputs, **kwargs):
        super().__init__(config, *inputs, **kwargs)
        self.encoder = GTMGCEncoder(config)
        self.decoder = GTMGCDecoder(config)
        self.conformer_head = ConformerPredictionHead(hidden_X_dim=getattr(config, "d_model", 256))
        self.__init_weights__()

    def forward(self, **inputs):
        conformer, node_mask = inputs.get("conformer"), inputs.get("node_mask")
        # encoder forward
        encoder_out = self.encoder(**inputs)
        node_embedding, encoder_attn_weight_dict = encoder_out["node_embedding"], encoder_out["attn_weight_dict"]
        cache_out = self.conformer_head(conformer=conformer, hidden_X=node_embedding, padding_mask=node_mask, compute_loss=True)
        loss_cache, conformer_cache = cache_out["loss"], cache_out["conformer_hat"]
        D_cache, D_M = torch.cdist(conformer_cache, conformer_cache), make_cdist_mask(node_mask)
        D_cache = compute_distance_residual_bias(cdist=D_cache, cdist_mask=D_M)  # masked and row-max subtracted distance matrix
        # D_cache = D_cache * D_M  # for ablation study
        inputs["node_embedding"] = node_embedding
        inputs["distance"] = D_cache
        decoder_out = self.decoder(**inputs)
        node_embedding, decoder_attn_weight_dict = decoder_out["node_embedding"], decoder_out["attn_weight_dict"]
        outputs = self.conformer_head(conformer=conformer, hidden_X=node_embedding, padding_mask=node_mask, compute_loss=True)  # final prediction
        return ConformerPredictionOutput(
            loss=(outputs["loss"] + loss_cache) / 2,
            # loss=outputs["loss"],
            cdist_mae=outputs["cdist_mae"],
            cdist_mse=outputs["cdist_mse"],
            coord_rmsd=outputs["coord_rmsd"],
            conformer=outputs["conformer"],
            conformer_hat=outputs["conformer_hat"],
            # attentions={**encoder_attn_weight_dict, **decoder_attn_weight_dict}
        )


class GTMGCForGraphRegression(GTMGCPretrainedModel):
    def __init__(self, config: PretrainedConfig, *inputs, **kwargs):
        super().__init__(config, *inputs, **kwargs)
        self.encoder = GTMGCEncoder(config)
        self.decoder = GraphRegressionHead(hidden_X_dim=getattr(config, "d_model", 256))
        self.__init_weights__()

    def forward(self, **inputs):
        encoder_out = self.encoder(**inputs)
        graph_rep = encoder_out["node_embedding"].mean(dim=1)
        decoder_outputs = self.decoder(hidden_X=graph_rep, labels=inputs.get("labels"))
        return GraphRegressionOutput(
            loss=decoder_outputs["loss"],
            mae=decoder_outputs["mae"],
            mse=decoder_outputs["mse"],
            logits=decoder_outputs["logits"],
            labels=decoder_outputs["labels"],
        )
