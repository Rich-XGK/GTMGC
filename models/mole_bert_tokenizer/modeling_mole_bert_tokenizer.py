"""
    This module contains the implementation of the Mole-BERT tokenizer
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Tuple
from torch_geometric.nn.models import GIN as GNN

from transformers import PreTrainedModel, PretrainedConfig
from transformers.configuration_utils import PretrainedConfig

from .configuration_mole_bert_tokenizer import MoleBERTTokenizerConfig

from ..modules import GNNEncoder
from ..modules import GraphReConstructionHead
from ..modules import MoleBERTTokenizerOutPut, GraphReConstructionOutPut


class VectorQuantizer(nn.Module):
    """
    - This class implements the Vector Quantizer module based on the Mole-BERT's implementation.
    - [Mole-BERT](https://openreview.net/pdf?id=jevY-DtiZTR)
        uses a graph VQ_VAE to tokenize the atoms in the molecule to context-aware meaningful discrete values.
    """

    def __init__(self, num_embedding: int = None, d_embedding: int = None, commitment_cost: float = 0.25) -> None:
        """Initialize the VectorQuantizer module.

        Args:
            - num_embedding (int, optional): the num of embedding. Defaults to None.
            - d_embedding (int, optional): the dim of embedding. Defaults to None.
            - commitment_cost (float, optional): the commitment cost. Defaults to 0.25.

        Returns:
            - None
        """
        super().__init__()
        assert d_embedding is not None and num_embedding is not None, f"d_embedding and num_embedding must be given in {self.__class__.__name__}"
        self.num_embedding = num_embedding
        self.d_embedding = d_embedding
        self.commitment_cost = commitment_cost
        # initialize codebook
        self.codebook = nn.Embedding(self.num_embedding, self.d_embedding)

    def forward(self, node_type: torch.Tensor, node_representation: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass of the VectorQuantizer module.

        Args:
            - node_type (torch.Tensor): the node type of the original graph with shape (b,).
            - node_representation (torch.Tensor): the node representation of the encoded graph after encoder with shape (b, d_embedding).

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: the loss, the code indices and the quantized embedding.
        """
        quantized_indices = self._get_code_indices(node_type, node_representation)
        quantized_embedding = self._quantize(quantized_indices)
        # compute loss
        # NOTE: Use Tensor.detach() to stop the gradient. There .detach() usages below are very important!
        # VQ loss: move the embeddings towards the encoder's output, update the codebook.
        vq_loss = F.mse_loss(quantized_embedding, node_representation.detach())
        # Commitment loss: encourages the output of the encoder to stay close to the chosen codebook embedding, update the encoder
        commitment_loss = self.commitment_cost * F.mse_loss(node_representation, quantized_embedding.detach())
        # NOTE: straight-through estimator, copy the gradient from the quantized embedding to the encoder's output.
        quantized_embedding = node_representation + (quantized_embedding - node_representation).detach().contiguous()
        return vq_loss, commitment_loss, quantized_indices, quantized_embedding

    def _get_code_indices(self, node_type: torch.Tensor, node_representation: torch.Tensor) -> torch.Tensor:
        """Get the code indices

        Args:
            - node_type (torch.Tensor): the node type of the original graph with shape (b,).
            - node_representation (torch.Tensor): the node representation of the encoded graph after encoder with shape (b, d_embedding).

        Returns:
            torch.Tensor: the code indices with shape (b,).
        """
        c_indexes = node_type == 5
        n_indexes = node_type == 6
        o_indexes = node_type == 7
        others_indexes = ~(c_indexes | n_indexes | o_indexes)
        c_last, n_last, o_last = int(self.num_embedding / 4), int(self.num_embedding / 2), int(self.num_embedding * 3 / 4)
        # compute L2 distance, copy from Mole-BERT
        encoding_indices = torch.ones(node_representation.shape[0], dtype=torch.long, device=node_representation.device)
        # C: the context-aware C atom embeddings are the first 377 embeddings in the codebook
        distances = (
            torch.sum(node_representation[c_indexes] ** 2, dim=1, keepdim=True)
            + torch.sum(self.codebook.weight[0:c_last] ** 2, dim=1)
            - 2.0 * torch.matmul(node_representation[c_indexes], self.codebook.weight[0:c_last].transpose(0, 1))
        )  # shape: (num_c, 377)
        encoding_indices[c_indexes] = torch.argmin(distances, dim=1)
        # N: the context-aware N atom embeddings are the next 55 embeddings in the codebook
        distances = (
            torch.sum(node_representation[n_indexes] ** 2, dim=1, keepdim=True)
            + torch.sum(self.codebook.weight[c_last:n_last] ** 2, dim=1)
            - 2.0 * torch.matmul(node_representation[n_indexes], self.codebook.weight[c_last:n_last].transpose(0, 1))
        )
        encoding_indices[n_indexes] = torch.argmin(distances, dim=1) + c_last
        # O: the context-aware O atom embeddings are the next 54 embeddings in the codebook
        distances = (
            torch.sum(node_representation[o_indexes] ** 2, dim=1, keepdim=True)
            + torch.sum(self.codebook.weight[n_last:o_last] ** 2, dim=1)
            - 2.0 * torch.matmul(node_representation[o_indexes], self.codebook.weight[n_last:o_last].transpose(0, 1))
        )
        encoding_indices[o_indexes] = torch.argmin(distances, dim=1) + n_last
        # others: the context-aware other atom embeddings are the last 22 embeddings in the codebook
        distances = (
            torch.sum(node_representation[others_indexes] ** 2, dim=1, keepdim=True)
            + torch.sum(self.codebook.weight[o_last:] ** 2, dim=1)
            - 2.0 * torch.matmul(node_representation[others_indexes], self.codebook.weight[o_last:].transpose(0, 1))
        )
        encoding_indices[others_indexes] = torch.argmin(distances, dim=1) + o_last

        return encoding_indices

    def _quantize(self, encoding_indices: torch.Tensor) -> torch.Tensor:
        """Quantize the code indices.

        Args:
            - encoding_indices (torch.Tensor): the code indices with shape (b,).

        Returns:
            torch.Tensor: the quantized code indices with shape (b,).
        """
        quantized_embedding = self.codebook(encoding_indices)
        return quantized_embedding


class MoleBERTTokenizerPreTrainedModel(PreTrainedModel):
    """
        An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    config_class = MoleBERTTokenizerConfig
    base_model_prefix = "mole_bert_tokenizer"
    main_input_name = "node_type"

    def __init_weights__(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.Embedding):
                nn.init.xavier_uniform_(m.weight)


class MoleBERTTokenizer(MoleBERTTokenizerPreTrainedModel):
    def __init__(self, config: PretrainedConfig = None, *inputs, **kwargs):
        assert config is not None, "MoleBERTTokenizer config can not be None."
        super().__init__(config, *inputs, **kwargs)
        self.gnn_encoder = GNNEncoder(
            num_layers=getattr(config, "gnn_encoder_num_layers", 5),
            embedding_dim=getattr(config, "gnn_encoder_embedding_dim", 300),
            layer_hidden_dim=getattr(config, "gnn_encoder_layer_hidden_dim", 600),
            jk=getattr(config, "gnn_encoder_jk", "last"),
            dropout=getattr(config, "gnn_encoder_dropout", 0.0),
        )
        self.vector_quantizer = VectorQuantizer(
            num_embedding=getattr(config, "atom_vocab_size", 512),
            d_embedding=getattr(config, "gnn_encoder_embedding_dim", 300),
            commitment_cost=getattr(config, "vq_commitment_cost", 0.25),
        )
        self.__init_weights__()

    def forward(self, **inputs):
        """Forward pass of the model.

        args:
            - inputs (dict): the inputs of the model
        """
        node_type = inputs.get("node_type")
        node_chiral_type = inputs.get("node_chiral_type")
        edge_type = inputs.get("edge_type")
        edge_dire_type = inputs.get("edge_dire_type")
        edge_index = inputs.get("edge_index")
        node_representation = self.gnn_encoder(
            node_type=node_type,
            node_chiral_type=node_chiral_type,
            edge_type=edge_type,
            edge_dire_type=edge_dire_type,
            edge_index=edge_index,
        )
        vq_loss, commitment_loss, quantized_indices, quantized_embedding = self.vector_quantizer(node_type, node_representation)
        return MoleBERTTokenizerOutPut(
            loss=vq_loss + commitment_loss,
            vq_loss=vq_loss.detach(),
            commitment_loss=commitment_loss.detach(),
            quantized_indices=quantized_indices.detach(),
            quantized_embedding=quantized_embedding,
        )


class MoleBERTTokenizerForGraphReconstruct(MoleBERTTokenizerPreTrainedModel):
    def __init__(self, config: PretrainedConfig = None, *inputs, **kwargs):
        assert config is not None, "MoleBERTTokenizer config can not be None."
        super().__init__(config, *inputs, **kwargs)
        self.tokenizer = MoleBERTTokenizer(config)
        self.graph_reconstruct_head = GraphReConstructionHead(
            in_dim=getattr(config, "gnn_encoder_embedding_dim", 300),
            hidden_dim=getattr(config, "graph_reconstruct_hidden_dim", 600),
            dropout=getattr(config, "graph_reconstruct_dropout", 0.0),
            re_build_edge=getattr(config, "re_build_edge", True),
        )
        self.__init_weights__()

    def forward(self, **inputs):
        node_type = inputs.get("node_type")
        node_chiral_type = inputs.get("node_chiral_type")
        edge_type = inputs.get("edge_type")
        edge_dire_type = inputs.get("edge_dire_type")
        edge_index = inputs.get("edge_index")
        tokenizer_output = self.tokenizer(
            node_type=node_type,
            node_chiral_type=node_chiral_type,
            edge_type=edge_type,
            edge_dire_type=edge_dire_type,
            edge_index=edge_index,
        )
        node_representation = tokenizer_output.quantized_embedding
        reconstruction_outputs = self.graph_reconstruct_head(
            node_representation=node_representation,
            node_type=node_type,
            node_chiral_type=node_chiral_type,
            edge_type=edge_type,
            edge_dire_type=edge_dire_type,
            edge_index=edge_index,
        )
        loss = tokenizer_output.loss + reconstruction_outputs["reconstruction_loss"]
        # print(f"loss: {loss:.4f}, acc:{reconstruction_outputs['reconstruction_accuracy']:.4f}")
        return GraphReConstructionOutPut(
            loss=loss,
            reconstruction_loss=reconstruction_outputs["reconstruction_loss"],
            tokenizer_loss=tokenizer_output.loss.detach(),
            vq_loss=tokenizer_output.vq_loss,
            commitment_loss=tokenizer_output.commitment_loss,
            reconstruction_accuracy=reconstruction_outputs["reconstruction_accuracy"],
        )
