"""
    This module contains self-defined ModelOutput classes based on transformers.modeling_outputs.BaseModelOutput.
"""

import torch

from dataclasses import dataclass
from typing import Optional, Tuple
from transformers.modeling_outputs import BaseModelOutput


@dataclass
class ConformerPredictionOutput(BaseModelOutput):
    loss: Optional[torch.Tensor] = None
    cdist_mae: Optional[torch.Tensor] = None
    cdist_mse: Optional[torch.Tensor] = None
    coord_rmsd: Optional[torch.Tensor] = None
    conformer: Optional[torch.Tensor] = None  # original conformer aligned to origin
    conformer_hat: Optional[torch.Tensor] = None  # predicted conformer aligned to origin


@dataclass
class MoleBERTTokenizerOutPut(BaseModelOutput):
    loss: Optional[torch.Tensor] = None
    vq_loss: Optional[torch.Tensor] = None
    commitment_loss: Optional[torch.Tensor] = None
    quantized_indices: Optional[torch.Tensor] = None
    quantized_embedding: Optional[torch.Tensor] = None


@dataclass
class GraphReConstructionOutPut(BaseModelOutput):
    loss: Optional[torch.Tensor] = None
    reconstruction_loss: Optional[torch.Tensor] = None
    tokenizer_loss: Optional[torch.Tensor] = None
    vq_loss: Optional[torch.Tensor] = None
    commitment_loss: Optional[torch.Tensor] = None
    reconstruction_accuracy: Optional[torch.Tensor] = None
    

@dataclass
class GraphRegressionOutput(BaseModelOutput):
    loss: Optional[torch.Tensor] = None
    mae: Optional[torch.Tensor] = None
    mse: Optional[torch.Tensor] = None
    logits: Optional[torch.Tensor] = None
    labels: Optional[torch.Tensor] = None
