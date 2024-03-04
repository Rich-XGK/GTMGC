import torch
import torch.nn as nn
from typing import Union, Sequence


class Residual(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, X: torch.Tensor, Y: torch.Tensor):
        return X + Y


class AddNorm(nn.Module):
    def __init__(self, norm_shape: Union[Sequence[int], int], dropout: float = 0, pre_ln: bool = True) -> None:
        """Residual connection and LayerNorm Module.

        Args:
            - norm_shape (Union[Sequence[int], int]): reference to LayerNorm.
            - dropout (float, optional): dropout rate. Defaults to 0.
            - pre_ln (bool, optional): whether to apply LayerNorm before residual connection. Defaults to True.
        """
        super().__init__()
        self.pre_ln = pre_ln
        self.residual = Residual()
        self.layer_norm = nn.LayerNorm(norm_shape)
        self.dropout = nn.Dropout(dropout)

    def forward(self, X: torch.Tensor, Y: torch.Tensor):
        if self.pre_ln:
            # layer norm before residual
            X = self.residual(X, self.layer_norm(self.dropout(Y)))
        else:
            # layer norm after residual
            X = self.residual(X, self.dropout(Y))
            X = self.layer_norm(X)
        return X


class PositionWiseFFN(nn.Module):
    def __init__(self, d_in: int = None, d_hidden: int = None, d_out: int = None, dropout: float = 0) -> None:
        super().__init__()
        assert d_in is not None and d_hidden is not None and d_out is not None, "d_in, d_hidden, d_out must be specified."
        self.ffn01 = nn.Linear(d_in, d_hidden)
        self.ffn02 = nn.Linear(d_hidden, d_out)
        # self.active = nn.ReLU()
        self.active = nn.GELU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, X: torch.Tensor):
        return self.ffn02(self.dropout(self.active(self.ffn01(X))))
