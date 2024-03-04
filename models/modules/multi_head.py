"""
    Multi-head attention module.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from . import attention
from . import utils


class MultiHeadAttention(nn.Module):
    def __init__(
        self,
        d_q: int = None,
        d_k: int = None,
        d_v: int = None,
        d_model: int = None,
        n_head: int = 1,
        qkv_bias: bool = False,
        attn_drop: float = 0,
        use_adjacency: bool = False,
        use_distance: bool = False,
    ) -> None:
        super().__init__()
        self.num_heads = n_head
        self.hidden_dims = d_model
        self.attention = attention.MSRSA(num_heads=n_head, dropout=attn_drop, use_adjacency=use_adjacency, use_distance=use_distance)
        assert d_q is not None and d_k is not None and d_v is not None and d_model is not None, "Please specify the dimensions of Q, K, V and d_model"
        assert d_model % n_head == 0, "d_model must be divisible by n_head"
        # q: q_dims, k: k_dims, v: v_dims, d: hidden_dims, h: num_heads, d_i: dims of each head
        self.W_q = nn.Linear(d_q, d_model, bias=qkv_bias)  # (q, h*d_i=d)
        self.W_k = nn.Linear(d_k, d_model, bias=qkv_bias)  # (k, h*d_i=d)
        self.W_v = nn.Linear(d_v, d_model, bias=qkv_bias)  # (v, h*d_i=d)
        self.W_o = nn.Linear(d_model, d_model, bias=qkv_bias)  # (h*d_i=d, d)

    def forward(
        self,
        queries: torch.Tensor,
        keys: torch.Tensor,
        values: torch.Tensor,
        attention_mask: torch.Tensor = None,
        adjacency_matrix: torch.Tensor = None,
        distance_matrix: torch.Tensor = None,
    ):
        """Compute multi-head attention || b: batch_size, l: seq_len, d: hidden_dims, h: num_heads

        Args:
            - queries (torch.Tensor): Query, shape: (b, l, q)
            - keys (torch.Tensor): Key, shape: (b, l, k)
            - values (torch.Tensor): Value, shape: (b, l, v)
            - attention_mask (torch.Tensor, optional): Attention mask, shape: (b, l) or (b, l, l). Defaults to None.
            - adjacency_matrix (torch.Tensor, optional): Adjacency matrix, shape: (b, l, l). Defaults to None.
            - distance_matrix (torch.Tensor, optional): Distance matrix, shape: (b, l, l). Defaults to None.
        Returns:
            torch.Tensor: Output after multi-head attention pooling with shape (b, l, d)
        """
        # b: batch_size, h:num_heads, l: seq_len, d: d_hidden
        b, h, l, d_i = queries.shape[0], self.num_heads, queries.shape[1], self.hidden_dims // self.num_heads
        Q, K, V = self.W_q(queries), self.W_k(keys), self.W_v(values)  # (b, l, h*d_i=d)
        Q, K, V = [M.view(b, l, h, d_i).permute(0, 2, 1, 3) for M in (Q, K, V)]  # (b, h, l, d_i)
        attn_out = self.attention(Q, K, V, attention_mask, adjacency_matrix, distance_matrix)
        out, attn_weight = attn_out["out"], attn_out["attn_weight"]
        out = out.permute(0, 2, 1, 3).contiguous().view(b, l, h * d_i)  # (b, l, h*d_i=d)
        # return self.W_o(out)  # (b, l, d)
        return {
            "out": self.W_o(out),  # (b, l, d)
            "attn_weight": attn_weight,  # (b, l, l) | (b, h, l, l)
        }


if __name__ == "__main__":
    # Test multi-head attention
    b, l, d, h = 2, 4, 4, 2
    q, k, v = torch.randn(b, l, d), torch.randn(b, l, d), torch.randn(b, l, d)
    mask = utils.valid_length_to_mask(torch.tensor([2, 3]), max_len=l)
    attn = MultiHeadAttention(d_q=d, d_k=d, d_v=d, d_model=d, n_head=h, attn_type="SelfAttention", qkv_bias=True)
    out = attn(q, k, v, mask)
    print(out.shape)
