import torch
import torch.nn as nn
from einops import einsum
from .norm_module import RMSNorm
from .positional_embedding import RotaryPositionalEmbeddingAdjacent
from .linear_module import LinearModule
from .activation_module import SwiGLU
from .attention import causal_multihead_self_attention


"""
测试函数的输入
d_model: int,
num_heads: int,
d_ff: int,
max_seq_len: int,
theta: float,
weights: dict[str, Tensor],
in_features: Float[Tensor, "batch sequence_length d_model"]

y = x + MultiHeadSelfAttention(RMSNorm(x))
"""


class TransformerBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        max_seq_len: int,
        theta: float,
        device=None,
    ):
        super().__init__()

        self.Q_proj_weight = nn.Parameter(torch.Tensor(d_model, d_model))
        self.K_proj_weight = nn.Parameter(torch.Tensor(d_model, d_model))
        self.V_proj_weight = nn.Parameter(torch.Tensor(d_model, d_model))
        self.O_proj_weight = nn.Parameter(torch.Tensor(d_model, d_model))
        nn.init.xavier_uniform_(self.Q_proj_weight)
        nn.init.xavier_uniform_(self.K_proj_weight)
        nn.init.xavier_uniform_(self.V_proj_weight)
        nn.init.xavier_uniform_(self.O_proj_weight)

        self.norm1 = RMSNorm(d_model)
        self.norm2 = RMSNorm(d_model)
        self.ffn = nn.Sequential(
            LinearModule(d_model, d_ff),
            SwiGLU(),
            LinearModule(d_ff, d_model),
        )
        self.rope = RotaryPositionalEmbeddingAdjacent(
            theta=theta,
            d_k=d_model // num_heads,
            max_seq_len=max_seq_len,
            device=device,
        )

    def forward(self, x, token_positions):

        return x
