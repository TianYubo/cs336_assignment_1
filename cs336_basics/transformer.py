import torch
import torch.nn as nn
from einops import einsum
from .norm_module import RMSNorm
from .attention import MultiHeadAttention
from .activation_module import SwiGLU


class TransformerBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        max_seq_len: int,
        theta: float | None = None,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()
        self.ln1 = RMSNorm(d_model, device=device, dtype=dtype, eps=1e-6)
        self.attn = MultiHeadAttention(
            d_model, num_heads, max_seq_len, theta, device=device, dtype=dtype
        )
        self.ln2 = RMSNorm(d_model, device=device, dtype=dtype)
        self.ffn = SwiGLU(d_model, d_ff)

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor | None = None):
        # Pre-norm architecture
        # x = x + attn(norm1(x))
        # x = x + ffn(norm2(x))
        x = x + self.attn(self.ln1(x), token_positions=token_positions)
        x = x + self.ffn(self.ln2(x))
        return x


class PostNormTransformerBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        max_seq_len: int,
        theta: float | None = None,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()
        self.attn = MultiHeadAttention(
            d_model, num_heads, max_seq_len, theta, device=device, dtype=dtype
        )
        self.ln1 = RMSNorm(d_model, device=device, dtype=dtype, eps=1e-6)
        self.ffn = SwiGLU(d_model, d_ff)
        self.ln2 = RMSNorm(d_model, device=device, dtype=dtype)

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor | None = None):
        # Post-norm architecture
        # x = norm1(x + attn(x))
        # x = norm2(x + ffn(x))
        x = self.ln1(x + self.attn(x, token_positions=token_positions))
        x = self.ln2(x + self.ffn(x))
        return x
