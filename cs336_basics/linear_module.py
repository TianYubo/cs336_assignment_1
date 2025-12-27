import torch.nn as nn
import torch
from einops import einsum


class LinearModule(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super(LinearModule, self).__init__()
        self.initialize_weights(in_features, out_features, device, dtype)

    def initialize_weights(
        self,
        in_features: int,
        out_features: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        self.weight = nn.Parameter(
            torch.empty((out_features, in_features), device=device, dtype=dtype)
        )

    def forward(self, x):
        return einsum(
            self.weight,
            x,
            "d_out d_in, ... d_in -> ... d_out",
        )


class EmbeddingModule(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super(EmbeddingModule, self).__init__()
        self.initialize_weights(vocab_size, embedding_dim, device, dtype)

    def initialize_weights(
        self,
        vocab_size: int,
        embedding_dim: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        self.weight = nn.Parameter(
            torch.empty((vocab_size, embedding_dim), device=device, dtype=dtype)
        )

    def forward(self, token_ids):
        return self.weight[token_ids]
