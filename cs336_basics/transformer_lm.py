import torch
import torch.nn as nn
from .transformer import TransformerBlock
from .norm_module import RMSNorm
from .linear_module import LinearModule, EmbeddingModule


class Transformer_LM(nn.Module):

    def __init__(
        self,
        vocab_size: int,
        context_length: int,
        d_model: int,
        num_layers: int,
        num_heads: int,
        d_ff: int,
        rope_theta: float,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()
        self.context_length = context_length
        self.token_embeddings = EmbeddingModule(
            vocab_size, d_model, device=device, dtype=dtype
        )
        self.layers = nn.ModuleList(
            [
                TransformerBlock(
                    d_model,
                    num_heads,
                    d_ff,
                    context_length,
                    rope_theta,
                    device=device,
                    dtype=dtype,
                )
                for _ in range(num_layers)
            ]
        )
        self.ln_final = RMSNorm(d_model, device=device, dtype=dtype)
        self.lm_head = LinearModule(d_model, vocab_size, device=device, dtype=dtype)

    def forward(self, input_ids, token_positions=None):
        seq_len = input_ids.size(1)
        if seq_len > self.context_length:
            raise ValueError(
                f"Input sequence length {seq_len} exceeds context length {self.context_length}"
            )
        x = self.token_embeddings(input_ids)
        for layer in self.layers:
            x = layer(x, token_positions=token_positions)
        x = self.ln_final(x)
        logits = self.lm_head(x)
        return logits
