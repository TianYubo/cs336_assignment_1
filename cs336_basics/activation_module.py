import torch
import torch.nn as nn
from einops import einsum


class SwiGLU(nn.Module):
    def __init__(self, d_model: int, d_ff: int | None = None):
        super().__init__()
        self.initialize_weights(d_model, d_ff)

    def initialize_weights(self, d_model: int, d_ff: int | None):
        if d_ff is None:
            d_ff = 8 / 3 * d_model
        d_ff = int(d_ff)
        self.w1 = nn.Parameter(torch.Tensor(d_ff, d_model))
        self.w2 = nn.Parameter(torch.Tensor(d_model, d_ff))
        self.w3 = nn.Parameter(torch.Tensor(d_ff, d_model))
        # self.b = nn.Parameter(torch.Tensor(d_model))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.w1)
        nn.init.xavier_uniform_(self.w2)
        nn.init.xavier_uniform_(self.w3)

    def forward(self, x: torch.Tensor):
        W1x = einsum(self.w1, x, "d_ff d_model, ... d_model -> ... d_ff")
        W3x = einsum(self.w3, x, "d_ff d_model, ... d_model -> ... d_ff")
        swishW1x_W3x = swish(W1x) * W3x
        output = einsum(
            self.w2,
            swishW1x_W3x,
            "d_model d_ff, ... d_ff -> ... d_model",
        )
        return output


def swish(x: torch.Tensor):
    return x * torch.sigmoid(x)
