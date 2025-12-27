import torch
import torch.nn as nn
from einops import einsum


from .linear_module import LinearModule


class SwiGLU(nn.Module):
    def __init__(self, d_model: int, d_ff: int | None = None):
        super().__init__()
        if d_ff is None:
            d_ff = int(8 / 3 * d_model)
        self.w1 = LinearModule(d_model, d_ff)
        self.w2 = LinearModule(d_ff, d_model)
        self.w3 = LinearModule(d_model, d_ff)

    def forward(self, x: torch.Tensor):
        W1x = self.w1(x)
        W3x = self.w3(x)
        swishW1x_W3x = swish(W1x) * W3x
        output = self.w2(swishW1x_W3x)
        return output


def swish(x: torch.Tensor):
    return x * torch.sigmoid(x)
