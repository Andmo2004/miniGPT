# model/norm.py – RMSNorm (Root Mean Square Layer Normalization)

import torch
import torch.nn as nn

# TODO: Implement RMSNorm as a drop-in replacement for nn.LayerNorm.
#
# ── Why RMSNorm instead of LayerNorm? ──
#   LayerNorm normalises by subtracting the mean AND dividing by std.
#   RMSNorm skips the mean-centering step and only divides by the
#   root-mean-square of the activations.  This is ~10-15 % faster
#   and empirically works just as well for Transformers.
#
# ── Math ──
#   Given input x of shape (..., d_model):
#
#     rms(x) = sqrt( (1/d_model) * sum(x_i^2) + eps )
#     output = (x / rms(x)) * gamma
#
#   where `gamma` is a learnable scale parameter of shape (d_model,)
#   initialized to ones, and `eps` is a small constant (1e-6) for
#   numerical stability.
#
# ── Class: RMSNorm(nn.Module) ──

class RMSNorm(nn.Module):
    """ RMSNorm as a drop-in replacement for nn.LayerNorm """

    def __init__(self, d_model: int, eps: float = 1e-6):
        # TODO:
        #   1. Call super().__init__()
        #   2. Store `eps` as an attribute.
        #   3. Create `self.gamma` as an nn.Parameter of shape (d_model,)
        #      initialized to torch.ones(d_model).

        super().__init__()
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(d_model)) #Initialized to ones

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # TODO:
        #   1. Compute the mean of x**2 along the last dimension (keepdim=True).
        #   2. Normalise: x_norm = x * torch.rsqrt(mean_sq + self.eps)
        #      (rsqrt = 1/sqrt, avoids separate sqrt + division)
        #   3. Scale: return x_norm * self.gamma
        #
        # Expected input/output shape:  (B, T, d_model) -> (B, T, d_model)
        # B: Batch size, T: Sequence length, d_model: Hidden dimension
        mean_sq = x.pow(2).mean(dim=-1, keepdim=True)
        x_norm = x*torch.rsqrt(mean_sq + self.eps)
        return x_norm * self.gamma 

# ── Testing hint ──
#   After implementing, verify:
#     norm = RMSNorm(384)
#     x = torch.randn(2, 128, 384)
#     out = norm(x)
#     assert out.shape == x.shape

if __name__ == "__main__":
    norm = RMSNorm(384)
    x = torch.randn(2,128,384)
    out = norm(x)
    assert out.shape == x.shape
    
