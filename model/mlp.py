# model/mlp.py – Position-wise Feed-Forward Network (MLP)

import torch
import torch.nn as nn
import torch.nn.functional as F

# TODO: Implement the MLP block used inside each Transformer decoder layer.
#
# ── What it does ──
#   After attention has mixed information across token positions, the MLP
#   processes each position *independently* with a two-layer fully-connected
#   network.  It projects up to a wider dimension (d_ff), applies a
#   non-linearity, then projects back down to d_model.
#
# ── Architecture ──
#
#   x (B, T, d_model)
#     │
#     ├─► Linear(d_model → d_ff, bias=config.bias)   # "up" projection
#     │
#     ├─► GELU activation                             # non-linearity
#     │        NOTE: use nn.GELU(approximate='tanh') for the fast variant
#     │        that GPT-2 uses.  Alternatively you can implement SwiGLU
#     │        (used in LLaMA) — that requires a *gated* variant with an
#     │        extra linear projection (see bonus section below).
#     │
#     ├─► Dropout(config.dropout)
#     │
#     └─► Linear(d_ff → d_model, bias=config.bias)   # "down" projection
#
#   output (B, T, d_model)
#
# ── Class: MLP(nn.Module) ──
class MLP(nn.Module):
    def __init__(self, config):
        """ Initialize the MLP block """
        # TODO:
        #   1. Call super().__init__()
        #   2. Create self.fc_up   = nn.Linear(config.d_model, config.d_ff, bias=config.bias)
        #   3. Create self.act     = nn.GELU(approximate='tanh')
        #   4. Create self.fc_down = nn.Linear(config.d_ff, config.d_model, bias=config.bias)
        #   5. Create self.dropout = nn.Dropout(config.dropout)

        super().__init__()
        self.fc_up = nn.Linear(config.d_model, config.d_ff, bias=config.bias)
        self.act = nn.GELU(approximate='tanh')
        self.fc_down = nn.Linear(config.d_ff, config.d_model, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """ Forward pass through the MLP block """
        # TODO:
        #   1. x = self.fc_up(x)       # (B,T,d_model) → (B,T,d_ff)
        #   2. x = self.act(x)         # element-wise GELU
        #   3. x = self.fc_down(x)     # (B,T,d_ff) → (B,T,d_model)
        #   4. x = self.dropout(x)
        #   5. return x

        x = self.fc_up(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.fc_down(x)

        return x

# ── SwiGLU variant (used in LLaMA) ──
#   Instead of GELU, SwiGLU uses a *gating* mechanism:
#     gate   = Linear(d_model → d_ff)
#     value  = Linear(d_model → d_ff)
#     hidden = SiLU(gate) * value       # element-wise gating
#     out    = Linear(d_ff → d_model)(hidden)
#   This needs THREE linear layers.  If you implement this, store all
#   three as attributes and modify forward() accordingly.

class SwiGLU(nn.Module):
    def __init__(self, config):
        """ SwiGLU Initializer """
        super().__init__()
        self.gate = nn.Linear(config.d_model, config.d_ff, bias=config.bias)
        self.value = nn.Linear(config.d_model, config.d_ff, bias=config.bias)
        self.out = nn.Linear(config.d_ff, config.d_model, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """ Forward pass through the MLP block """
        
        # 1. x goes trough two different branches
        g = self.gate(x)    # Gate: (B,T, d_ff)
        v = self.value(x)   # Value: (B,T, d_ff) 

        # 2. We have to apply SiLU to the gate and multiply by value
        hidden = F.silu(g) * v # Hidden: (B,T, d_ff) 

        # 3. Proyect to d_model and apply a dropout
        out = self.out(hidden)
        return self.dropout(out)


# ── Testing hint ──
#   mlp = MLP(config)
#   x = torch.randn(2, 128, config.d_model)
#   out = mlp(x)
#   assert out.shape == (2, 128, config.d_model)

if __name__ == "__main__":
    from config import GPTConfig
    cfg = GPTConfig()
    mlp = MLP(cfg)
    x = torch.randn(2, 128, cfg.d_model)
    out = mlp(x)
    assert out.shape == (2, 128, cfg.d_model)
    print("MLP OK")
