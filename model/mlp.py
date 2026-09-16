# model/mlp.py – Position-wise Feed-Forward Network (MLP)
#
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
#
#   def __init__(self, config):
#       # TODO:
#       #   1. Call super().__init__()
#       #   2. Create self.fc_up   = nn.Linear(config.d_model, config.d_ff, bias=config.bias)
#       #   3. Create self.act     = nn.GELU(approximate='tanh')
#       #   4. Create self.fc_down = nn.Linear(config.d_ff, config.d_model, bias=config.bias)
#       #   5. Create self.dropout = nn.Dropout(config.dropout)
#
#   def forward(self, x: torch.Tensor) -> torch.Tensor:
#       # TODO:
#       #   1. x = self.fc_up(x)       # (B,T,d_model) → (B,T,d_ff)
#       #   2. x = self.act(x)         # element-wise GELU
#       #   3. x = self.fc_down(x)     # (B,T,d_ff) → (B,T,d_model)
#       #   4. x = self.dropout(x)
#       #   5. return x
#
# ── BONUS: SwiGLU variant (used in LLaMA) ──
#   Instead of GELU, SwiGLU uses a *gating* mechanism:
#     gate   = Linear(d_model → d_ff)
#     value  = Linear(d_model → d_ff)
#     hidden = SiLU(gate) * value       # element-wise gating
#     out    = Linear(d_ff → d_model)(hidden)
#   This needs THREE linear layers.  If you implement this, store all
#   three as attributes and modify forward() accordingly.
#
# ── Testing hint ──
#   mlp = MLP(config)
#   x = torch.randn(2, 128, config.d_model)
#   out = mlp(x)
#   assert out.shape == (2, 128, config.d_model)
