# model/block.py – Single Transformer Decoder Block

import torch
import torch.nn as nn

from norm import RMSNorm
from attention import CausalSelfAttention
from mlp import MLP, SwiGLU

# TODO: Implement a single Transformer decoder block.
#
# ── Architecture (Pre-Norm style, used in GPT-2 and modern LLMs) ──
#
#   x ──┬──► RMSNorm ──► CausalSelfAttention ──►(+)──┬──► RMSNorm ──► MLP ──►(+)──► output
#       │                                         │   │                         │
#       └──────────── residual connection ─────────┘   └──── residual connection ┘
#
#   Key insight: the residual connections let gradients flow directly from
#   the output all the way back to the input (skip connection), which is
#   critical for training deep networks.
#
#   "Pre-Norm" means we normalise BEFORE the sub-layer (attention or MLP),
#   not after.  This is more stable than post-norm for deep models.
#
# ── Class: TransformerBlock(nn.Module) ──
class TransformerBlock(nn.Module):
    def __init__(self, config):
        # TODO:
        #   1. Call super().__init__()
        #   2. self.norm1 = RMSNorm(config.d_model)       # norm before attention
        #   3. self.attn  = CausalSelfAttention(config)    # attention sub-layer
        #   4. self.norm2 = RMSNorm(config.d_model)       # norm before MLP
        #   5. self.mlp   = MLP(config)                    # MLP sub-layer
        #
        #   You need to import RMSNorm from model.norm,
        #   CausalSelfAttention from model.attention, and MLP from model.mlp.
        
        super().__init__()
        
        self.norm1 = RMSNorm(config.d_model)
        self.attn = CausalSelfAttention(config)
        self.norm2 = RMSNorm(config.d_model)
        
        self.mlp = MLP(config) 
        # self.mlp SwiGLU(config)

    def forward(self, x: torch.Tensor) -> torch.Tensor: 
        # TODO:
        #   Shape: (B, T, d_model) → (B, T, d_model) — dimensions never change.

        x = x + self.attn(self.norm1(x))   # residual around attention
        x = x + self.mlp(self.norm2(x))    # residual around MLP
        
        return x
