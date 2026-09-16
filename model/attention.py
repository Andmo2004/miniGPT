# model/attention.py – Causal Multi-Head Self-Attention   HARDEST FILE 

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
 
# TODO: Implement Causal (Masked) Multi-Head Self-Attention from scratch.
#
# This is the core mechanism of the GPT Transformer.  Study this carefully.
#

# HIGH-LEVEL INTUITION

#
#   For each token in the sequence the model asks:
#     "Which previous tokens should I pay attention to, and how much?"
#
#   It does this by computing three vectors per token:
#     Q (Query)  – "What am I looking for?"
#     K (Key)    – "What do I contain?"
#     V (Value)  – "What information do I give if selected?"
#
#   Attention scores = softmax( (Q @ K^T) / sqrt(d_k) )
#   Output           = scores @ V
#
#   The "causal mask" forces each token to attend ONLY to itself and earlier
#   tokens (the upper triangle of the attention matrix is set to -inf before
#   softmax).  This is what makes the model autoregressive.
#
#   Multi-head means we run `n_heads` parallel attention operations, each
#   operating on a slice of size `d_k = d_model // n_heads`, then
#   concatenate and project the results back to d_model.
#

# CLASS: CausalSelfAttention(nn.Module)
class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        # TODO:
        
        #   1. Call super().__init__()
        super().__init__()

        #   2. Assert config.d_model % config.n_heads == 0
        assert config.d_model % config.n_heads == 0

        #   3. Store useful constants:
        self.n_heads = config.n_heads
        self.d_model = config.d_model
        self.d_k = config.d_model // config.n_heads
        
        #   4. Create the Q, K, V projection.
        #
        #      Single fused projection (more efficient, GPT-2 style):
        #          self.c_attn = nn.Linear(d_model, 3 * d_model, bias=config.bias)
        #          Then in forward() you split the output into Q, K, V.
        #      Three separate projections (easier to understand):
        #          self.W_q = nn.Linear(d_model, d_model, bias=config.bias)
        #          self.W_k = nn.Linear(d_model, d_model, bias=config.bias)
        #          self.W_v = nn.Linear(d_model, d_model, bias=config.bias)

        self.c_attn = nn.Linear(config.d_model, 3 * config.d_model, bias=config.bias)
      
        #   5. Output projection:
        self.c_proj = nn.Linear(config.d_model, config.d_model, bias=config.bias)        
        
        #   6. Dropout layers:
        self.attn_dropout = nn.Dropout(config.Dropout)
        self.resid_dropout = nn.Dropout(config.Dropout)

        #   7. Register the causal mask as a buffer (not a parameter!):
        #        This is a lower-triangular boolean matrix of shape
        #        (1, 1, block_size, block_size) so it broadcasts over
        #        batch and head dimensions.

        self.register_buffer(
            "mask",
            torch.tril(torch.ones(config.block_size, config.block_size)).view(1,1,config.block_size, config.block_size)
        )

        #        register_buffer makes it part of the module state (saved in
        #        checkpoints, moved to GPU with .to(device)) but NOT trained.

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, T, d_model), input embeddings
        Returns:
            (B, T, d_model): contextualised representations
        """
        # TODO – step by step:
        
        # 1. Unpack dimensions
        B, T, C = x.size()      # batch, sequence length, d_model
        
        # 2. Compute Q, K, V
        #    Using fused projection:
        #       q = self.W_q(x)   # (B, T, d_model)
        #       k = self.W_k(x)   # (B, T, d_model)
        #       v = self.W_v(x)   # (B, T, d_model)

        qkv = self.c_attn(x)                     # (B, T, 3*d_model)
        q, k, v = qkv.split(self.d_model, dim=2) # 3 × (B, T, d_model)

        # 3. Reshape into multiple heads
        q = q.view(B, T, self.n_heads, self.d_k).transpose(1, 2)  # (B, n_heads, T, d_k)
        k = k.view(B, T, self.n_heads, self.d_k).transpose(1, 2)  # (B, n_heads, T, d_k)
        v = v.view(B, T, self.n_heads, self.d_k).transpose(1, 2)  # (B, n_heads, T, d_k)

        #    The transpose(1,2) swaps the T and n_heads dimensions so
        #        each head can be processed independently via batched matmul.
        
        # 4. Compute scaled dot-product attention scores
        att = (q @ k.transpose(-2, -1)) / math.sqrt(self.d_k)  # (B, n_heads, T, T)
        #
        #    WHY scale by sqrt(d_k)?
        #    Without scaling, when d_k is large the dot products grow large,
        #    pushing softmax into regions with tiny gradients (vanishing grads).
        
        # 5. Apply the causal mask
        att = att.masked_fill(self.mask[:, :, :T, :T] == 0, float('-inf'))
        
        #    This sets future positions to -inf so softmax gives them 0 weight.
        #    We slice the mask to [:T, :T] because T may be < block_size.
        
        # 6. Softmax + dropout
        att = F.softmax(att, dim=-1)          # (B, n_heads, T, T)  rows sum to 1
        att = self.attn_dropout(att)
        
        # 7. Weighted sum of values
        y = att @ v                           # (B, n_heads, T, d_k)
        
        # 8. Re-assemble heads
        y = y.transpose(1, 2).contiguous().view(B, T, self.d_model)
        
        #    .contiguous() is needed because transpose makes the tensor
        #    non-contiguous in memory, and .view() requires contiguity.
        
        # 9. Output projection + dropout
        y = self.resid_dropout(self.c_proj(y))   # (B, T, d_model)
        return y

# SHAPE CHEAT-SHEET
#
#   Input:   (B, T, d_model)
#   Q, K, V: (B, n_heads, T, d_k)       d_k = d_model // n_heads
#   Scores:  (B, n_heads, T, T)
#   Output:  (B, T, d_model)
#
# COMMON BUGS TO WATCH FOR
#   • Forgetting .contiguous() before .view() after transpose.
#   • Using the wrong mask slice (must be [:T, :T], not full block_size).
#   • Dividing by d_model instead of d_k in the scaling factor.
#   • Applying softmax on the wrong dimension (must be dim=-1).
