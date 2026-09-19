# model/gpt.py – Full GPT Model Assembly HARD FILE 

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.block import TransformerBlock
from model.norm import RMSNorm

# TODO: Assemble the full GPT model by stacking embeddings, transformer
#       blocks, a final norm, and a language-model head.
#
# ── ARCHITECTURE OVERVIEW ──
#
#   token_ids (B, T)            ← integer input
#       │
#       ├──► Token Embedding     nn.Embedding(vocab_size, d_model)
#       │        -> (B, T, d_model)
#       │
#       ├──► Position Embedding  nn.Embedding(block_size, d_model)
#       │        -> (1, T, d_model)   (same positions for every sample)
#       │
#       ├──► Dropout(drop)       on the sum of the two embeddings
#       │
#       ├──► TransformerBlock × n_layers    (stacked sequentially)
#       │
#       ├──► RMSNorm(d_model)    final norm after all blocks
#       │
#       └──► Linear(d_model, vocab_size, bias=False)   ← "LM head"
#                -> (B, T, vocab_size)   raw logits over vocabulary
#

# CLASS: GPT(nn.Module) 
class GPT(nn.Module):

    def __init__(self, config):
        pass
        # TODO:
        super().__init__()  

        #   1. Store the config: 
        self.config = config

        #   2. Create the embedding layers:
        self.tok_emb = nn.Embedding(config.vocab_size, config.d_model)
        self.pos_emb = nn.Embedding(config.block_size, config.d_model)
        self.drop    = nn.Dropout(config.dropout)
    
        #   3. Create the stack of transformer blocks:
        self.blocks = nn.ModuleList([
            TransformerBlock(config) for _ in range(config.n_layers)
        ])
        #        Use nn.ModuleList, NOT a plain Python list!
        #           A plain list won't register parameters with PyTorch.
    
        #   4. Final layer norm:
        self.ln_f = RMSNorm(config.d_model)
        
        #   6. Language model head (projects d_model -> vocab_size):
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

        #   7. WEIGHT TYING
        self.tok_emb.weight = self.lm_head.weight

        #        This makes the input embedding and the output projection
        #        share the SAME weight matrix.  Why?
        #        - Reduces total parameters by ~vocab_size × d_model
        #        - The embedding learns "what does each token mean" and the
        #          LM head learns "which token should come next" — these are
        #          related tasks, so sharing weights is beneficial.
        #        - Used in GPT-2, LLaMA, and most modern LLMs.
        
        #   8. Initialize weights:
        self.apply(self._init_weights)
        
        #   9. Print total parameter count (nice for sanity checking):
        n_params = sum(p.numel() for p in self.parameters())
        print(f"Model parameters: {n_params / 1e6:.2f}M")

    def _init_weights(self, module):
        """Apply custom weight initialization to all sub-modules."""
        # TODO:
        #   GPT-2 style initialization:
        #

        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
    
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        
        #   WHY std=0.02?
        #   It keeps initial outputs small so the model starts with nearly
        #   uniform predictions, and gradients don't explode in early training.
        
        #   ADVANCED (optional): GPT-2 scales residual projections by
        #   1/sqrt(2 * n_layers) to account for the accumulation through
        #   the residual stream.  You'd apply this to the output projection
        #   of attention (c_proj) and the down projection of MLP (fc_down).


    def forward(self, idx, targets=None):
        """
        Args:
            idx:     (B, T) tensor of token indices
            targets: (B, T) tensor of target token indices (optional, for loss)
        Returns:
            logits: (B, T, vocab_size)
            loss:   scalar cross-entropy loss (or None if targets not provided)
        """
        
        # TODO:
        #   1. Get device and sequence length:
        device = idx.device
        B, T = idx.size()
        assert T <= self.config.block_size, "Sequence too long"
        
        #   2. Create position indices:
        pos = torch.arange(0, T, dtype=torch.long, device=device)  # (T,)
        
        #   3. Compute embeddings:
        tok_emb = self.tok_emb(idx)       # (B, T, d_model)
        pos_emb = self.pos_emb(pos)       # (T, d_model) -> broadcasts to (B, T, d_model)
        x = self.drop(tok_emb + pos_emb)
        
        #   4. Pass through all transformer blocks:
        for block in self.blocks:
            x = block(x)
        
        #   5. Final norm:
        x = self.ln_f(x)

        #   6. Project to vocabulary:
        logits = self.lm_head(x)    # (B, T, vocab_size)

        #   7. Compute loss if targets are given:
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),  # (B*T, vocab_size)
                targets.view(-1)                   # (B*T,)
            )
        else:
            loss = None

        return logits, loss

    # NOTE: Generation logic has been centralised in engine/sampler.py
    #       Use engine.generate_text() for all text generation needs.
    #       This avoids duplicating sampling strategies (top-k, top-p,
    #       greedy, temperature, seed) across multiple locations.

# TESTING
if __name__ == "__main__":
    
    from config import GPTConfig

    config = GPTConfig(vocab_size=100, d_model=64, n_heads=4, n_layers=2, block_size=32)
    model = GPT(config)
    idx = torch.randint(0, 100, (2, 16))
    logits, loss = model(idx, targets=idx)
    assert logits.shape == (2, 16, 100)
    assert loss is not None
