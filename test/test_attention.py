# test/test_attention.py – Causality and correctness tests for attention

import torch
import pytest
from config import GPTConfig

cfg = GPTConfig(
    vocab_size=100,
    d_model=64,
    n_heads=4,
    n_layers=2,
    block_size=32,
    dropout=0.0,
)


def test_causal_mask():
    """Verify that changing a future token does NOT affect the output at
    earlier positions. This is the fundamental property of causal attention."""
    from model.gpt import GPT

    model = GPT(cfg)
    model.eval()

    # Create two sequences that differ only at position 5
    x1 = torch.randint(0, cfg.vocab_size, (1, 10))
    x2 = x1.clone()
    x2[0, 5] = (x1[0, 5] + 1) % cfg.vocab_size  # change token at pos 5

    with torch.no_grad():
        logits1, _ = model(x1)
        logits2, _ = model(x2)

    # Positions 0-4 should be IDENTICAL (they can't see position 5)
    assert torch.allclose(logits1[:, :5, :], logits2[:, :5, :], atol=1e-5), \
        "Causal violation: changing token at pos 5 affected outputs at pos 0-4"

    # Position 5 and beyond should be DIFFERENT
    assert not torch.allclose(logits1[:, 5:, :], logits2[:, 5:, :], atol=1e-5), \
        "Tokens at/after the changed position should produce different outputs"


def test_attention_shape():
    """Basic shape test for CausalSelfAttention."""
    from model.attention import CausalSelfAttention

    attn = CausalSelfAttention(cfg)
    x = torch.randn(2, 16, cfg.d_model)
    out = attn(x)
    assert out.shape == (2, 16, cfg.d_model)


def test_attention_deterministic():
    """Same input should produce same output (no randomness in eval mode)."""
    from model.attention import CausalSelfAttention

    attn = CausalSelfAttention(cfg)
    attn.eval()
    x = torch.randn(2, 8, cfg.d_model)

    with torch.no_grad():
        out1 = attn(x)
        out2 = attn(x)

    assert torch.allclose(out1, out2, atol=1e-6), \
        "Attention should be deterministic in eval mode"


def test_attention_variable_length():
    """Attention should handle sequences shorter than block_size."""
    from model.attention import CausalSelfAttention

    attn = CausalSelfAttention(cfg)
    attn.eval()

    # Sequence much shorter than block_size=32
    x_short = torch.randn(1, 3, cfg.d_model)
    out = attn(x_short)
    assert out.shape == (1, 3, cfg.d_model)

    # Single token
    x_single = torch.randn(1, 1, cfg.d_model)
    out = attn(x_single)
    assert out.shape == (1, 1, cfg.d_model)
