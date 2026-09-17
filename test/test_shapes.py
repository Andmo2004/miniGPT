# test/test_shapes.py – Tensor shape sanity checks

import torch
import pytest
from config import GPTConfig

# Small config for fast testing
cfg = GPTConfig(
    vocab_size=100,
    d_model=64,
    n_heads=4,
    n_layers=2,
    block_size=32,
    dropout=0.0,
)


def test_rmsnorm_shape():
    from model.norm import RMSNorm

    norm = RMSNorm(cfg.d_model)
    x = torch.randn(2, 16, cfg.d_model)
    assert norm(x).shape == (2, 16, cfg.d_model)


def test_mlp_shape():
    from model.mlp import MLP

    mlp = MLP(cfg)
    x = torch.randn(2, 16, cfg.d_model)
    assert mlp(x).shape == (2, 16, cfg.d_model)


def test_attention_shape():
    from model.attention import CausalSelfAttention

    attn = CausalSelfAttention(cfg)
    x = torch.randn(2, 16, cfg.d_model)
    assert attn(x).shape == (2, 16, cfg.d_model)


def test_block_shape():
    from model.block import TransformerBlock

    block = TransformerBlock(cfg)
    x = torch.randn(2, 16, cfg.d_model)
    assert block(x).shape == (2, 16, cfg.d_model)


def test_gpt_logits_shape():
    from model.gpt import GPT

    model = GPT(cfg)
    idx = torch.randint(0, cfg.vocab_size, (2, 16))
    logits, loss = model(idx)
    assert logits.shape == (2, 16, cfg.vocab_size)
    assert loss is None  # no targets provided


def test_gpt_loss_shape():
    from model.gpt import GPT

    model = GPT(cfg)
    idx = torch.randint(0, cfg.vocab_size, (2, 16))
    logits, loss = model(idx, targets=idx)
    assert logits.shape == (2, 16, cfg.vocab_size)
    assert loss.dim() == 0  # scalar loss


def test_generate_shape():
    from model.gpt import GPT

    model = GPT(cfg)
    idx = torch.randint(0, cfg.vocab_size, (1, 4))
    out = model.generate(idx, max_new_tokens=10)
    assert out.shape == (1, 14)  # 4 prompt + 10 generated
