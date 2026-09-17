# test/test_overfit.py – Single-batch overfit sanity test

import torch
import pytest
from config import GPTConfig
from model.gpt import GPT


def test_overfit_single_batch():
    # 1. Create a tiny config
    cfg = GPTConfig(
        vocab_size=100,
        d_model=64,
        n_heads=4,
        n_layers=2,
        block_size=32,
        dropout=0.0,  # critical: no stochastic dropout for perfect memorization
    )

    # 2. Create model and optimizer
    model = GPT(cfg)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    # 3. Create ONE random batch
    x = torch.randint(0, cfg.vocab_size, (4, cfg.block_size))
    y = torch.randint(0, cfg.vocab_size, (4, cfg.block_size))

    # 4. Train on that single batch for 200 steps
    for step in range(200):
        logits, loss = model(x, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # 5. Verify loss is very small (converged)
    assert loss.item() < 0.1, f"Loss didn't converge: {loss.item()}"
