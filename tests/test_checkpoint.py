# test/test_checkpoint.py – Checkpoint save/load roundtrip tests

import os
import tempfile
import torch
import pytest
from config import GPTConfig
from model.gpt import GPT
from engine.trainer import (
    configure_optimizer,
    save_checkpoint,
    load_checkpoint,
)


cfg = GPTConfig(
    vocab_size=100,
    d_model=64,
    n_heads=4,
    n_layers=2,
    block_size=32,
    dropout=0.0,
)


def test_checkpoint_roundtrip():
    """Save a checkpoint and reload it. Model outputs should be identical."""
    model1 = GPT(cfg)
    optimizer1 = configure_optimizer(model1, cfg)

    # Do a dummy forward + backward to put state into optimizer
    x = torch.randint(0, cfg.vocab_size, (2, cfg.block_size))
    logits, loss = model1(x, targets=x)
    loss.backward()
    optimizer1.step()
    optimizer1.zero_grad()

    # Save checkpoint
    with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
        ckpt_path = f.name

    try:
        save_checkpoint(
            model1, optimizer1, None, iter_num=42,
            best_val_loss=1.234, config=cfg,
            filepath=ckpt_path, use_grad_scaler=False,
        )

        # Load into a fresh model
        model2 = GPT(cfg)
        optimizer2 = configure_optimizer(model2, cfg)

        iter_num, best_val_loss, config_dict = load_checkpoint(
            ckpt_path, model2, optimizer2, device="cpu",
        )

        assert iter_num == 42
        assert abs(best_val_loss - 1.234) < 1e-6

        # Config should round-trip correctly
        assert config_dict["d_model"] == cfg.d_model
        assert config_dict["n_layers"] == cfg.n_layers
        assert config_dict["n_heads"] == cfg.n_heads

        # Model outputs should match exactly
        model1.eval()
        model2.eval()
        with torch.no_grad():
            logits1, _ = model1(x)
            logits2, _ = model2(x)

        assert torch.allclose(logits1, logits2, atol=1e-6), \
            "Model outputs differ after checkpoint roundtrip"

    finally:
        os.unlink(ckpt_path)


def test_checkpoint_config_from_dict():
    """Verify that config saved in checkpoint can reconstruct a valid GPTConfig."""
    config_dict = cfg.to_dict()
    restored = GPTConfig.from_dict(config_dict)

    assert restored.d_model == cfg.d_model
    assert restored.n_layers == cfg.n_layers
    assert restored.n_heads == cfg.n_heads
    assert restored.block_size == cfg.block_size
    assert restored.mlp_type == cfg.mlp_type


def test_checkpoint_ignores_extra_keys():
    """from_dict should gracefully ignore keys that don't exist in GPTConfig."""
    config_dict = cfg.to_dict()
    config_dict["some_future_field"] = 999
    config_dict["another_unknown"] = "hello"

    # Should not raise
    restored = GPTConfig.from_dict(config_dict)
    assert restored.d_model == cfg.d_model
