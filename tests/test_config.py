# test/test_config.py – Hyperparameter config and CLI verification tests

import pytest
import subprocess
import sys
from config import GPTConfig


def test_valid_short_run_warmup():
    """max_iters=5, warmup_iters=5 should be valid (smoke test profile)."""
    cfg = GPTConfig(
        vocab_size=100,
        d_model=64,
        n_heads=4,
        n_layers=2,
        block_size=32,
        max_iters=5,
        warmup_iters=5,
    )
    assert cfg.max_iters == 5
    assert cfg.warmup_iters == 5


def test_invalid_warmup_greater_than_max_iters():
    """max_iters=5, warmup_iters=500 should raise AssertionError."""
    with pytest.raises(AssertionError):
        GPTConfig(
            vocab_size=100,
            d_model=64,
            n_heads=4,
            n_layers=2,
            block_size=32,
            max_iters=5,
            warmup_iters=500,
        )


def test_valid_standard_run_warmup():
    """max_iters=5000, warmup_iters=500 should be valid."""
    cfg = GPTConfig(
        vocab_size=100,
        d_model=64,
        n_heads=4,
        n_layers=2,
        block_size=32,
        max_iters=5000,
        warmup_iters=500,
    )
    assert cfg.max_iters == 5000
    assert cfg.warmup_iters == 500


def test_valid_real_training_profile():
    """max_iters=20000, warmup_iters=1000 should be valid (real training profile)."""
    cfg = GPTConfig(
        vocab_size=50257,
        d_model=384,
        n_heads=6,
        n_layers=6,
        block_size=256,
        max_iters=20000,
        warmup_iters=1000,
        batch_size=32,
        grad_accumulation_steps=2,
    )
    assert cfg.max_iters == 20000
    assert cfg.warmup_iters == 1000
    assert cfg.batch_size == 32
    assert cfg.grad_accumulation_steps == 2


def test_train_cli_accepts_warmup_iters():
    """train.py CLI should accept --warmup_iters without error."""
    result = subprocess.run(
        [sys.executable, "train.py", "--help"],
        capture_output=True,
        text=True,
        check=True,
    )
    assert "--warmup_iters" in result.stdout
    assert "--eval_interval" in result.stdout
    assert "--save_interval" in result.stdout


def test_prepare_data_cli_args():
    """data/prepare_data.py CLI should expose --dataset, --max_train_tokens, and --smoke_test."""
    result = subprocess.run(
        [sys.executable, "data/prepare_data.py", "--help"],
        capture_output=True,
        text=True,
        check=True,
    )
    assert "--dataset" in result.stdout
    assert "--max_train_tokens" in result.stdout
    assert "--max_val_tokens" in result.stdout
    assert "--smoke_test" in result.stdout
