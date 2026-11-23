import sys

import torch

import generate
from config import GPTConfig
from model import GPT


def test_generation_uses_available_device_instead_of_checkpoint_device(tmp_path, monkeypatch, capsys):
    config = GPTConfig(vocab_size=16, d_model=8, n_heads=2, n_layers=1, block_size=8)
    model = GPT(config)
    checkpoint = tmp_path / "model.pt"
    saved_config = config.to_dict()
    saved_config["device"] = "cuda"
    torch.save({"config": saved_config, "model": model.state_dict()}, checkpoint)

    class Tokenizer:
        def encode(self, text):
            return [1, 2]

        def decode(self, tokens):
            return "generated"

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: False)
    monkeypatch.setattr(generate.tiktoken, "get_encoding", lambda name: Tokenizer())
    monkeypatch.setattr(sys, "argv", ["generate.py", "--checkpoint", str(checkpoint), "--max_tokens", "1"])
    generate.main()

    output = capsys.readouterr().out
    assert "Inference device: cpu" in output
    assert "generated" in output
