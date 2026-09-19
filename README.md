# miniGPT side project

## tree template:
```
miniGPT/
├── .gitignore
├── README.md
├── requirements.txt
├── config.py                 # Dataclass with all hyperparameters + validation + serialization
│
├── data/
│   ├── prepare_data.py       # Downloads raw text, trains BPE tokenizer, saves token IDs
│   └── dataset.py            # PyTorch Dataset / DataLoader with token shifting (x, y)
│
├── model/
│   ├── __init__.py
│   ├── norm.py               # RMSNorm implementation
│   ├── attention.py          # Causal Multi-Head Attention (Q, K, V + mask)
│   ├── mlp.py                # Feed-Forward network (GELU or SwiGLU, selectable via config)
│   ├── block.py              # Single Transformer Decoder block (Norm -> Attn -> Norm -> MLP)
│   └── gpt.py                # Full Transformer model assembly + weight tying
│
├── engine/
│   ├── __init__.py
│   ├── trainer.py            # Training loop, checkpointing, grad accumulation, AMP (BF16/FP16)
│   └── sampler.py            # Centralised generation (temperature, top-k, top-p, greedy, seed)
│
├── test/
│   ├── test_shapes.py        # Verifies tensor dimensions for attention, MLP, and full model
│   ├── test_overfit.py       # Single-batch sanity test (asserts loss < 0.1)
│   ├── test_attention.py     # Causality, determinism, and variable-length tests
│   └── test_checkpoint.py    # Save/load roundtrip, config serialisation
│
├── runs/                     # Training outputs (checkpoints, best/final models)
│
├── train.py                  # Entrypoint: parses args, initializes model, runs trainer
└── generate.py               # Entrypoint: loads checkpoint + config, prompts model via CLI
```