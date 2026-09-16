# miniGPT side project

## tree template:
```
mini-gpt/
├── .gitignore
├── README.md
├── requirements.txt
├── config.py                 # Dataclass containing hyperparameters (d_model, n_layers, etc.)
│
├── data/
│   ├── prepare_data.py       # Downloads raw text, trains BPE tokenizer, saves token IDs
│   └── dataset.py            # PyTorch Dataset / DataLoader with token shifting (x, y)
│
├── model/
│   ├── __init__.py
│   ├── norm.py               # RMSNorm implementation
│   ├── attention.py          # Causal Multi-Head Attention (Q, K, V + mask)
│   ├── mlp.py                # Feed-Forward network (GELU or SwiGLU)
│   ├── block.py              # Single Transformer Decoder block (Norm -> Attn -> Norm -> MLP)
│   └── gpt.py                # Full Transformer model assembly + weight tying
│
├── engine/
│   ├── __init__.py
│   ├── trainer.py            # Training step, validation loop, gradient clipping, AMP
│   └── sampler.py            # Generation logic (temperature, top-k, top-p nucleus)
│
├── tests/
│   ├── test_shapes.py        # Verifies tensor dimensions for attention, MLP, and full model
│   └── test_overfit.py       # Single-batch sanity test (asserts loss < 0.1)
│
├── train.py                  # Entrypoint: parses args, initializes model, runs trainer
└── generate.py               # Entrypoint: loads checkpoint, prompts model via CLI
```