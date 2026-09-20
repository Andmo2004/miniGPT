# miniGPT

<div align="center">

![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)
![Tests](https://img.shields.io/badge/Tests-21%20Passed-brightgreen?style=for-the-badge&logo=pytest&logoColor=white)
![Hardware](https://img.shields.io/badge/Hardware-CUDA%20%7C%20Apple%20MPS%20%7C%20CPU-blueviolet?style=for-the-badge)

**A clean, modular, and educational implementation of an autoregressive Transformer decoder (GPT) written in pure PyTorch from scratch.**

*Engineered with modern architecture choices (RMSNorm, SwiGLU, Weight Tying), mixed precision training (AMP), advanced sampling strategies (Top-k, Top-p Nucleus), and automated cloud training integration.*

[Features](#-key-features) • [Architecture](#-architecture--design) • [Quickstart](#-quickstart) • [Training](#-training) • [Inference](#-text-generation--sampling) • [Cloud & Kaggle](#-cloud-training--hugging-face-sync) • [Configuration](#-configuration-reference) • [Tests](#-testing--verification)

</div>

---

## Key Features

- **Built Completely from Scratch**: Every single component—RMSNorm, Causal Multi-Head Self-Attention, Feed-Forward layers (GELU & SwiGLU), residual blocks, and the full Transformer decoder—is implemented in pure PyTorch with clear, educational code.
- **Modern Architecture Choices**:
  - **RMSNorm (Root Mean Square Normalization)**: Applied in a Pre-LayerNorm arrangement for faster execution and numerical stability during deep training.
  - **Modular MLP**: Choose between standard GPT-2 **GELU** (`approximate='tanh'`) or modern LLaMA-style **SwiGLU** (`SiLU(gate) * value`) via a single configuration flag.
  - **Weight Tying**: Input token embedding weights are tied with the final LM head projection matrix, drastically reducing parameter count while improving semantic representations.
  - **Fused Attention**: Single projection for Queries, Keys, and Values ($W_{qkv}$), scaled dot-product attention, and lower-triangular causal masking.
- **Production-Ready Training Loop**:
  - **Decoupled AdamW Optimizer**: Custom parameter grouping that applies weight decay only to 2D+ weight matrices (linear projections) while preserving 1D biases, norm scales, and embedding matrices.
  - **Cosine Learning Rate Scheduler**: Linear warmup followed by smooth cosine annealing decay to 10% of peak learning rate.
  - **Gradient Accumulation**: Enables training with large effective batch sizes on constrained consumer GPUs or Apple Silicon.
  - **Hardware-Aware AMP (Automatic Mixed Precision)**: Automatic detection of hardware capabilities—native `bfloat16` on modern NVIDIA GPUs, `float16` with `GradScaler` on earlier GPUs, and Apple Silicon Metal (`mps`) FP16 autocast.
  - **Resilient Checkpointing**: Automatically tracks best validation loss, saves periodic and final checkpoints, preserves full configuration dictionaries, and supports seamless training resumption (`--resume`).
- **Flexible Decoding & Sampling Pipeline**:
  - Centralized autoregressive text generation supporting **Greedy decoding**, **Temperature scaling**, **Top-k filtering**, and **Top-p (Nucleus) sampling**.
  - Seed-controlled generation for reproducible outputs.
- **Efficient Data Pipelines**:
  - Streaming tokenization and chunking for **TinyStories** (`roneneldan/TinyStories`) and **Tiny Shakespeare** datasets.
  - Compact `uint16` binary NumPy storage (`train.npy`, `val.npy`) with automatic `meta.json` tracking.
  - Built-in smoke testing flag (`--smoke_test`) for rapid local iteration.
- **Cloud & Hugging Face Hub Integration**:
  - Out-of-the-box support for remote execution (Kaggle GPU, Hugging Face Jobs) via `cloud/train_hf.py`, with automatic artifact synchronization to Hugging Face Hub.
- **21 Passing Unit Tests**:
  - Exhaustive test suite covering tensor dimensions, attention causality, weight initialization, single-batch overfitting sanity checks, and checkpoint serialization.

---

## Architecture & Design

`miniGPT` follows the **Pre-LayerNorm Decoder** paradigm popularized by modern Large Language Models:

```text
Input Tokens: (B, T)
      │
      ├──► Token Embedding (vocab_size -> d_model)  ──┐
      └──► Position Embedding (block_size -> d_model) ─┴─► (+) ──► Dropout
                                                                    │
      ┌─────────────────────────────────────────────────────────────┘
      │
      ▼
┌────────────────────────────────────────────────────────────────────────┐
│ TransformerBlock × n_layers                                            │
│                                                                        │
│   x ──┬──► RMSNorm ──► CausalSelfAttention ──► Dropout ──►(+)          │
│       │                                                   │            │
│       └────────────────── Residual Connection ────────────┘            │
│                                                           │            │
│   x ──┬──► RMSNorm ──► MLP (GELU or SwiGLU) ──► Dropout ─►(+)          │
│       │                                                   │            │
│       └────────────────── Residual Connection ────────────┘            │
└────────────────────────────────────────────────────────────────────────┘
      │
      ▼
   RMSNorm (ln_f)
      │
      ▼
   LM Head Linear (d_model -> vocab_size)  [Weights Tied with Token Embedding]
      │
      ▼
   Logits: (B, T, vocab_size) ──► Cross Entropy Loss
```

### Mathematical Formulations

#### 1. RMSNorm (Root Mean Square Layer Normalization)
Unlike standard LayerNorm which computes both mean and variance, RMSNorm normalizes by the root-mean-square alone, reducing computational overhead by ~10–15%:
$$\text{RMS}(x) = \sqrt{\frac{1}{d} \sum_{i=1}^{d} x_i^2 + \epsilon}, \quad \text{RMSNorm}(x) = \frac{x}{\text{RMS}(x)} \odot \gamma$$

#### 2. Causal Multi-Head Self-Attention
Queries ($Q$), Keys ($K$), and Values ($V$) are computed via a single fused projection `c_attn` ($d_{\text{model}} \to 3 \times d_{\text{model}}$), reshaped across $n_{\text{heads}}$, and scaled:
$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{Q K^T}{\sqrt{d_k}} + M\right) V$$
where $M$ is a causal lower-triangular mask ensuring position $t$ only attends to positions $\le t$.

#### 3. Feed-Forward Networks (GELU vs. SwiGLU)
- **Standard GELU (GPT-2)**:
  $$\text{FFN}_{\text{GELU}}(x) = \text{GELU}(x W_{\text{up}} + b_{\text{up}}) W_{\text{down}} + b_{\text{down}}$$
- **SwiGLU (LLaMA)**:
  $$\text{FFN}_{\text{SwiGLU}}(x) = \left( \text{SiLU}(x W_{\text{gate}}) \otimes (x W_{\text{value}}) \right) W_{\text{out}}$$

---

## 📁 Repository Structure

```text
miniGPT/
├── config.py             # Strongly-typed dataclass configuration with validation & serialization
├── train.py              # CLI training entrypoint with full hyperparameter flags
├── generate.py           # CLI inference & sampling entrypoint (greedy, top-k, top-p, temp)
├── requirements.txt      # Core project dependencies
├── pytest.ini            # Pytest configuration
│
├── model/                # Transformer neural network modules
│   ├── __init__.py       # Model exports (GPT, RMSNorm, CausalSelfAttention, MLP, SwiGLU)
│   ├── norm.py           # Custom RMSNorm implementation
│   ├── attention.py      # Causal Multi-Head Attention with fused QKV projection
│   ├── mlp.py            # Position-wise Feed-Forward Networks (GELU and SwiGLU variants)
│   ├── block.py          # Single Transformer Decoder block (Norm -> Attn -> Norm -> MLP)
│   └── gpt.py            # Full model assembly, weight tying & parameter initialization
│
├── engine/               # Training and generation engines
│   ├── __init__.py       # Engine exports (train, generate_text)
│   ├── trainer.py        # Complete training loop, optimizer grouping, cosine LR, AMP, checkpoints
│   └── sampler.py        # Centralized sampling logic (top-k, top-p nucleus, temperature)
│
├── data/                 # Data loading and preprocessing pipelines
│   ├── prepare_data.py   # Corpus downloader (TinyStories & TinyShakespeare) & BPE tokenizer
│   └── dataset.py        # PyTorch Dataset and DataLoader with token shift (x, y)
│
├── cloud/                # Remote execution & cloud workflows
│   └── train_hf.py       # Standalone runner for Kaggle/Cloud GPU + Hugging Face Hub sync
│
├── test/                 # Test suite (21 unit tests)
│   ├── test_shapes.py    # Verifies forward pass tensor dimensions across all submodules
│   ├── test_attention.py # Validates causality, attention masking, determinism, and variable length
│   ├── test_overfit.py   # Single micro-batch sanity test (asserts loss converges < 0.1)
│   ├── test_checkpoint.py# Checkpoint save/load roundtrips and config persistence
│   └── test_config.py    # Config validation, assertions, and dictionary serialization
│
└── runs/                 # Output directory for checkpoints (best_model.pt, final_model.pt)
```

---

## Quickstart

### 1. Clone & Setup Environment

```bash
git clone https://github.com/Andmo2004/miniGPT.git
cd miniGPT

# Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Verify Installation

Run the full unit test suite:

```bash
pytest
```
*All 21 unit tests should pass in less than 2 seconds.*

---

## 📊 Dataset Preparation

The `data/prepare_data.py` script downloads raw corpora, encodes text into tokens using `tiktoken` (GPT-2 BPE tokenizer, vocabulary size 50,257), and stores memory-efficient `uint16` NumPy arrays.

### Options:

1. **TinyStories** (*Recommended for coherent mini-language modeling*):
   ```bash
   # Stream from Hugging Face and prepare default splits (100M train tokens, 1M val tokens)
   python data/prepare_data.py --dataset tinystories

   # Or run a rapid smoke test (200k train tokens, 20k val tokens)
   python data/prepare_data.py --dataset tinystories --smoke_test
   ```

2. **Tiny Shakespeare** (*Classic character/dialogue benchmark*):
   ```bash
   python data/prepare_data.py --dataset tinyshakespeare
   ```

Once completed, the dataset directory will contain:
- `train.npy`: Encoded training tokens.
- `val.npy`: Encoded validation tokens.
- `meta.json`: Dataset metadata including vocabulary size and split lengths.

---

## Training

### Basic Training Run
To launch training with default hyperparameters (6 layers, 6 heads, $d_{\text{model}}=384$, batch size 32):

```bash
python train.py
```

### Modern Architecture & Performance Tuning
Switch to modern **SwiGLU** activations, increase sequence context, enable **AMP**, and accumulate gradients:

```bash
python train.py \
  --mlp_type swiglu \
  --d_model 384 \
  --n_layers 6 \
  --n_heads 6 \
  --block_size 256 \
  --batch_size 32 \
  --grad_accumulation_steps 2 \
  --learning_rate 3e-4 \
  --warmup_iters 1000 \
  --max_iters 20000 \
  --use_amp
```

### Resuming from a Checkpoint
If training is interrupted, you can resume seamlessly from any saved checkpoint:

```bash
python train.py --resume runs/best_model.pt
```

### Training CLI Arguments

| Argument | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `--d_model` | `int` | `384` | Embedding and hidden state dimension |
| `--n_layers` | `int` | `6` | Number of stacked Transformer decoder layers |
| `--n_heads` | `int` | `6` | Number of parallel attention heads ($d_{\text{model}}$ must be divisible by $n_{\text{heads}}$) |
| `--block_size` | `int` | `256` | Maximum context length (sequence window) |
| `--mlp_type` | `str` | `"gelu"` | Feed-Forward activation: `"gelu"` or `"swiglu"` |
| `--batch_size` | `int` | `32` | Micro-batch size per optimization step |
| `--grad_accumulation_steps` | `int` | `2` | Number of micro-batches to accumulate before optimizer step |
| `--learning_rate` | `float` | `3e-4` | Peak learning rate for AdamW |
| `--warmup_iters` | `int` | `1000` | Number of warmup iterations with linear LR scaling |
| `--max_iters` | `int` | `20000` | Total training iterations |
| `--eval_interval` | `int` | `250` | Interval (in iterations) to compute validation loss |
| `--save_interval` | `int` | `1000` | Interval (in iterations) to save periodic checkpoints |
| `--log_interval` | `int` | `100` | Interval (in iterations) to print loss and token throughput |
| `--use_amp` | `flag` | `False` | Enable Automatic Mixed Precision (`bfloat16`/`float16`) |
| `--data_dir` | `str` | `"data"` | Path to directory containing tokenized datasets |
| `--out_dir` | `str` | `"runs"` | Directory to store checkpoints and training logs |
| `--resume` | `str` | `None` | Path to checkpoint `.pt` to resume training from |
| `--seed` | `int` | `42` | Random seed for reproducibility |

---

## Text Generation & Sampling

The generation pipeline in `generate.py` loads model checkpoints and generates text autoregressively using configurable sampling strategies.

### Examples

#### 1. Creative Nucleus Sampling (Top-p + Temperature)
```bash
python generate.py \
  --prompt "Once upon a time in a deep forest" \
  --temperature 0.8 \
  --top_p 0.9 \
  --max_tokens 150
```

#### 2. Top-k Constrained Sampling
```bash
python generate.py \
  --prompt "One sunny morning, Lily found" \
  --temperature 0.7 \
  --top_k 40 \
  --max_tokens 200
```

#### 3. Deterministic Greedy Decoding (Argmax)
```bash
python generate.py \
  --prompt "The brave knight took his sword and" \
  --temperature 0.0 \
  --max_tokens 100
```

#### 4. Reproducible Generation with Seed
```bash
python generate.py \
  --prompt "In a quiet village" \
  --temperature 0.8 \
  --seed 1337 \
  --max_tokens 120
```

### Generation CLI Arguments

| Argument | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `--prompt` | `str` | `"ROMEO:"` | Initial input prompt text to condition the model |
| `--max_tokens` | `int` | `200` | Maximum number of new tokens to generate |
| `--temperature` | `float` | `0.8` | Sampling temperature (`0.0` triggers greedy decoding) |
| `--top_k` | `int` | `50` | Keep only top-k highest probability tokens (`<=0` to disable) |
| `--top_p` | `float` | `None` | Nucleus sampling: keep smallest set of tokens with cumulative prob $\ge$ `top_p` |
| `--checkpoint` | `str` | `"runs/best_model.pt"` | Path to checkpoint file (`.pt`) |
| `--seed` | `int` | `None` | Random seed for reproducible generation |

---

## Cloud Training & Hugging Face Sync

`cloud/train_hf.py` provides an all-in-one automation script tailored for cloud environments such as **Kaggle Notebooks** (with free T4 or P100 GPUs) or **Hugging Face Jobs**.

### Features:
- Installs dependencies using `uv` or `pip`.
- Downloads and tokenizes the chosen dataset.
- Trains `miniGPT` with automatic validation and checkpointing.
- Automatically authenticates with Hugging Face using your API token and pushes all checkpoints, `meta.json`, and sample generated outputs directly to your repository (`HF_REPO_ID`).

### Running on Kaggle / Remote Server:

```bash
# Set secrets and run using uv or python
export HF_TOKEN="your_hf_write_token"
export HF_REPO_ID="username/miniGPT-tinystories"

# Execute cloud training script
uv run cloud/train_hf.py
```

### Supported Environment Variables

| Variable | Default | Description |
| :--- | :--- | :--- |
| `HF_TOKEN` | *Required* | Hugging Face user access token with write permissions |
| `HF_REPO_ID` | *Required* | Target repository ID on HF Hub (e.g., `username/miniGPT`) |
| `MAX_ITERS` | `20000` | Total iterations (`<= 10` automatically activates Smoke Test) |
| `SMOKE_TEST` | `0` | Set to `1` for a fast sanity run (200k tokens, 5 iterations) |
| `BATCH_SIZE` | `32` | Micro-batch size |
| `GRAD_ACCUM` | `2` | Gradient accumulation steps |
| `D_MODEL` | `384` | Hidden dimension size |
| `N_LAYERS` | `6` | Transformer decoder layers |
| `N_HEADS` | `6` | Attention heads |
| `MLP_TYPE` | `"gelu"` | Activation function (`"gelu"` or `"swiglu"`) |
| `DATASET` | `"tinystories"` | Corpus to prepare (`"tinystories"` or `"tinyshakespeare"`) |

---

## Configuration Reference

The model and training hyperparameters are defined in [`config.py`](file:///Users/Andres/Desktop/side_projects/miniGPT/config.py) through the `@dataclass GPTConfig`. Every configuration instance is strictly validated in `__post_init__`:

```python
from config import GPTConfig
from model import GPT

# Custom configuration
cfg = GPTConfig(
    vocab_size=50257,
    d_model=512,
    n_heads=8,
    n_layers=8,
    block_size=512,
    mlp_type="swiglu",
    dropout=0.1,
    learning_rate=3e-4,
    batch_size=64,
    grad_accumulation_steps=4,
    use_amp=True,
)

# Instantiate model
model = GPT(cfg)
```

### Config Validation Rules:
- $d_{\text{model}}$ must be strictly divisible by $n_{\text{heads}}$.
- Numerical bounds: $0.0 \le \text{dropout} < 1.0$, $\text{warmup\_iters} \le \text{max\_iters}$, etc.
- Automatic fallback: $d_{\text{ff}}$ defaults to $4 \times d_{\text{model}}$ when omitted.
- Serializes to and from standard Python dictionaries via `to_dict()` and `from_dict()`.

---

## Testing & Verification

The project includes an automated test suite with **21 tests** covering core components:

```bash
pytest -v
```

### Test Breakdown:
- **`test/test_shapes.py`**: Verifies tensor shapes through `RMSNorm`, `CausalSelfAttention`, `MLP`, `SwiGLU`, `TransformerBlock`, and the complete `GPT` model.
- **`test/test_attention.py`**: Asserts attention causality (verifies upper-triangular masking prevents attending to future tokens), validates deterministic seed behavior, and verifies variable-length sequence handling.
- **`test/test_overfit.py`**: Trains `miniGPT` on a single micro-batch to ensure gradients flow correctly and loss drops below $0.1$.
- **`test/test_checkpoint.py`**: Tests full checkpoint saving, state loading, and config persistence across model instances.
- **`test/test_config.py`**: Validates assertion checks, illegal parameter handling, and dictionary serialization roundtrips.

---

## 💡 Hardware Acceleration Notes

- **Apple Silicon (M1/M2/M3/M4)**:
  - Defaults to Metal Performance Shaders (`device='mps'`).
  - Native support for `torch.amp.autocast(device_type='mps', dtype=torch.float16)`. The trainer automatically bypasses `GradScaler` as it is not needed on MPS.
- **NVIDIA GPUs**:
  - Automatically activates CUDA when available.
  - Automatically queries `torch.cuda.is_bf16_supported()`: utilizes native `bfloat16` without loss scaling when supported (Ampere architecture and newer), or falls back to `float16` with dynamic `GradScaler`.
- **CPU**:
  - Full CPU execution fallback for testing and development.

---

## License & Citation

Distributed under the **MIT License**. See `LICENSE` for more details.

If you find this repository educational or helpful in your research and projects, feel free to star the repository!

```bibtex
@misc{andres2024minigpt,
  author = {Andres Moros},
  title = {miniGPT: A Clean, Modular GPT-2/LLaMA-style Transformer in PyTorch},
  year = {2024},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/Andmo2004/miniGPT}}
}
```

### Acknowledgments
- Andrej Karpathy for [nanoGPT](https://github.com/karpathy/nanoGPT) and educational resources on autoregressive transformers.
- Vaswani et al. (2017) – [Attention Is All You Need](https://arxiv.org/abs/1706.03762).
- Touvron et al. (2023) – [LLaMA: Open and Efficient Foundation Language Models](https://arxiv.org/abs/2302.13971) (RMSNorm & SwiGLU).
- Eldan & Li (2023) – [TinyStories](https://arxiv.org/abs/2305.07759).