# config.py – Hyperparameter configuration for miniGPT

from torch.distributed.rpc import server_process_global_profiler
from dataclasses import dataclass
import torch

# TODO: Create a dataclass (or plain class) called `GPTConfig` that holds
#       every hyperparameter the rest of the project will reference.
#       Using a single config object keeps magic numbers out of your code
#       and makes experiments reproducible.

# Required fields (with sensible defaults):
#
#   ── Model architecture ──
#   vocab_size  : int   = 50257   # Size of the tokenizer vocabulary.
#                                  # Must match the tokenizer you train in prepare_data.py.
#   d_model     : int   = 384     # Embedding dimension (a.k.a. hidden size).
#                                  # Every token is projected into a vector of this length.
#   n_heads     : int   = 6       # Number of parallel attention heads.
#                                  # d_model must be divisible by n_heads.
#   n_layers    : int   = 6       # Number of stacked Transformer decoder blocks.
#   d_ff        : int   = 4 * d_model  # Inner dimension of the MLP / feed-forward block.
#                                       # Classic GPT uses 4× the embedding dim.
#   block_size  : int   = 256     # Maximum sequence length (context window).
#                                  # Also determines the size of the causal mask.
#   dropout     : float = 0.1     # Dropout probability applied in attention and MLP.
#   bias        : bool  = False   # Whether to use bias in Linear layers and LayerNorm.
#                                  # GPT-2 uses True; modern practice often sets False.
#   ── Training ──
#   batch_size     : int   = 64
#   learning_rate  : float = 3e-4    # Peak LR for AdamW.
#   weight_decay   : float = 0.1     # L2 regularisation coefficient.
#   max_iters      : int   = 5000    # Total training iterations.
#   warmup_iters   : int   = 500     # Linear warm-up steps before cosine decay.
#   grad_clip      : float = 1.0     # Max gradient norm (0.0 = disabled).
#   eval_interval  : int   = 250     # How often (in iters) to run validation.
#   eval_iters     : int   = 50      # Number of batches used to estimate val loss.
#   use_amp        : bool  = False   # Use automatic mixed precision (float16).
#                                     #   MPS NOTE: AMP with GradScaler is NOT fully supported
#                                     #   on MPS as of PyTorch 2.x. Set to False for MPS.
#                                     #   You CAN still use torch.amp.autocast('mps', dtype=torch.float16)
#                                     #   but skip the GradScaler. See trainer.py for details.
#
#   ── System ──
#   device : str = 'mps' if torch.backends.mps.is_available() else \
#                  'cuda' if torch.cuda.is_available() else 'cpu'
#
#                  MPS NOTE: Metal Performance Shaders (Apple Silicon: M1/M2/M3/M4).
#                  On your M4 Mac this will resolve to 'mps'.
#                  MPS accelerates most PyTorch ops on the GPU, but a few ops
#                  may fall back to CPU silently. If you hit errors, test with
#                  device='cpu' first to isolate the issue.
#   seed   : int = 42
#
# Steps:
#   1. `from dataclasses import dataclass`
#   2. Decorate the class with `@dataclass`.
#   3. Define each field with a type hint and default value.
#   4. Add a `__post_init__` method that asserts:
#       - d_model % n_heads == 0  ("d_model must be divisible by n_heads")
#       - All numeric hypers are > 0
#   5. Optionally add a helper `from_dict(cls, d: dict)` classmethod
#      so you can override defaults from a JSON / CLI later.

@dataclass 
class GPTConfig:
    """ MiniGPT Config Class """
    #   ── Model architecture ──
    vocab_size  : int   = 50257
    d_model     : int   = 384
    n_heads     : int   = 6
    n_layers    : int   = 6
    d_ff        : int   = None
    block_size  : int   = 256
    dropout     : float = 0.1
    bias        : bool  = False

    #   ── Training ──
    batch_size     : int   = 64
    learning_rate  : float = 3e-4
    weight_decay   : float = 0.1
    max_iters      : int   = 5000
    warmup_iters   : int   = 500
    grad_clip      : float = 1.0
    eval_interval  : int   = 250
    eval_iters     : int   = 50
    use_amp        : bool  = False

    #   ── System ──
    device : str = 'mps' if torch.backends.mps.is_available() else \
                     'cuda' if torch.cuda.is_available() else 'cpu'
    seed   : int = 42

    def __post_init__(self):
        """ Post-initialisation method for validating hyper-parameters """
        
        # 1. Check if d_model is divisible by n_heads
        assert self.d_model % self.n_heads == 0, f"d_model ({self.d_model}) has to be divisible by n_heads ({self.n_heads})"
        
        # 2. Range validation (ex: dropout between 0 and 1, etc.)
        assert 0.0 <= self.dropout < 1.0, f"dropout ({self.dropout}) has to be between [0,1)"
        assert 0.0 <= self.learning_rate <= 1.0, f"learning_rate ({self.learning_rate}) has to be between [0,1]"
        assert 0.0 <= self.weight_decay <= 1.0, f"weight_decay ({self.weight_decay}) has to be between [0,1]"
        assert 0.0 <= self.warmup_iters <= self.max_iters, f"warmup_iters ({self.warmup_iters}) has to be between [0, max_iters ({self.max_iters})]"
        assert self.grad_clip >= 0.0, f"grad_clip ({self.grad_clip}) has to be greater than or equal to 0"
        assert self.eval_interval > 0, f"eval_interval ({self.eval_interval}) has to be greater than 0"
        assert self.eval_iters > 0, f"eval_iters ({self.eval_iters}) has to be greater than 0"
        assert self.seed > 0, f"seed ({self.seed}) has to be greater than 0"

        # 3. Optional d_ff
        if self.d_ff is None:
            self.d_ff = 4 * self.d_model

        # 4. Check positive values
        assert self.vocab_size > 0
        assert self.d_model > 0
        assert self.n_heads > 0
        assert self.n_layers > 0
        assert self.d_ff > 0
        assert self.block_size > 0
        assert self.batch_size > 0
        assert self.learning_rate > 0
        assert self.weight_decay > 0
        assert self.max_iters > 0
        assert self.warmup_iters > 0
        assert self.grad_clip > 0
        assert self.eval_interval > 0
        assert self.eval_iters > 0
        assert self.seed > 0

# - Helper method for creating config from a dictionary
    @classmethod
    def from_dict(cls, d: dict):
        """ Create a GPTConfig instance from a dictionary """
        return cls(**d)

    def to_dict(self):
        """ Convert a GPTConfig instance to a dictionary """
        return self.__dict__

