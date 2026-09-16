# model/__init__.py – Package exports

# TODO: Import and re-export the main classes so the rest of the project
#       can do `from model import GPT, GPTConfig` cleanly.
#   from model.gpt import GPT
#   (You may also want to export sub-components for testing):

from model.norm import RMSNorm
from model.attention import CausalSelfAttention
from model.mlp import MLP
from model.block import TransformerBlock
