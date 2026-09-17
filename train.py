# train.py – Training entrypoint

import argparse
import os
import json
import torch

from config import GPTConfig
from data.dataset import get_dataloaders
from model import GPT
from engine import train


def main():
    # 1. Base configuration
    config = GPTConfig()

    # 2. CLI arguments to allow easy hyperparameter overrides
    parser = argparse.ArgumentParser(description="Train miniGPT on text corpus")
    parser.add_argument("--batch_size", type=int, default=config.batch_size, help="Batch size")
    parser.add_argument("--learning_rate", type=float, default=config.learning_rate, help="Peak learning rate")
    parser.add_argument("--max_iters", type=int, default=config.max_iters, help="Total training iterations")
    parser.add_argument("--eval_interval", type=int, default=config.eval_interval, help="Validation interval")
    parser.add_argument("--d_model", type=int, default=config.d_model, help="Embedding dimension")
    parser.add_argument("--n_layers", type=int, default=config.n_layers, help="Number of transformer layers")
    parser.add_argument("--n_heads", type=int, default=config.n_heads, help="Number of attention heads")
    parser.add_argument("--block_size", type=int, default=config.block_size, help="Context length")
    parser.add_argument("--data_dir", type=str, default="data", help="Directory containing data files")
    args = parser.parse_args()

    # Apply overrides
    config.batch_size = args.batch_size
    config.learning_rate = args.learning_rate
    config.max_iters = args.max_iters
    config.eval_interval = args.eval_interval
    config.d_model = args.d_model
    config.n_layers = args.n_layers
    config.n_heads = args.n_heads
    config.block_size = args.block_size

    # Check if meta.json exists to ensure vocab_size matches tokenization
    meta_path = os.path.join(args.data_dir, "meta.json")
    if os.path.exists(meta_path):
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
            if "vocab_size" in meta:
                config.vocab_size = meta["vocab_size"]

    # 3. Set random seeds for reproducibility
    torch.manual_seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.seed)

    print("=" * 60)
    print(" miniGPT Training Configuration ")
    print("=" * 60)
    print(f"Device: {config.device}")
    print(f"Vocab size: {config.vocab_size}")
    print(f"Embedding dim (d_model): {config.d_model}")
    print(f"Layers: {config.n_layers}, Heads: {config.n_heads}")
    print(f"Context length (block_size): {config.block_size}")
    print(f"Batch size: {config.batch_size}")
    print(f"Total iterations: {config.max_iters}")
    print(f"Peak learning rate: {config.learning_rate:.2e}")
    print("=" * 60)

    # 4. Load dataset
    print("Loading data...")
    train_loader, val_loader = get_dataloaders(
        data_dir=args.data_dir,
        block_size=config.block_size,
        batch_size=config.batch_size,
    )

    # 5. Initialize model
    print("Initializing model...")
    model = GPT(config)

    # 6. Launch training
    train(model, train_loader, val_loader, config)


if __name__ == "__main__":
    main()
