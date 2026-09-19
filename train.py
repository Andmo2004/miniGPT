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
    # 1. CLI arguments — all hyperparameters can be overridden
    parser = argparse.ArgumentParser(description="Train miniGPT on text corpus")
    
    # Model architecture
    parser.add_argument("--d_model", type=int, default=384, help="Embedding dimension")
    parser.add_argument("--n_layers", type=int, default=6, help="Number of transformer layers")
    parser.add_argument("--n_heads", type=int, default=6, help="Number of attention heads")
    parser.add_argument("--block_size", type=int, default=256, help="Context length")
    parser.add_argument("--mlp_type", type=str, default="gelu", choices=["gelu", "swiglu"],
                        help="MLP activation type")
    
    # Training
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size")
    parser.add_argument("--learning_rate", type=float, default=3e-4, help="Peak learning rate")
    parser.add_argument("--max_iters", type=int, default=5000, help="Total training iterations")
    parser.add_argument("--eval_interval", type=int, default=250, help="Validation interval")
    parser.add_argument("--save_interval", type=int, default=1000, help="Checkpoint save interval")
    parser.add_argument("--log_interval", type=int, default=100, help="Logging interval")
    parser.add_argument("--grad_accumulation_steps", type=int, default=1,
                        help="Gradient accumulation steps")
    parser.add_argument("--use_amp", action="store_true", help="Enable AMP (auto BF16/FP16)")
    
    # System
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--data_dir", type=str, default="data", help="Directory containing data files")
    parser.add_argument("--out_dir", type=str, default="runs",
                        help="Output directory for checkpoints and logs")
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to checkpoint to resume training from")
    
    args = parser.parse_args()

    # 2. Build config directly from CLI args (triggers __post_init__ validation)
    config_kwargs = {
        "d_model": args.d_model,
        "n_layers": args.n_layers,
        "n_heads": args.n_heads,
        "block_size": args.block_size,
        "mlp_type": args.mlp_type,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "max_iters": args.max_iters,
        "eval_interval": args.eval_interval,
        "save_interval": args.save_interval,
        "log_interval": args.log_interval,
        "grad_accumulation_steps": args.grad_accumulation_steps,
        "use_amp": args.use_amp,
        "seed": args.seed,
        "out_dir": args.out_dir,
    }

    # Check if meta.json exists to ensure vocab_size matches tokenization
    meta_path = os.path.join(args.data_dir, "meta.json")
    if os.path.exists(meta_path):
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
            if "vocab_size" in meta:
                config_kwargs["vocab_size"] = meta["vocab_size"]

    config = GPTConfig(**config_kwargs)

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
    print(f"MLP type: {config.mlp_type}")
    print(f"Context length (block_size): {config.block_size}")
    print(f"Batch size: {config.batch_size}")
    print(f"Grad accumulation steps: {config.grad_accumulation_steps}")
    print(f"Effective batch size: {config.batch_size * config.grad_accumulation_steps}")
    print(f"Total iterations: {config.max_iters}")
    print(f"Peak learning rate: {config.learning_rate:.2e}")
    print(f"AMP: {'ON' if config.use_amp else 'OFF'}")
    if args.resume:
        print(f"Resuming from: {args.resume}")
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
    train(model, train_loader, val_loader, config, resume_from=args.resume)


if __name__ == "__main__":
    main()
