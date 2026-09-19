# generate.py – Inference / text generation entrypoint

import argparse
import os
import torch
import tiktoken

from config import GPTConfig
from model import GPT
from engine import generate_text


def main():
    parser = argparse.ArgumentParser(description="Generate text with trained miniGPT")
    parser.add_argument("--prompt", type=str, default="ROMEO:", help="Initial text prompt")
    parser.add_argument("--max_tokens", type=int, default=200, help="Maximum number of new tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.8,
                        help="Sampling temperature (higher = more creative, 0 = greedy)")
    parser.add_argument("--top_k", type=int, default=50, help="Top-k filtering threshold (<=0 to disable)")
    parser.add_argument("--top_p", type=float, default=None, help="Top-p nucleus sampling threshold (None to disable)")
    parser.add_argument("--checkpoint", type=str, default="runs/best_model.pt",
                        help="Path to checkpoint file")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducible generation")
    args = parser.parse_args()

    # 1. Find checkpoint
    checkpoint_path = args.checkpoint
    fallback_paths = [
        "runs/final_model.pt",
        "checkpoints/best_model.pt",   # legacy location
        "checkpoints/final_model.pt",  # legacy location
        "best_model.pt",               # legacy location
        "final_model.pt",              # legacy location
    ]

    if not os.path.exists(checkpoint_path):
        found = False
        for fallback in fallback_paths:
            if os.path.exists(fallback):
                print(f"Checkpoint '{checkpoint_path}' not found, using '{fallback}'...")
                checkpoint_path = fallback
                found = True
                break
        if not found:
            raise FileNotFoundError(
                f"No checkpoint found. Run `python train.py` first to train the model."
            )

    # 2. Load checkpoint and extract config
    print(f"Loading model checkpoint from {checkpoint_path}...")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    # Support both new-style (dict with 'model' key) and legacy (plain state_dict) checkpoints
    if isinstance(checkpoint, dict) and "model" in checkpoint:
        # New-style checkpoint: load config from checkpoint
        config_dict = checkpoint.get("config", {})
        config = GPTConfig.from_dict(config_dict)
        state_dict = checkpoint["model"]
        print(f"Loaded config from checkpoint (d_model={config.d_model}, "
              f"n_layers={config.n_layers}, n_heads={config.n_heads})")
    else:
        # Legacy checkpoint: plain state_dict, use default config
        print("Legacy checkpoint detected (no config saved). Using default GPTConfig.")
        config = GPTConfig()
        state_dict = checkpoint

    # 3. Tokenizer (BPE GPT-2 style used during preparation)
    enc = tiktoken.get_encoding("gpt2")
    encode_fn = enc.encode
    decode_fn = enc.decode

    # 4. Build model and load weights
    model = GPT(config)
    model.load_state_dict(state_dict)
    model.to(config.device)
    model.eval()

    # 5. Generate text
    top_k = args.top_k if (args.top_k is not None and args.top_k > 0) else None

    print("-" * 60)
    print(f"Prompt: {args.prompt}")
    print(f"Temperature: {args.temperature} {'(greedy)' if args.temperature == 0.0 else ''}")
    if top_k: print(f"Top-k: {top_k}")
    if args.top_p: print(f"Top-p: {args.top_p}")
    if args.seed is not None: print(f"Seed: {args.seed}")
    print("-" * 60)

    generated_output = generate_text(
        model=model,
        encode_fn=encode_fn,
        decode_fn=decode_fn,
        prompt=args.prompt,
        max_new_tokens=args.max_tokens,
        temperature=args.temperature,
        top_k=top_k,
        top_p=args.top_p,
        device=config.device,
        seed=args.seed,
    )

    print(generated_output)
    print("-" * 60)


if __name__ == "__main__":
    main()
