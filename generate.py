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
    parser.add_argument("--temperature", type=float, default=0.8, help="Sampling temperature (higher = more creative)")
    parser.add_argument("--top_k", type=int, default=50, help="Top-k filtering threshold (None or <=0 to disable)")
    parser.add_argument("--top_p", type=float, default=None, help="Top-p nucleus sampling threshold (None to disable)")
    parser.add_argument("--checkpoint", type=str, default="best_model.pt", help="Path to checkpoint file")
    args = parser.parse_args()

    # 1. Config matching the trained architecture
    config = GPTConfig()

    # 2. Tokenizer (BPE GPT-2 style used during preparation)
    enc = tiktoken.get_encoding("gpt2")
    encode_fn = enc.encode
    decode_fn = enc.decode

    # 3. Build model and load weights
    if not os.path.exists(args.checkpoint):
        # Fallback to final_model.pt if best_model.pt not found
        fallback = "final_model.pt"
        if os.path.exists(fallback):
            print(f"Checkpoint '{args.checkpoint}' not found, using '{fallback}'...")
            args.checkpoint = fallback
        else:
            raise FileNotFoundError(
                f"Checkpoint not found at '{args.checkpoint}'. Run `python train.py` first to train the model."
            )

    print(f"Loading model checkpoint from {args.checkpoint}...")
    model = GPT(config)
    state_dict = torch.load(args.checkpoint, map_location=config.device, weights_only=True)
    model.load_state_dict(state_dict)
    model.to(config.device)
    model.eval()

    # 4. Generate text
    top_k = args.top_k if (args.top_k is not None and args.top_k > 0) else None

    print("-" * 60)
    print(f"Prompt: {args.prompt}")
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
    )

    print(generated_output)
    print("-" * 60)


if __name__ == "__main__":
    main()
