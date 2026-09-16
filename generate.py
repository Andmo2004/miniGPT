# generate.py – Inference / text generation entrypoint
#
# TODO: Create a script that loads a trained checkpoint and generates text
#       from a user-provided prompt via CLI.
#
#   Run with: python generate.py --prompt "To be or not to be"
#
# ── Implementation ──
#
#   def main():
#       # TODO:
#       #   1. Parse arguments:
#       #        parser = argparse.ArgumentParser()
#       #        parser.add_argument('--prompt', type=str, default="Once upon a time")
#       #        parser.add_argument('--max_tokens', type=int, default=200)
#       #        parser.add_argument('--temperature', type=float, default=0.8)
#       #        parser.add_argument('--top_k', type=int, default=50)
#       #        parser.add_argument('--top_p', type=float, default=None)
#       #        parser.add_argument('--checkpoint', type=str, default="best_model.pt")
#       #        args = parser.parse_args()
#       #
#       #   2. Recreate the config (must match the model that was trained):
#       #        config = GPTConfig()
#       #
#       #   3. Load the tokenizer (same one used during training):
#       #        # If char-level: load the stoi/itos dicts from the saved file
#       #        # If BPE: enc = tiktoken.get_encoding("gpt2")
#       #
#       #   4. Build the model and load checkpoint weights:
#       #        model = GPT(config)
#       #        model.load_state_dict(torch.load(args.checkpoint, map_location=config.device))
#       #        model.to(config.device)
#       #        model.eval()
#       #
#       #   5. Generate:
#       #        from engine.sampler import generate_text
#       #        output = generate_text(
#       #            model, encode_fn, decode_fn,
#       #            prompt=args.prompt,
#       #            max_new_tokens=args.max_tokens,
#       #            temperature=args.temperature,
#       #            top_k=args.top_k,
#       #            top_p=args.top_p,
#       #            device=config.device,
#       #        )
#       #        print(output)
#
#   if __name__ == "__main__":
#       main()
