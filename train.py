# train.py – Training entrypoint
#
# TODO: Create the main script that ties everything together and launches
#       training.  Run this with: python train.py
#
# ── What this script does ──
#   1. Parse command-line arguments (or use defaults from GPTConfig).
#   2. Set random seeds for reproducibility.
#   3. Load data and create DataLoaders.
#   4. Instantiate the GPT model.
#   5. Launch the training loop.
#
# ── Implementation ──
#
#   def main():
#       # TODO:
#       #   1. Create config:
#       #        config = GPTConfig()
#       #
#       #   2. (Optional) Parse CLI args to override config defaults:
#       #        import argparse
#       #        parser = argparse.ArgumentParser()
#       #        parser.add_argument('--batch_size', type=int, default=config.batch_size)
#       #        parser.add_argument('--learning_rate', type=float, default=config.learning_rate)
#       #        parser.add_argument('--max_iters', type=int, default=config.max_iters)
#       #        parser.add_argument('--d_model', type=int, default=config.d_model)
#       #        parser.add_argument('--n_layers', type=int, default=config.n_layers)
#       #        parser.add_argument('--n_heads', type=int, default=config.n_heads)
#       #        args = parser.parse_args()
#       #        # Apply overrides to config...
#       #
#       #   3. Set seeds for reproducibility:
#       #        torch.manual_seed(config.seed)
#       #        if torch.cuda.is_available():
#       #            torch.cuda.manual_seed(config.seed)
#       #        # MPS doesn't have a separate manual_seed — torch.manual_seed
#       #        # already covers it. No extra step needed for your M4.
#       #
#       #   4. Load data:
#       #        from data.dataset import get_dataloaders
#       #        train_loader, val_loader = get_dataloaders(
#       #            data_dir="data/", block_size=config.block_size,
#       #            batch_size=config.batch_size
#       #        )
#       #
#       #   5. (Optional) Load vocab_size from data metadata and update config.
#       #
#       #   6. Create model:
#       #        from model import GPT
#       #        model = GPT(config)
#       #
#       #   7. Train:
#       #        from engine.trainer import train
#       #        train(model, train_loader, val_loader, config)
#
#   if __name__ == "__main__":
#       main()
