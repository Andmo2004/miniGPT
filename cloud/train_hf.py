# train_hf.py

# /// script
# dependencies = [
#   "torch",
#   "numpy",
#   "requests",
#   "tiktoken",
#   "huggingface_hub",
# ]
# ///

"""
    Run Andmo2004/miniGPT on a Hugging Face Jobs GPU.

Required secret:
  HF_TOKEN

Required environment variable:
  HF_REPO_ID=YOUR_HF_USERNAME/miniGPT

Optional environment variables:
  MAX_ITERS=5000
  BATCH_SIZE=64
  GRAD_ACCUM=2
  D_MODEL=384
  N_LAYERS=6
  N_HEADS=6
  BLOCK_SIZE=256
  MLP_TYPE=gelu
  HF_TIMEOUT=8h

"""

import os
import shutil
import subprocess
from pathlib import Path

# pyrefly: ignore [missing-import]
from huggingface_hub import HfApi


REPO_URL = "https://github.com/Andmo2004/miniGPT.git"
WORKDIR = Path("/tmp/miniGPT")


def run(cmd, cwd=None):
    print("+", " ".join(map(str, cmd)), flush=True)
    subprocess.run(cmd, cwd=cwd, check=True)


def main():
    token = os.environ.get("HF_TOKEN")
    hf_repo_id = os.environ.get("HF_REPO_ID")

    if not token:
        raise RuntimeError("HF_TOKEN is required. Pass it with --secrets HF_TOKEN.")
    if not hf_repo_id:
        raise RuntimeError("HF_REPO_ID is required, e.g. yourname/miniGPT.")

    if WORKDIR.exists():
        shutil.rmtree(WORKDIR)

    run(["git", "clone", "--depth", "1", REPO_URL, str(WORKDIR)])

    # Prepare Tiny Shakespeare + GPT-2 BPE data.
    run(["python", "data/prepare_data.py"], cwd=WORKDIR)

    max_iters = os.getenv("MAX_ITERS", "5000")
    batch_size = os.getenv("BATCH_SIZE", "64")
    grad_accum = os.getenv("GRAD_ACCUM", "2")
    d_model = os.getenv("D_MODEL", "384")
    n_layers = os.getenv("N_LAYERS", "6")
    n_heads = os.getenv("N_HEADS", "6")
    block_size = os.getenv("BLOCK_SIZE", "256")
    mlp_type = os.getenv("MLP_TYPE", "gelu")

    # A10G/L4/A100 GPUs should use AMP. The trainer auto-selects BF16 when supported.
    run([
        "python", "train.py",
        "--data_dir", "data",
        "--out_dir", "runs",
        "--max_iters", max_iters,
        "--batch_size", batch_size,
        "--grad_accumulation_steps", grad_accum,
        "--d_model", d_model,
        "--n_layers", n_layers,
        "--n_heads", n_heads,
        "--block_size", block_size,
        "--mlp_type", mlp_type,
        "--use_amp",
    ], cwd=WORKDIR)

    api = HfApi(token=token)
    api.create_repo(
        repo_id=hf_repo_id,
        repo_type="model",
        exist_ok=True,
    )

    # Jobs storage is ephemeral, so persist the trained artifacts on the Hub.
    for filename in ("best_model.pt", "final_model.pt"):
        path = WORKDIR / "runs" / filename
        if path.exists():
            print(f"Uploading {path.name} to {hf_repo_id}...", flush=True)
            api.upload_file(
                path_or_fileobj=str(path),
                path_in_repo=filename,
                repo_id=hf_repo_id,
                repo_type="model",
            )

    meta = WORKDIR / "data" / "meta.json"
    if meta.exists():
        api.upload_file(
            path_or_fileobj=str(meta),
            path_in_repo="meta.json",
            repo_id=hf_repo_id,
            repo_type="model",
        )

    print(f"Training finished. Model artifacts uploaded to {hf_repo_id}", flush=True)


if __name__ == "__main__":
    main()
