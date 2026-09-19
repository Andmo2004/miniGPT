"""
Train miniGPT on a Kaggle GPU and upload the trained artifacts
to the Hugging Face Hub.

Required environment variables:
    HF_TOKEN
    HF_REPO_ID

Optional environment variables:
    MAX_ITERS=5000
    BATCH_SIZE=64
    GRAD_ACCUM=2
    D_MODEL=384
    N_LAYERS=6
    N_HEADS=6
    BLOCK_SIZE=256
    MLP_TYPE=gelu

This script is designed to be executed with:

    uv run train_hf.py
"""

# /// script
# dependencies = [
#   "torch",
#   "numpy",
#   "requests",
#   "tiktoken",
#   "huggingface_hub",
# ]
# ///

import os
import shutil
import subprocess
from pathlib import Path

# pyrefly: ignore [missing-import]
from huggingface_hub import HfApi


REPO_URL = "https://github.com/Andmo2004/miniGPT.git"
WORKDIR = Path("/tmp/miniGPT")


def run(cmd, cwd=None):
    """Run a command and stop immediately if it fails."""
    print("+", " ".join(map(str, cmd)), flush=True)
    subprocess.run(cmd, cwd=cwd, check=True)


def get_env(name, default):
    """Read an environment variable with a default value."""
    return os.getenv(name, default)


def main():
    # ------------------------------------------------------------------
    # Environment
    # ------------------------------------------------------------------

    token = os.environ.get("HF_TOKEN")
    hf_repo_id = os.environ.get("HF_REPO_ID")

    if not token:
        raise RuntimeError(
            "HF_TOKEN is required. "
            "Add it as a Kaggle Secret and expose it to the notebook."
        )

    if not hf_repo_id:
        raise RuntimeError(
            "HF_REPO_ID is required, e.g. Andmo2004/miniGPT."
        )

    print(f"HF repository: {hf_repo_id}", flush=True)

    # ------------------------------------------------------------------
    # Configuration
    # ------------------------------------------------------------------

    max_iters = get_env("MAX_ITERS", "5000")
    batch_size = get_env("BATCH_SIZE", "64")
    grad_accum = get_env("GRAD_ACCUM", "2")
    d_model = get_env("D_MODEL", "384")
    n_layers = get_env("N_LAYERS", "6")
    n_heads = get_env("N_HEADS", "6")
    block_size = get_env("BLOCK_SIZE", "256")
    mlp_type = get_env("MLP_TYPE", "gelu")

    print("\nTraining configuration:", flush=True)
    print(f"  MAX_ITERS   = {max_iters}", flush=True)
    print(f"  BATCH_SIZE  = {batch_size}", flush=True)
    print(f"  GRAD_ACCUM  = {grad_accum}", flush=True)
    print(f"  D_MODEL     = {d_model}", flush=True)
    print(f"  N_LAYERS    = {n_layers}", flush=True)
    print(f"  N_HEADS     = {n_heads}", flush=True)
    print(f"  BLOCK_SIZE  = {block_size}", flush=True)
    print(f"  MLP_TYPE    = {mlp_type}", flush=True)

    # ------------------------------------------------------------------
    # Prepare working directory
    # ------------------------------------------------------------------

    if WORKDIR.exists():
        print(f"\nRemoving existing directory: {WORKDIR}", flush=True)
        shutil.rmtree(WORKDIR)

    print(f"\nCloning repository into {WORKDIR}...", flush=True)

    run(
        [
            "git",
            "clone",
            "--depth",
            "1",
            REPO_URL,
            str(WORKDIR),
        ]
    )

    # ------------------------------------------------------------------
    # Prepare dataset
    # ------------------------------------------------------------------

    print("\nPreparing dataset...", flush=True)

    run(
        [
            "python",
            "data/prepare_data.py",
        ],
        cwd=WORKDIR,
    )

    # ------------------------------------------------------------------
    # Train
    # ------------------------------------------------------------------

    print("\nStarting training...", flush=True)

    run(
        [
            "python",
            "train.py",
            "--data_dir",
            "data",
            "--out_dir",
            "runs",
            "--max_iters",
            max_iters,
            "--batch_size",
            batch_size,
            "--grad_accumulation_steps",
            grad_accum,
            "--d_model",
            d_model,
            "--n_layers",
            n_layers,
            "--n_heads",
            n_heads,
            "--block_size",
            block_size,
            "--mlp_type",
            mlp_type,
            "--use_amp",
        ],
        cwd=WORKDIR,
    )

    # ------------------------------------------------------------------
    # Upload artifacts to Hugging Face Hub
    # ------------------------------------------------------------------

    print("\nTraining finished.", flush=True)
    print("Connecting to Hugging Face Hub...", flush=True)

    api = HfApi(token=token)

    api.create_repo(
        repo_id=hf_repo_id,
        repo_type="model",
        exist_ok=True,
    )

    # Upload trained models.
    for filename in (
        "best_model.pt",
        "final_model.pt",
    ):
        path = WORKDIR / "runs" / filename

        if path.exists():
            print(
                f"Uploading {filename} to {hf_repo_id}...",
                flush=True,
            )

            api.upload_file(
                path_or_fileobj=str(path),
                path_in_repo=filename,
                repo_id=hf_repo_id,
                repo_type="model",
            )

        else:
            print(
                f"Warning: {filename} was not found.",
                flush=True,
            )

    # Upload dataset metadata.
    meta = WORKDIR / "data" / "meta.json"

    if meta.exists():
        print("Uploading meta.json...", flush=True)

        api.upload_file(
            path_or_fileobj=str(meta),
            path_in_repo="meta.json",
            repo_id=hf_repo_id,
            repo_type="model",
        )

    print(
        f"\nTraining complete. "
        f"Artifacts uploaded to https://huggingface.co/{hf_repo_id}",
        flush=True,
    )


if __name__ == "__main__":
    main()
