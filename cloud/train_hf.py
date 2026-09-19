"""
Train miniGPT on a Kaggle GPU (or HF Jobs) and upload the trained artifacts
to the Hugging Face Hub.

Profiles:
  - Smoke Test: Automatically activated if MAX_ITERS <= 10 or SMOKE_TEST=1.
    Runs fast 200k tokens prep, 5 iters, 5 eval, 5 save, and uploads artifacts.
  - Real Training: Defaults to 20,000 iters, 100M tokens TinyStories, 1,000 warmup,
    eval every 250 iters, save every 1,000 iters.

Required environment variables:
    HF_TOKEN
    HF_REPO_ID

Optional environment variables:
    MAX_ITERS=20000
    WARMUP_ITERS=1000
    BATCH_SIZE=32
    GRAD_ACCUM=2
    D_MODEL=384
    N_LAYERS=6
    N_HEADS=6
    BLOCK_SIZE=256
    MLP_TYPE=gelu
    LEARNING_RATE=3e-4
    EVAL_INTERVAL=250
    SAVE_INTERVAL=1000
    DATASET=tinystories
    MAX_TRAIN_TOKENS=100000000
    MAX_VAL_TOKENS=1000000

This script is designed to be executed with:

    uv run cloud/train_hf.py
"""

# /// script
# dependencies = [
#   "torch",
#   "numpy",
#   "requests",
#   "tiktoken",
#   "huggingface_hub",
#   "wrapt",
#   "datasets",
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
    # Environment & Secrets
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
    # Configuration & Profiles (Smoke Test vs Real Training)
    # ------------------------------------------------------------------

    raw_max_iters = get_env("MAX_ITERS", "20000")
    max_iters_val = int(raw_max_iters)
    is_smoke_test = max_iters_val <= 10 or os.getenv("SMOKE_TEST", "0") == "1"

    if is_smoke_test:
        default_warmup = min(5, max_iters_val)
        default_eval = max_iters_val
        default_save = max_iters_val
        default_max_train_tokens = "200000"
        default_max_val_tokens = "20000"
        profile_name = "Smoke Test (Quick Verification)"
    else:
        default_warmup = 1000
        default_eval = 250
        default_save = 1000
        default_max_train_tokens = "100000000"
        default_max_val_tokens = "1000000"
        profile_name = "Real Training"

    warmup_iters = int(get_env("WARMUP_ITERS", str(default_warmup)))
    eval_interval = int(get_env("EVAL_INTERVAL", str(default_eval)))
    save_interval = int(get_env("SAVE_INTERVAL", str(default_save)))

    batch_size = get_env("BATCH_SIZE", "32")
    grad_accum = get_env("GRAD_ACCUM", "2")
    d_model = get_env("D_MODEL", "384")
    n_layers = get_env("N_LAYERS", "6")
    n_heads = get_env("N_HEADS", "6")
    block_size = get_env("BLOCK_SIZE", "256")
    mlp_type = get_env("MLP_TYPE", "gelu")
    learning_rate = get_env("LEARNING_RATE", "3e-4")

    dataset_name = get_env("DATASET", "tinystories")
    max_train_tokens = get_env("MAX_TRAIN_TOKENS", default_max_train_tokens)
    max_val_tokens = get_env("MAX_VAL_TOKENS", default_max_val_tokens)

    print("=" * 60, flush=True)
    print(f" Profile: {profile_name}", flush=True)
    print("=" * 60, flush=True)
    print(f"  DATASET          = {dataset_name}", flush=True)
    print(f"  MAX_TRAIN_TOKENS = {int(max_train_tokens):,}", flush=True)
    print(f"  MAX_VAL_TOKENS   = {int(max_val_tokens):,}", flush=True)
    print(f"  MAX_ITERS        = {max_iters_val}", flush=True)
    print(f"  WARMUP_ITERS     = {warmup_iters}", flush=True)
    print(f"  BATCH_SIZE       = {batch_size}", flush=True)
    print(f"  GRAD_ACCUM       = {grad_accum}", flush=True)
    print(f"  EFFECTIVE_BATCH  = {int(batch_size) * int(grad_accum)}", flush=True)
    print(f"  D_MODEL          = {d_model}", flush=True)
    print(f"  N_LAYERS         = {n_layers}", flush=True)
    print(f"  N_HEADS          = {n_heads}", flush=True)
    print(f"  BLOCK_SIZE       = {block_size}", flush=True)
    print(f"  MLP_TYPE         = {mlp_type}", flush=True)
    print(f"  LEARNING_RATE    = {learning_rate}", flush=True)
    print(f"  EVAL_INTERVAL    = {eval_interval}", flush=True)
    print(f"  SAVE_INTERVAL    = {save_interval}", flush=True)
    print("=" * 60, flush=True)

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

    print(f"\nPreparing dataset ({dataset_name})...", flush=True)

    prep_cmd = [
        "python",
        "data/prepare_data.py",
        "--dataset",
        dataset_name,
        "--max_train_tokens",
        str(max_train_tokens),
        "--max_val_tokens",
        str(max_val_tokens),
    ]
    run(prep_cmd, cwd=WORKDIR)

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
            str(max_iters_val),
            "--warmup_iters",
            str(warmup_iters),
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
            "--learning_rate",
            learning_rate,
            "--eval_interval",
            str(eval_interval),
            "--save_interval",
            str(save_interval),
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

    # If best_model.pt was not produced (e.g. max_iters < eval_interval), copy final_model.pt
    best_path = WORKDIR / "runs" / "best_model.pt"
    final_path = WORKDIR / "runs" / "final_model.pt"
    if final_path.exists() and not best_path.exists():
        shutil.copy2(final_path, best_path)

    # Upload trained models and checkpoints.
    for filename in (
        "best_model.pt",
        "final_model.pt",
        "latest.pt",
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

        elif filename != "latest.pt":
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
