# data/prepare_data.py – Download text corpus, tokenize, and save train/val token IDs

import argparse
import json
import os
import numpy as np
import requests

# pyrefly: ignore [missing-import]
import tiktoken


def build_bpe_tokenizer():
    """BPE tokenizer with tiktoken (GPT-2 standard, 50,257 vocab)."""
    enc = tiktoken.get_encoding("gpt2")
    return enc, enc.encode, enc.decode, enc.n_vocab


def download_text(url: str, save_path: str) -> str:
    """Download a raw text corpus and save it locally."""
    response = requests.get(url, timeout=30)
    response.raise_for_status()

    with open(save_path, "w", encoding="utf-8") as f:
        f.write(response.text)

    return response.text


def prepare_tinyshakespeare(data_dir: str = "data", max_train_tokens: int | None = None, max_val_tokens: int | None = None):
    """Download Tiny Shakespeare, tokenize with GPT-2 BPE, and save train/val splits."""
    data_dir = os.path.abspath(data_dir)
    os.makedirs(data_dir, exist_ok=True)

    raw_path = os.path.join(data_dir, "input.txt")
    url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"

    if not os.path.exists(raw_path):
        print("Downloading Tiny Shakespeare corpus...", flush=True)
        text = download_text(url, raw_path)
    else:
        with open(raw_path, "r", encoding="utf-8") as f:
            text = f.read()

    enc, encode, _, vocab_size = build_bpe_tokenizer()
    token_ids = encode(text)
    data = np.asarray(token_ids, dtype=np.uint16)

    n = int(0.9 * len(data))
    train_data = data[:n]
    val_data = data[n:]

    if max_train_tokens is not None and len(train_data) > max_train_tokens:
        train_data = train_data[:max_train_tokens]
    if max_val_tokens is not None and len(val_data) > max_val_tokens:
        val_data = val_data[:max_val_tokens]

    np.save(os.path.join(data_dir, "train.npy"), train_data)
    np.save(os.path.join(data_dir, "val.npy"), val_data)

    metadata = {
        "dataset": "tinyshakespeare",
        "vocab_size": int(vocab_size),
        "total_tokens": int(len(train_data) + len(val_data)),
        "train_tokens": int(len(train_data)),
        "val_tokens": int(len(val_data)),
    }
    with open(os.path.join(data_dir, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print("\nTiny Shakespeare preparation complete:")
    print(f"  Train tokens: {len(train_data):,}")
    print(f"  Val tokens:   {len(val_data):,}")
    print(f"  Vocab size:   {vocab_size}")

    return metadata


def prepare_tinystories(data_dir: str = "data", max_train_tokens: int | None = 100_000_000, max_val_tokens: int | None = 1_000_000):
    """
    Stream and tokenize the TinyStories dataset (roneneldan/TinyStories)
    from Hugging Face using tiktoken (GPT-2 BPE) with <|endoftext|> delimiters.
    """
    try:
        # pyrefly: ignore [missing-import]
        from datasets import load_dataset
    except ImportError:
        raise ImportError(
            "The 'datasets' library is required to prepare TinyStories.\n"
            "Please install it with: pip install datasets"
        )

    data_dir = os.path.abspath(data_dir)
    os.makedirs(data_dir, exist_ok=True)

    enc, _, _, vocab_size = build_bpe_tokenizer()
    eot_token = enc.eot_token

    print("Connecting to Hugging Face datasets: roneneldan/TinyStories (streaming)...", flush=True)

    def process_split(split_name: str, max_tokens: int | None):
        target_str = f"{max_tokens:,}" if max_tokens else "all"
        print(f"\nProcessing '{split_name}' split (target tokens: {target_str})...", flush=True)

        ds = load_dataset("roneneldan/TinyStories", split=split_name, streaming=True)
        tokens_list = []
        total_tokens = 0

        for idx, sample in enumerate(ds):
            text = sample.get("text", "")
            if not text:
                continue

            story_tokens = enc.encode(text)
            story_tokens.append(eot_token)
            tokens_list.extend(story_tokens)
            total_tokens += len(story_tokens)

            if idx > 0 and idx % 2000 == 0:
                print(f"  [{split_name}] Processed {idx:,} stories ({total_tokens:,} tokens)...", flush=True)

            if max_tokens is not None and total_tokens >= max_tokens:
                tokens_list = tokens_list[:max_tokens]
                total_tokens = max_tokens
                break

        print(f"Finished '{split_name}': {total_tokens:,} tokens collected.", flush=True)
        return np.asarray(tokens_list, dtype=np.uint16)

    train_data = process_split("train", max_train_tokens)
    val_data = process_split("validation", max_val_tokens)

    np.save(os.path.join(data_dir, "train.npy"), train_data)
    np.save(os.path.join(data_dir, "val.npy"), val_data)

    metadata = {
        "dataset": "tinystories",
        "vocab_size": int(vocab_size),
        "total_tokens": int(len(train_data) + len(val_data)),
        "train_tokens": int(len(train_data)),
        "val_tokens": int(len(val_data)),
    }
    with open(os.path.join(data_dir, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print("\nTinyStories preparation complete:")
    print(f"  Train tokens: {len(train_data):,}")
    print(f"  Val tokens:   {len(val_data):,}")
    print(f"  Vocab size:   {vocab_size}")

    return metadata


def parse_args():
    parser = argparse.ArgumentParser(description="Prepare dataset for miniGPT training")
    parser.add_argument(
        "--dataset",
        type=str,
        default="tinystories",
        choices=["tinystories", "tinyshakespeare"],
        help="Dataset to download and prepare (default: tinystories)",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="data",
        help="Target directory for processed files (default: data)",
    )
    parser.add_argument(
        "--max_train_tokens",
        type=int,
        default=None,
        help="Max tokens for training split (default: 100,000,000 for tinystories)",
    )
    parser.add_argument(
        "--max_val_tokens",
        type=int,
        default=None,
        help="Max tokens for validation split (default: 1,000,000 for tinystories)",
    )
    parser.add_argument(
        "--smoke_test",
        action="store_true",
        help="Shortcut for smoke test: 200,000 train tokens and 20,000 val tokens",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    max_train = args.max_train_tokens
    max_val = args.max_val_tokens

    if args.smoke_test:
        max_train = 200_000
        max_val = 20_000
    elif args.dataset == "tinystories":
        if max_train is None:
            max_train = 100_000_000
        if max_val is None:
            max_val = 1_000_000

    if args.dataset == "tinystories":
        prepare_tinystories(
            data_dir=args.data_dir,
            max_train_tokens=max_train,
            max_val_tokens=max_val,
        )
    else:
        prepare_tinyshakespeare(
            data_dir=args.data_dir,
            max_train_tokens=max_train,
            max_val_tokens=max_val,
        )


if __name__ == "__main__":
    main()
