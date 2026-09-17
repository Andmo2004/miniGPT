# data/prepare_data.py – Download raw text, train tokenizer, save token IDs

import os
import json
import numpy as np
import requests

# pyrefly: ignore [missing-import]
import tiktoken

# TODO: Create a script that prepares the training data end-to-end.
#
# ── What this script does (run once before training) ──
#   1. Download a raw text dataset (e.g., TinyShakespeare or OpenWebText subset).
#   2. Train a character-level or BPE tokenizer on the text.
#   3. Encode the full text into integer token IDs.
#   4. Split into train / val sets.
#   5. Save the encoded arrays to disk as .bin or .npy files.
  
# ── OPTION A: Character-level tokenizer (simplest, great for learning) ── NON TAKEN
#   def build_char_tokenizer(text):
#       # TODO:
#       #   1. Get sorted list of unique characters:
#       #   2. Build char->int and int->char mappings:
#       #   3. Create encode/decode functions:
#       #   4. Save the mappings (pickle or json) for later use
#       #   5. Return encode, decode, vocab_size

# ── OPTION B: BPE tokenizer with `tiktoken` (GPT-2 style, production) ──

def build_bpe_tokenizer():
    """ BPE tokenizer with `tiktoken` (GPT-2 style, production) """
    # TODO:
    #   1. import tiktoken
    #   2. enc = tiktoken.get_encoding("gpt2")   # 50257 vocab
    #   3. encode = enc.encode
    #   4. decode = enc.decode
    #   5. vocab_size = enc.n_vocab
    #   6. Return encode, decode, vocab_size

    enc = tiktoken.get_encoding("gpt2")
    encode = enc.encode
    decode = enc.decode
    vocab_size = enc.n_vocab

    return encode, decode, vocab_size

# ── Encode and split ──

def download_text(url: str, save_path: str) -> str:
    """Download a raw text corpus and save it locally."""
    response = requests.get(url, timeout=30)
    response.raise_for_status()

    with open(save_path, "w", encoding="utf-8") as f:
        f.write(response.text)

    return response.text


def prepare(data_dir="data/"):
    """Download or load the raw corpus, tokenize it, and save train/val splits."""
    # TODO:
    #   1. Download or load raw text.
    #   2. Build tokenizer (pick Option A or B).
    #   3. Encode the entire text:
    #   4. Split into train (90%) and val (10%):
    #   5. Save to disk:
    #   6. Save vocab_size to a metadata file (JSON) so train.py can read it.
    #   7. Print stats: total tokens, train tokens, val tokens, vocab size.
    
    data_dir = os.path.abspath(data_dir)
    os.makedirs(data_dir, exist_ok=True)

    raw_path = os.path.join(data_dir, "input.txt")
    url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"

    if not os.path.exists(raw_path):
        text = download_text(url, raw_path)
    else:
        with open(raw_path, "r", encoding="utf-8") as f:
            text = f.read()

    encode, decode, vocab_size = build_bpe_tokenizer()
    token_ids = encode(text)
    data = np.asarray(token_ids, dtype=np.uint16)

    n = int(0.9 * len(data))
    train_data = data[:n]
    val_data = data[n:]

    np.save(os.path.join(data_dir, "train.npy"), train_data)
    np.save(os.path.join(data_dir, "val.npy"), val_data)

    metadata = {
        "vocab_size": int(vocab_size),
        "total_tokens": int(len(data)),
        "train_tokens": int(len(train_data)),
        "val_tokens": int(len(val_data)),
    }
    with open(os.path.join(data_dir, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print(f"Total tokens: {len(data)}")
    print(f"Train tokens: {len(train_data)}")
    print(f"Val tokens: {len(val_data)}")
    print(f"Vocab size: {vocab_size}")

    return metadata


# ── Run as script ──

if __name__ == "__main__":
    prepare()

