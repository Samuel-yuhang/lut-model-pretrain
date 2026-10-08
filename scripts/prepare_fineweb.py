"""Download FineWeb sample-10BT shards, tokenize with the Qwen3 tokenizer, write packed uint32 token files.

Documents are joined with EOS. Outputs (in --out_dir):
  train.bin   first --train_tokens tokens of shard 000
  calib.bin   next --calib_tokens tokens of shard 000 (k-means init, disjoint from train)
  heldout.bin first --heldout_tokens tokens of shard 001
"""

import argparse
import os
from multiprocessing import Pool

import numpy as np
import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download
from transformers import AutoTokenizer

TOK = None


def _init(name):
    global TOK
    TOK = AutoTokenizer.from_pretrained(name)


def _encode(texts):
    ids = TOK(texts, add_special_tokens=False).input_ids
    out = []
    for x in ids:
        out.extend(x)
        out.append(TOK.eos_token_id)
    return np.asarray(out, dtype=np.uint32)


def tokenize_shard(path, n_tokens, tokenizer, workers):
    pf = pq.ParquetFile(path)
    chunks, total = [], 0
    with Pool(workers, initializer=_init, initargs=(tokenizer,)) as pool:
        for rg in range(pf.num_row_groups):
            texts = pf.read_row_group(rg, columns=["text"]).column("text").to_pylist()
            batches = [texts[i : i + 256] for i in range(0, len(texts), 256)]
            for arr in pool.imap(_encode, batches):
                chunks.append(arr)
                total += arr.size
            print(f"{os.path.basename(path)} row_group {rg}: {total / 1e6:.1f}M tokens", flush=True)
            if total >= n_tokens:
                break
    return np.concatenate(chunks)[:n_tokens]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokenizer", default="Qwen/Qwen3-0.6B-Base")
    ap.add_argument("--out_dir", default="data/fineweb10bt")
    ap.add_argument("--train_tokens", type=int, default=210_000_000)
    ap.add_argument("--calib_tokens", type=int, default=2_000_000)
    ap.add_argument("--heldout_tokens", type=int, default=2_000_000)
    ap.add_argument("--workers", type=int, default=96)
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    shard0 = hf_hub_download("HuggingFaceFW/fineweb", "sample/10BT/000_00000.parquet", repo_type="dataset")
    shard1 = hf_hub_download("HuggingFaceFW/fineweb", "sample/10BT/001_00000.parquet", repo_type="dataset")

    ids = tokenize_shard(shard0, args.train_tokens + args.calib_tokens, args.tokenizer, args.workers)
    ids[: args.train_tokens].tofile(f"{args.out_dir}/train.bin")
    ids[args.train_tokens :].tofile(f"{args.out_dir}/calib.bin")
    tokenize_shard(shard1, args.heldout_tokens, args.tokenizer, args.workers).tofile(f"{args.out_dir}/heldout.bin")
    for f in ("train", "calib", "heldout"):
        print(f, os.path.getsize(f"{args.out_dir}/{f}.bin") // 4, "tokens")


if __name__ == "__main__":
    main()
