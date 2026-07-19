#!/usr/bin/env python3
"""Download WikiText-2 + tokenizer files only (no model weights)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from datasets import load_dataset
from transformers import AutoTokenizer

from mxfp4_lib.util import ensure_dir, hf_env, load_cfg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    args = ap.parse_args()
    cfg = load_cfg(args.config)
    hf_env(cfg)

    tok_id = cfg["tokenizer_id"]
    print(f"[prepare] Loading tokenizer files only: {tok_id}")
    # tokenizer files only — never download model weights
    tok = AutoTokenizer.from_pretrained(tok_id, use_fast=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok_dir = ensure_dir(Path(cfg["paths"]["data_dir"]) / "tokenizer")
    tok.save_pretrained(tok_dir)
    print(f"[prepare] Tokenizer saved to {tok_dir} vocab_size={len(tok)}")

    print("[prepare] Loading WikiText-2 raw...")
    ds = load_dataset(cfg["data"]["dataset"], cfg["data"]["dataset_config"])
    data_dir = ensure_dir(cfg["paths"]["data_dir"])
    ds.save_to_disk(str(data_dir / "wikitext2"))
    print(f"[prepare] Dataset saved to {data_dir / 'wikitext2'}")
    for split in ds:
        print(f"  {split}: {len(ds[split])} rows")


if __name__ == "__main__":
    main()
