#!/usr/bin/env python3
"""Download tokenizer + WikiText-2 (eval) + optionally warm FineWeb-Edu token caches."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from datasets import load_dataset
from transformers import AutoTokenizer

from mxfp4_lib.data import build_fineweb_token_buffer
from mxfp4_lib.util import ensure_dir, hf_env, load_cfg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--prefetch-fineweb", action="store_true", help="Build train/val token caches now")
    args = ap.parse_args()
    cfg = load_cfg(args.config, seed=args.seed)
    hf_env(cfg)

    tok_id = cfg["tokenizer_id"]
    print(f"[prepare] Loading tokenizer files only: {tok_id}")
    tok = AutoTokenizer.from_pretrained(tok_id, use_fast=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok_dir = ensure_dir(Path(cfg["paths"]["data_dir"]) / "tokenizer")
    tok.save_pretrained(tok_dir)
    print(f"[prepare] Tokenizer saved to {tok_dir} vocab_size={len(tok)}")

    wikitext_name = cfg["data"].get("eval_dataset", cfg["data"].get("dataset", "Salesforce/wikitext"))
    wikitext_config = cfg["data"].get(
        "eval_dataset_config", cfg["data"].get("dataset_config", "wikitext-2-raw-v1")
    )
    print(f"[prepare] Loading WikiText-2 for eval: {wikitext_name} / {wikitext_config}")
    ds = load_dataset(wikitext_name, wikitext_config)
    data_dir = ensure_dir(cfg["paths"]["data_dir"])
    ds.save_to_disk(str(data_dir / "wikitext2"))
    print(f"[prepare] Dataset saved to {data_dir / 'wikitext2'}")
    for split in ds:
        print(f"  {split}: {len(ds[split])} rows")

    if args.prefetch_fineweb or cfg.get("data", {}).get("prefetch_fineweb", False):
        target = int(cfg["train"].get("target_tokens") or 0)
        vtok = int(cfg.get("eval", {}).get("val_tokens") or 0)
        if target > 0:
            build_fineweb_token_buffer(cfg, tok, target_tokens=target, name="train", seed=cfg["seed"])
        if vtok > 0:
            build_fineweb_token_buffer(
                cfg, tok, target_tokens=vtok, name="val", seed=cfg["seed"] + 10_000
            )


if __name__ == "__main__":
    main()
