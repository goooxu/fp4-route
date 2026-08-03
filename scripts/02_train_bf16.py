#!/usr/bin/env python3
"""R1/R2/R4 shared: from-scratch BF16 training (R2=NVFP4 infer, R4=MXFP8 infer)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from transformers import AutoModelForCausalLM

from mxfp4_lib.train_loop import train_loop
from mxfp4_lib.util import hf_env, load_cfg, set_seed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--max_steps", type=int, default=None)
    ap.add_argument("--seed", type=int, default=None)
    args = ap.parse_args()
    cfg = load_cfg(args.config, seed=args.seed)
    hf_env(cfg)
    set_seed(cfg["seed"])

    init_dir = cfg["paths"]["init_model"]
    print(f"[bf16] Loading shared init from {init_dir}")
    model = AutoModelForCausalLM.from_pretrained(init_dir)
    train_loop(
        model,
        cfg,
        out_dir=cfg["paths"]["ckpt_bf16"],
        log_name="train_bf16.jsonl",
        max_steps=args.max_steps,
    )


if __name__ == "__main__":
    main()
