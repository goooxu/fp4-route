#!/usr/bin/env python3
"""Route 3: from-scratch MXFP4 fake-quant training from shared init_model."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from transformers import AutoModelForCausalLM

from mxfp4_lib.replace import replace_linears_with_mxfp4, revert_mxfp4_to_linear
from mxfp4_lib.train_loop import train_loop
from mxfp4_lib.util import hf_env, load_cfg, set_seed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--max_steps", type=int, default=None)
    args = ap.parse_args()
    cfg = load_cfg(args.config)
    hf_env(cfg)
    set_seed(cfg["seed"])

    init_dir = cfg["paths"]["init_model"]
    print(f"[mxfp4] Loading shared init from {init_dir}")
    model = AutoModelForCausalLM.from_pretrained(init_dir)
    n = replace_linears_with_mxfp4(model, train_fq=True)
    print(f"[mxfp4] Replaced {n} Linear layers with Mxfp4Linear (train FQ)")

    train_loop(
        model,
        cfg,
        out_dir=cfg["paths"]["ckpt_mxfp4"],
        log_name="train_mxfp4.jsonl",
        max_steps=args.max_steps,
        pre_save=lambda m: revert_mxfp4_to_linear(m),
    )
    (Path(cfg["paths"]["ckpt_mxfp4"]) / "USE_MXFP4").write_text("fq_train\n")


if __name__ == "__main__":
    main()
