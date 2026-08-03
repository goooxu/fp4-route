#!/usr/bin/env python3
"""TE low-precision training for R3 (NVFP4) or R5 (MXFP8).

Block Linears only. Requires NGC pytorch with Transformer Engine
(e.g. nvcr.io/nvidia/pytorch:26.07-py3).

  # R3
  torchrun --nproc_per_node=4 scripts/03_train_nvfp4.py --seed 42 --recipe nvfp4
  # R5
  torchrun --nproc_per_node=4 scripts/03_train_nvfp4.py --seed 42 --recipe mxfp8
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from transformers import AutoModelForCausalLM

from mxfp4_lib.te_linear import (
    count_te_linears,
    get_recipe,
    make_te_autocast_ctx,
    replace_linears_with_te,
    revert_te_to_linear,
    te_available,
)
from mxfp4_lib.train_loop import train_loop
from mxfp4_lib.util import hf_env, load_cfg, set_seed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--max_steps", type=int, default=None)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument(
        "--recipe",
        default="nvfp4",
        choices=["nvfp4", "mxfp8"],
        help="TE recipe: nvfp4 (R3) or mxfp8 (R5)",
    )
    args = ap.parse_args()
    cfg = load_cfg(args.config, seed=args.seed)
    hf_env(cfg)
    set_seed(cfg["seed"])

    if not te_available():
        raise SystemExit(
            "transformer_engine not available. Run inside "
            "nvcr.io/nvidia/pytorch:26.07-py3 (or install TE with CUDA support)."
        )

    recipe_key = args.recipe.lower()
    if recipe_key == "mxfp8":
        out_key = "ckpt_mxfp8"
        log_name = "train_mxfp8.jsonl"
        marker_name = "USE_MXFP8"
        tag = "mxfp8"
    else:
        out_key = "ckpt_nvfp4"
        log_name = "train_nvfp4.jsonl"
        marker_name = "USE_NVFP4"
        tag = "nvfp4"

    init_dir = cfg["paths"]["init_model"]
    print(f"[{tag}] Loading shared init from {init_dir}")
    model = AutoModelForCausalLM.from_pretrained(init_dir)
    model = model.to(dtype=torch.bfloat16)
    recipe, recipe_name = get_recipe(recipe_key, probe=True)
    n = replace_linears_with_te(model, include_lm_head=False)
    print(f"[{tag}] Replaced {n} block Linears with te.Linear; recipe={recipe_name}")
    te_ctx = make_te_autocast_ctx(recipe)

    out_dir = cfg["paths"].get(out_key)
    if not out_dir:
        out_dir = str(Path(cfg["paths"]["init_model"]).parent / out_key.replace("ckpt_", "ckpt_"))
        cfg["paths"][out_key] = out_dir

    def pre_save(raw):
        n_rev = revert_te_to_linear(raw)
        print(f"[{tag}] pre_save: reverted {n_rev} te.Linear → nn.Linear for HF save")
        n_left = count_te_linears(raw)
        if n_left:
            raise RuntimeError(f"pre_save left {n_left} te.Linear modules unreverted")

    train_loop(
        model,
        cfg,
        out_dir=out_dir,
        log_name=log_name,
        max_steps=args.max_steps,
        pre_save=pre_save,
        te_ctx=te_ctx,
    )
    marker = Path(out_dir) / marker_name
    marker.parent.mkdir(parents=True, exist_ok=True)
    marker.write_text(f"{recipe_name}\n")


if __name__ == "__main__":
    main()
