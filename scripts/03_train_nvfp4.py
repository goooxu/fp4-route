#!/usr/bin/env python3
"""R3: from-scratch NVFP4 training via Transformer Engine (block Linears only).

Requires TE (use NGC: nvcr.io/nvidia/pytorch:26.06-py3). No software fake-quant.
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
    get_preferred_recipe,
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
    args = ap.parse_args()
    cfg = load_cfg(args.config, seed=args.seed)
    hf_env(cfg)
    set_seed(cfg["seed"])

    if not te_available():
        raise SystemExit(
            "transformer_engine not available. Run inside "
            "nvcr.io/nvidia/pytorch:26.06-py3 (or install TE with CUDA support)."
        )

    init_dir = cfg["paths"]["init_model"]
    print(f"[nvfp4] Loading shared init from {init_dir}")
    model = AutoModelForCausalLM.from_pretrained(init_dir)
    model = model.to(dtype=torch.bfloat16)
    recipe, recipe_name = get_preferred_recipe()
    n = replace_linears_with_te(model, include_lm_head=False)
    print(f"[nvfp4] Replaced {n} block Linears with te.Linear; recipe={recipe_name}")
    te_ctx = make_te_autocast_ctx(recipe)

    out_dir = cfg["paths"].get("ckpt_nvfp4")
    if not out_dir:
        out_dir = str(Path(cfg["paths"]["init_model"]).parent / "ckpt_nvfp4")
        cfg["paths"]["ckpt_nvfp4"] = out_dir

    def pre_save(raw):
        # TE often registers empty bias Parameters when bias=False; revert handles that.
        n_rev = revert_te_to_linear(raw)
        print(f"[nvfp4] pre_save: reverted {n_rev} te.Linear → nn.Linear for HF save")
        n_left = count_te_linears(raw)
        if n_left:
            raise RuntimeError(f"pre_save left {n_left} te.Linear modules unreverted")

    train_loop(
        model,
        cfg,
        out_dir=out_dir,
        log_name="train_nvfp4.jsonl",
        max_steps=args.max_steps,
        pre_save=pre_save,
        te_ctx=te_ctx,
    )
    marker = Path(out_dir) / "USE_NVFP4"
    marker.parent.mkdir(parents=True, exist_ok=True)
    marker.write_text(f"{recipe_name}\n")


if __name__ == "__main__":
    main()
