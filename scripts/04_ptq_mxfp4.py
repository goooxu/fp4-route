#!/usr/bin/env python3
"""Route 2: MXFP4 PTQ on BF16 checkpoint (block Linears only; embed+lm_head BF16)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from transformers import AutoModelForCausalLM, AutoTokenizer

from mxfp4_lib.quant import set_scale_mode
from mxfp4_lib.replace import (
    pack_all_mxfp4_weights,
    replace_linears_with_mxfp4,
    revert_mxfp4_to_linear,
)
from mxfp4_lib.util import ensure_dir, hf_env, load_cfg, set_seed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--include-lm-head", action="store_true")
    ap.add_argument("--scale-mode", default=None, choices=["rtn", "floor"])
    args = ap.parse_args()
    cfg = load_cfg(args.config, seed=args.seed)
    hf_env(cfg)
    set_seed(cfg["seed"])
    scale_mode = args.scale_mode or cfg.get("quant", {}).get("scale_mode", "rtn")
    set_scale_mode(scale_mode)

    src = cfg["paths"]["ckpt_bf16"]
    dst = ensure_dir(cfg["paths"]["ckpt_bf16_mxfp4_ptq"])
    print(f"[ptq] Loading BF16 ckpt from {src}")
    model = AutoModelForCausalLM.from_pretrained(src)
    tok = AutoTokenizer.from_pretrained(src)

    include_lm = bool(args.include_lm_head or cfg.get("quant", {}).get("include_lm_head", False))
    n = replace_linears_with_mxfp4(model, train_fq=False, include_lm_head=include_lm)
    packed = pack_all_mxfp4_weights(model)
    print(
        f"[ptq] Replaced {n} block Linears (include_lm_head={include_lm}); "
        f"packed {packed}; scale_mode={scale_mode}; embed+lm_head stay BF16 unless include_lm_head"
    )
    revert_mxfp4_to_linear(model)

    # Re-tie after revert if we did not include lm_head (HF may have kept tie)
    if not include_lm and getattr(model.config, "tie_word_embeddings", False):
        if hasattr(model, "tie_weights"):
            model.tie_weights()

    model.save_pretrained(dst, safe_serialization=True)
    tok.save_pretrained(dst)
    meta = {
        "route": "R2",
        "quant": "mxfp4_ptq_blocks",
        "num_linears": n,
        "include_lm_head": include_lm,
        "scale_mode": scale_mode,
        "note": "Default: quantize transformer-block Linears only; embed+lm_head BF16",
    }
    with open(Path(dst) / "quant_meta.json", "w") as f:
        json.dump(meta, f, indent=2)
    (Path(dst) / "USE_MXFP4").write_text("ptq\n")
    print(f"[ptq] Saved to {dst}")


if __name__ == "__main__":
    main()
