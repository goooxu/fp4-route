#!/usr/bin/env python3
"""Route 2: full-model MXFP4 PTQ on BF16 checkpoint (all Linear incl. lm_head)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from transformers import AutoModelForCausalLM, AutoTokenizer

from mxfp4_lib.replace import (
    pack_all_mxfp4_weights,
    replace_linears_with_mxfp4,
    revert_mxfp4_to_linear,
)
from mxfp4_lib.util import ensure_dir, hf_env, load_cfg, set_seed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    args = ap.parse_args()
    cfg = load_cfg(args.config)
    hf_env(cfg)
    set_seed(cfg["seed"])

    src = cfg["paths"]["ckpt_bf16"]
    dst = ensure_dir(cfg["paths"]["ckpt_bf16_mxfp4_ptq"])
    print(f"[ptq] Loading BF16 ckpt from {src}")
    model = AutoModelForCausalLM.from_pretrained(src)
    tok = AutoTokenizer.from_pretrained(src)

    n = replace_linears_with_mxfp4(model, train_fq=False)
    packed = pack_all_mxfp4_weights(model)
    print(f"[ptq] Replaced {n} layers; packed {packed} weight tensors to MXFP4")
    revert_mxfp4_to_linear(model)

    model.save_pretrained(dst, safe_serialization=True)
    tok.save_pretrained(dst)
    meta = {"route": "R2", "quant": "mxfp4_ptq_full", "num_linears": n}
    with open(Path(dst) / "quant_meta.json", "w") as f:
        json.dump(meta, f, indent=2)
    (Path(dst) / "USE_MXFP4").write_text("ptq\n")
    print(f"[ptq] Saved to {dst}")


if __name__ == "__main__":
    main()
