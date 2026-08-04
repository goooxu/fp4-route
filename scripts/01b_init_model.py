#!/usr/bin/env python3
"""Create shared randomly-initialized model from open-source architecture only."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from transformers import AutoTokenizer

from mxfp4_lib.model_config import build_model_from_arch, count_parameters
from mxfp4_lib.util import ensure_dir, hf_env, load_cfg, set_seed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--seed", type=int, default=None)
    args = ap.parse_args()
    cfg = load_cfg(args.config, seed=args.seed)
    hf_env(cfg)
    set_seed(cfg["seed"])

    tok_dir = Path(cfg["paths"]["data_dir"]) / "tokenizer"
    tok = AutoTokenizer.from_pretrained(tok_dir)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    arch_id = cfg["arch_model_id"]
    print(f"[init] Architecture from config only: {arch_id}")
    print("[init] Random init via from_config — NOT loading pretrained weights")
    model, model_cfg = build_model_from_arch(cfg, vocab_size=len(tok))
    n = count_parameters(model)
    out = ensure_dir(cfg["paths"]["init_model"])
    model.save_pretrained(out, safe_serialization=True)
    tok.save_pretrained(out)
    # NFS root_squash: make weights readable by host user / later containers
    try:
        import os
        for p in Path(out).iterdir():
            try:
                os.chmod(p, 0o666 if p.is_file() else 0o777)
            except OSError:
                pass
    except Exception:
        pass

    meta = {
        "seed": cfg["seed"],
        "arch_model_id": arch_id,
        "num_parameters": n,
        "vocab_size": len(tok),
        "tie_word_embeddings": getattr(model_cfg, "tie_word_embeddings", None),
        "note": (
            "Open-source architecture via AutoConfig.from_pretrained; "
            "weights via AutoModelForCausalLM.from_config (random). "
            "No pretrained .safetensors / pytorch_model.bin loaded."
        ),
        "model_type": getattr(model_cfg, "model_type", None),
        "hidden_size": getattr(model_cfg, "hidden_size", None),
        "num_hidden_layers": getattr(model_cfg, "num_hidden_layers", None),
        "intermediate_size": getattr(model_cfg, "intermediate_size", None),
        "num_attention_heads": getattr(model_cfg, "num_attention_heads", None),
    }
    with open(Path(out) / "init_meta.json", "w") as f:
        json.dump(meta, f, indent=2)
    # Also mirror into results for convenience
    res = ensure_dir(cfg["paths"]["results"])
    with open(Path(res) / "init_meta.json", "w") as f:
        json.dump(meta, f, indent=2)
    print(f"[init] Saved random init model to {out} ({n/1e6:.2f}M params)")


if __name__ == "__main__":
    main()
