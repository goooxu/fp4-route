#!/usr/bin/env python3
"""Export R3 HF checkpoint from NVFP4 train resume/ (no retrain, CPU-safe).

After a full NVFP4 run, final ``save_pretrained`` can fail in ``revert_te_to_linear``
while ``resume/model_state.pt`` is already on disk. This script loads the resume
state into a plain HF model (shape-compatible tensors only) and writes
``model.safetensors`` for R3 PPL/bench.

Default path is **CPU-only** (no TE, no CUDA) so export works even when GPUs are
busy or TE is unavailable. Optional ``--via-te`` uses replace→load→revert.

Usage (NGC container or host with torch+transformers):
  python scripts/04_export_nvfp4_from_resume.py --config configs/main_360m.yaml --seed 42
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from mxfp4_lib.util import ensure_dir, hf_env, load_cfg, set_seed


def _export_cpu_filtered(model: torch.nn.Module, sd: dict) -> dict:
    """Load shape-compatible non-empty tensors; drop TE-only / empty bias params."""
    base = model.state_dict()
    filtered = {}
    skipped = []
    for k, v in base.items():
        if k not in sd:
            skipped.append((k, "missing_in_resume"))
            continue
        src = sd[k]
        if not torch.is_tensor(src) or src.numel() == 0:
            skipped.append(
                (k, f"empty_or_non_tensor shape={getattr(src, 'shape', None)}")
            )
            continue
        if tuple(src.shape) != tuple(v.shape):
            skipped.append((k, f"shape {tuple(src.shape)} != {tuple(v.shape)}"))
            continue
        filtered[k] = src
    missing, unexpected = model.load_state_dict(filtered, strict=False)
    return {
        "filtered": len(filtered),
        "skipped": len(skipped),
        "missing": len(missing),
        "unexpected": len(unexpected),
        "skipped_samples": skipped[:20],
        "missing_samples": list(missing)[:20],
    }


def _export_via_te(model: torch.nn.Module, sd: dict) -> dict:
    from mxfp4_lib.te_linear import (
        count_te_linears,
        replace_linears_with_te,
        revert_te_to_linear,
        te_available,
    )

    if not te_available():
        raise RuntimeError("TE not available for --via-te")
    # Keep on CPU: te.Linear can be constructed on CPU for weight copy-out
    n = replace_linears_with_te(model, include_lm_head=False)
    missing, unexpected = model.load_state_dict(sd, strict=False)
    n_rev = revert_te_to_linear(model)
    n_left = count_te_linears(model)
    if n_left:
        raise RuntimeError(f"revert left {n_left} te.Linear modules")
    return {
        "te_replaced": n,
        "reverted": n_rev,
        "missing": len(missing),
        "unexpected": len(unexpected),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--resume-dir", default=None)
    ap.add_argument("--out-dir", default=None)
    ap.add_argument(
        "--via-te",
        action="store_true",
        help="Use TE replace→load→revert (needs TE; prefer CPU filter default)",
    )
    args = ap.parse_args()
    cfg = load_cfg(args.config, seed=args.seed)
    hf_env(cfg)
    set_seed(cfg["seed"])

    # Force CPU for default path so we never touch busy GPUs during export.
    if not args.via_te:
        import os

        os.environ["CUDA_VISIBLE_DEVICES"] = ""

    out_dir = Path(args.out_dir or cfg["paths"]["ckpt_nvfp4"])
    resume_dir = Path(args.resume_dir or (out_dir / "resume"))
    ms = resume_dir / "model_state.pt"
    if not ms.exists():
        raise SystemExit(f"missing resume weights: {ms}")

    init_dir = Path(cfg["paths"]["init_model"])
    if not (init_dir / "config.json").exists():
        raise SystemExit(f"missing init model: {init_dir}")

    print(f"[export] init={init_dir}", flush=True)
    print(f"[export] load resume {ms}", flush=True)
    sd = torch.load(ms, map_location="cpu", weights_only=True)
    n_empty = sum(1 for v in sd.values() if torch.is_tensor(v) and v.numel() == 0)
    print(f"[export] state_dict keys={len(sd)} empty_tensors={n_empty}", flush=True)

    model = AutoModelForCausalLM.from_pretrained(init_dir)
    model = model.to(dtype=torch.bfloat16)

    if args.via_te:
        print("[export] path=via_te", flush=True)
        stats = _export_via_te(model, sd)
    else:
        print("[export] path=cpu_shape_filter", flush=True)
        stats = _export_cpu_filtered(model, sd)
    print(f"[export] stats={stats}", flush=True)
    if stats.get("skipped_samples"):
        for item in stats["skipped_samples"][:12]:
            print(f"  skip {item}", flush=True)

    # Coverage check: most non-embed weights should load
    if stats.get("filtered") is not None and stats["filtered"] < 50:
        raise SystemExit(
            f"too few tensors loaded ({stats['filtered']}); refuse incomplete export"
        )

    ensure_dir(out_dir)
    tok_src = resume_dir / "tokenizer"
    if not (tok_src / "tokenizer.json").exists():
        tok_src = init_dir
    tok = AutoTokenizer.from_pretrained(tok_src)

    model.cpu()
    model.save_pretrained(out_dir, safe_serialization=True)
    tok.save_pretrained(out_dir)

    meta = {
        "seed": cfg["seed"],
        "source_resume": str(ms),
        "out_dir": str(out_dir),
        "path": "via_te" if args.via_te else "cpu_shape_filter",
        "stats": {k: v for k, v in stats.items() if k != "skipped_samples"},
        "note": "Exported from NVFP4 train resume for R3 eval/bench.",
    }
    ck = resume_dir / "checkpoint_meta.json"
    if ck.exists():
        meta["resume_meta"] = json.loads(ck.read_text())
    (out_dir / "export_meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    if not (out_dir / "train_meta.json").exists():
        (out_dir / "train_meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    (out_dir / "USE_NVFP4").write_text("NVFP4BlockScaling\n")
    safetensors = out_dir / "model.safetensors"
    print(f"[export] wrote HF ckpt to {out_dir}", flush=True)
    print(f"[export] model.safetensors exists={safetensors.exists()} size={safetensors.stat().st_size if safetensors.exists() else 0}", flush=True)


if __name__ == "__main__":
    main()
