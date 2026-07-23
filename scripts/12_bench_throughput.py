#!/usr/bin/env python3
"""Throughput microbench: bf16 / software MXFP4 FQ / TE hardware FP4 (if available).

Examples:
  python scripts/12_bench_throughput.py --backend bf16 --phase train
  python scripts/12_bench_throughput.py --backend sw_fq --phase infer --batch-size 64
  python scripts/12_bench_throughput.py --backend te_fp4 --phase train
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from transformers import AutoModelForCausalLM

from mxfp4_lib.bench import (
    make_infer_step,
    make_lm_batch,
    make_train_step,
    run_bench,
    write_result,
)
from mxfp4_lib.model_config import build_model_from_arch
from mxfp4_lib.quant import set_scale_mode
from mxfp4_lib.replace import replace_linears_with_mxfp4
from mxfp4_lib.util import ensure_dir, hf_env, load_cfg, set_seed


def _build_model(cfg, backend: str, device: torch.device):
    """Build or load a 360M causal LM for the given backend."""
    init = Path(cfg["paths"].get("init_model", "checkpoints/init_model"))
    ckpt_bf16 = Path(cfg["paths"].get("ckpt_bf16", ""))
    notes = []

    if ckpt_bf16 and ckpt_bf16.exists() and (ckpt_bf16 / "config.json").exists():
        print(f"[bench] load weights from {ckpt_bf16}")
        model = AutoModelForCausalLM.from_pretrained(ckpt_bf16)
        notes.append(f"weights={ckpt_bf16}")
    elif init.exists() and (init / "config.json").exists():
        print(f"[bench] load init from {init}")
        model = AutoModelForCausalLM.from_pretrained(init)
        notes.append(f"weights={init}")
    else:
        print("[bench] from_config random init (no ckpt on disk)")
        model, _ = build_model_from_arch(cfg)
        notes.append("weights=from_config_random")

    te_ctx = None
    recipe_name = None

    if backend == "bf16":
        model = model.to(device=device, dtype=torch.bfloat16)
    elif backend == "fp16":
        model = model.to(device=device, dtype=torch.float16)
    elif backend == "sw_fq":
        model = model.to(device=device, dtype=torch.bfloat16)
        n = replace_linears_with_mxfp4(model, train_fq=True, include_lm_head=False)
        notes.append(f"sw_fq_linears={n}")
    elif backend == "te_fp4":
        from mxfp4_lib.te_linear import (
            get_preferred_recipe,
            make_te_autocast_ctx,
            replace_linears_with_te,
        )

        model = model.to(device=device, dtype=torch.bfloat16)
        recipe, recipe_name = get_preferred_recipe()
        n = replace_linears_with_te(model, include_lm_head=False)
        notes.append(f"te_linears={n},recipe={recipe_name}")
        te_ctx = make_te_autocast_ctx(recipe)
    else:
        raise SystemExit(f"Unknown backend: {backend}")

    return model, te_ctx, ";".join(notes), recipe_name


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/bench_360m.yaml")
    ap.add_argument(
        "--backend",
        required=True,
        choices=["bf16", "fp16", "sw_fq", "te_fp4"],
    )
    ap.add_argument("--phase", required=True, choices=["train", "infer"])
    ap.add_argument("--batch-size", type=int, default=None)
    ap.add_argument("--seq-len", type=int, default=None)
    ap.add_argument("--warmup", type=int, default=None)
    ap.add_argument("--measure", type=int, default=None)
    ap.add_argument("--out", default=None, help="JSON path under results/perf")
    args = ap.parse_args()

    cfg = load_cfg(args.config)
    hf_env(cfg)
    set_seed(cfg.get("seed", 42))
    set_scale_mode(cfg.get("quant", {}).get("scale_mode", "rtn"))

    if not torch.cuda.is_available():
        raise SystemExit("CUDA required for throughput bench")

    device = torch.device("cuda")
    tcfg = cfg.get("train") or {}
    bcfg = cfg.get("bench") or {}
    if tcfg.get("allow_tf32", True):
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    if tcfg.get("cudnn_benchmark", True):
        torch.backends.cudnn.benchmark = True

    batch = int(args.batch_size or tcfg.get("batch_size", 64))
    seq = int(args.seq_len or cfg.get("data", {}).get("seq_len", 512))
    warmup = int(args.warmup if args.warmup is not None else bcfg.get("warmup", 10))
    measure = int(args.measure if args.measure is not None else bcfg.get("measure", 40))

    model, te_ctx, notes, recipe_name = _build_model(cfg, args.backend, device)
    # vocab from config
    vocab = int(getattr(model.config, "vocab_size", 49152))
    batch_t = make_lm_batch(batch, seq, vocab, device)

    if args.phase == "train":
        model.train()
        opt = torch.optim.AdamW(model.parameters(), lr=float(tcfg.get("lr", 3e-4)), weight_decay=0.0)
        amp = torch.bfloat16 if args.backend == "bf16" else None
        # sw_fq / te already mixed; don't double-autocast bf16 for te
        if args.backend in ("sw_fq", "te_fp4", "fp16"):
            amp = None
        step_fn = make_train_step(model, opt, batch_t, amp_dtype=amp, te_ctx=te_ctx)
    else:
        model.eval()
        step_fn = make_infer_step(
            model,
            batch_t,
            use_fp16=(args.backend == "fp16"),
            te_ctx=te_ctx,
        )

    backend_label = args.backend
    if args.backend == "te_fp4" and recipe_name:
        backend_label = f"te_{recipe_name.replace('BlockScaling', '').lower()}"

    result = run_bench(
        backend=backend_label,
        phase=args.phase,
        step_fn=step_fn,
        batch_size=batch,
        seq_len=seq,
        nproc=1,
        warmup=warmup,
        measure=measure,
        notes=notes,
        extra={"recipe": recipe_name, "config": args.config},
    )

    out_dir = ensure_dir(Path(cfg["paths"]["results"]))
    out_path = Path(args.out) if args.out else out_dir / f"bench_{backend_label}_{args.phase}_bs{batch}.json"
    write_result(out_path, result)


if __name__ == "__main__":
    main()
