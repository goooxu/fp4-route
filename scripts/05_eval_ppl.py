#!/usr/bin/env python3
"""WikiText-2 PPL for BF16 and TE NVFP4 checkpoints (no software fake-quant)."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from datasets import load_from_disk
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from mxfp4_lib.te_linear import (
    get_preferred_recipe,
    make_te_autocast_ctx,
    replace_linears_with_te,
    te_available,
)
from mxfp4_lib.util import ensure_dir, hf_env, load_cfg, set_seed


@torch.no_grad()
def eval_ppl(model, tok, cfg, device, *, te_ctx=None, use_fp16: bool = False) -> float:
    from contextlib import nullcontext

    ds = load_from_disk(str(Path(cfg["paths"]["data_dir"]) / "wikitext2"))["test"]
    texts = [t for t in ds["text"] if t and t.strip()]
    eos = tok.eos_token or ""
    big = eos.join(texts)
    enc = tok(big, return_tensors="pt", add_special_tokens=False)
    input_ids = enc["input_ids"].to(device)
    seq_len = cfg["data"]["seq_len"]
    stride = cfg["eval"]["stride"]
    nll_sum = 0.0
    n_tokens = 0
    total = input_ids.size(1)
    for begin in tqdm(list(range(0, max(1, total - 1), stride)), desc="ppl"):
        end = min(begin + seq_len, total)
        chunk = input_ids[:, begin:end]
        if chunk.size(1) < 2:
            continue
        trg_len = end - begin if begin == 0 else end - (begin + max(0, seq_len - stride))
        labels = chunk.clone()
        if begin != 0:
            labels[:, : chunk.size(1) - trg_len] = -100
        amp = torch.autocast(
            device_type=device.type,
            dtype=torch.float16 if use_fp16 else torch.bfloat16,
            enabled=te_ctx is None and device.type == "cuda",
        )
        with amp:
            with te_ctx() if te_ctx is not None else nullcontext():
                out = model(input_ids=chunk, labels=labels)
        n_valid = (labels != -100).sum().item()
        nll_sum += out.loss.item() * n_valid
        n_tokens += n_valid
        if end >= total:
            break
    return math.exp(nll_sum / max(1, n_tokens))


def _load_route(cfg, route: str, device: torch.device):
    paths = cfg["paths"]
    te_ctx = None
    infer_dtype = route
    if route == "bf16":
        path = Path(paths["ckpt_bf16"])
        model = AutoModelForCausalLM.from_pretrained(path, torch_dtype=torch.float16)
        tok = AutoTokenizer.from_pretrained(path)
        model.to(device)
        use_fp16 = True
        train_dtype = "bf16"
        infer_dtype = "fp16"
    elif route == "nvfp4":
        if not te_available():
            raise SystemExit("TE required for nvfp4 eval")
        path = Path(paths.get("ckpt_nvfp4") or paths.get("ckpt_mxfp4", ""))
        if not path or not path.exists():
            raise SystemExit(f"nvfp4 ckpt missing: {path}")
        model = AutoModelForCausalLM.from_pretrained(path, torch_dtype=torch.bfloat16)
        tok = AutoTokenizer.from_pretrained(path)
        recipe, rname = get_preferred_recipe()
        n = replace_linears_with_te(model, include_lm_head=False)
        model.to(device)
        te_ctx = make_te_autocast_ctx(recipe)
        use_fp16 = False
        train_dtype = f"te_{rname}"
        infer_dtype = f"te_nvfp4(linears={n})"
    else:
        raise SystemExit(f"unknown route {route}")
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return model, tok, te_ctx, use_fp16, train_dtype, infer_dtype


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/main_360m.yaml")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument(
        "--routes",
        default="bf16,nvfp4",
        help="Comma list: bf16,nvfp4",
    )
    args = ap.parse_args()
    cfg = load_cfg(args.config, seed=args.seed)
    hf_env(cfg)
    set_seed(cfg["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    results_dir = ensure_dir(Path(cfg["paths"]["results"]))
    metrics = []

    for route in [r.strip() for r in args.routes.split(",") if r.strip()]:
        print(f"[eval] route={route}")
        model, tok, te_ctx, use_fp16, train_dtype, infer_dtype = _load_route(cfg, route, device)
        ppl = eval_ppl(model, tok, cfg, device, te_ctx=te_ctx, use_fp16=use_fp16)
        row = {
            "route": route,
            "train_dtype": train_dtype,
            "infer_dtype": infer_dtype,
            "ppl": ppl,
            "seed": cfg["seed"],
        }
        metrics.append(row)
        print(f"[eval] {route} ppl={ppl:.4f}")
        del model
        torch.cuda.empty_cache()

    with open(results_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    lines = [
        "# Eval results (hardware TE path; no software fake-quant)",
        "",
        "| Route | Train | Infer | WikiText-2 PPL |",
        "|-------|-------|-------|----------------|",
    ]
    for m in metrics:
        lines.append(
            f"| {m['route']} | {m['train_dtype']} | {m['infer_dtype']} | {m['ppl']:.4f} |"
        )
    (results_dir / "summary.md").write_text("\n".join(lines) + "\n")
    print(f"[eval] wrote {results_dir / 'metrics.json'}")


if __name__ == "__main__":
    main()
