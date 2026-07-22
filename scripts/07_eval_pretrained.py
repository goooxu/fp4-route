#!/usr/bin/env python3
"""Batch C: evaluate official pretrained SmolLM2 PPL (FP16 + MXFP4 PTQ blocks)."""

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

from mxfp4_lib.asserts import assert_default_tie_scope, assert_mxfp4_weights_on_grid
from mxfp4_lib.quant import set_scale_mode
from mxfp4_lib.replace import pack_all_mxfp4_weights, replace_linears_with_mxfp4
from mxfp4_lib.util import ensure_dir, hf_env, load_cfg, set_seed


@torch.no_grad()
def eval_ppl(model, tok, cfg, device, *, use_autocast: bool) -> float:
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
        with torch.autocast(
            device_type=device.type,
            dtype=torch.bfloat16,
            enabled=bool(use_autocast and device.type == "cuda"),
        ):
            out = model(input_ids=chunk, labels=labels)
        n_valid = (labels != -100).sum().item()
        nll_sum += out.loss.item() * n_valid
        n_tokens += n_valid
        if end >= total:
            break
    return math.exp(nll_sum / max(1, n_tokens))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/main_360m.yaml")
    ap.add_argument("--model-id", default=None)
    ap.add_argument("--scale-mode", default=None, choices=["rtn", "floor"])
    ap.add_argument("--include-lm-head", action="store_true")
    args = ap.parse_args()
    cfg = load_cfg(args.config)
    hf_env(cfg)
    set_seed(cfg["seed"])
    scale_mode = args.scale_mode or cfg.get("quant", {}).get("scale_mode", "rtn")
    set_scale_mode(scale_mode)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_id = args.model_id or cfg["arch_model_id"]
    results_dir = ensure_dir(cfg["_root"] / "results" / "pretrained")

    if not (Path(cfg["paths"]["data_dir"]) / "wikitext2").exists():
        raise SystemExit("Run scripts/01_prepare_data.py first (need WikiText-2 on disk)")

    out = {"model_id": model_id, "scale_mode": scale_mode, "variants": {}}

    print(f"[pretrained] Loading {model_id} FP16...")
    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float16)
    tok = AutoTokenizer.from_pretrained(model_id)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model.to(device)
    if hasattr(model, "tie_weights"):
        model.tie_weights()
    assert_default_tie_scope(model)
    ppl_fp16 = eval_ppl(model, tok, cfg, device, use_autocast=False)
    out["variants"]["fp16"] = {"ppl": ppl_fp16}
    print(f"[pretrained] FP16 PPL={ppl_fp16:.4f}")
    del model
    torch.cuda.empty_cache()

    print(f"[pretrained] PTQ blocks (include_lm_head={args.include_lm_head})...")
    model = AutoModelForCausalLM.from_pretrained(model_id)
    n = replace_linears_with_mxfp4(
        model, train_fq=False, include_lm_head=bool(args.include_lm_head)
    )
    pack_all_mxfp4_weights(model)
    model.to(device)
    if not args.include_lm_head:
        assert_default_tie_scope(model)
        assert_mxfp4_weights_on_grid(model)
    ppl_ptq = eval_ppl(model, tok, cfg, device, use_autocast=True)
    key = "mxfp4_ptq_blocks" if not args.include_lm_head else "mxfp4_ptq_incl_lm_head"
    out["variants"][key] = {
        "ppl": ppl_ptq,
        "num_linears": n,
        "include_lm_head": bool(args.include_lm_head),
    }
    print(f"[pretrained] {key} PPL={ppl_ptq:.4f} linears={n}")

    path = results_dir / "pretrained_baseline.json"
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    legacy = ensure_dir(cfg["_root"] / "results") / "pretrained_baseline.json"
    with open(legacy, "w") as f:
        json.dump(out, f, indent=2)
    print(f"[pretrained] Wrote {path}")


if __name__ == "__main__":
    main()
