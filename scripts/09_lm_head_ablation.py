#!/usr/bin/env python3
"""lm_head ablation on pretrained weights: FP16 / PTQ blocks / PTQ + untied lm_head."""

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
    args = ap.parse_args()
    cfg = load_cfg(args.config)
    hf_env(cfg)
    set_seed(cfg["seed"])
    set_scale_mode(cfg.get("quant", {}).get("scale_mode", "rtn"))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_id = args.model_id or cfg["arch_model_id"]
    out_dir = ensure_dir(cfg["_root"] / "results" / "ablations")

    rows = []

    # FP16
    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float16).to(device)
    tok = AutoTokenizer.from_pretrained(model_id)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    if hasattr(model, "tie_weights"):
        model.tie_weights()
    assert_default_tie_scope(model)
    ppl = eval_ppl(model, tok, cfg, device, use_autocast=False)
    rows.append({"variant": "fp16", "ppl": ppl, "num_linears": 0, "include_lm_head": False})
    print(f"[ablate] fp16 ppl={ppl:.4f}")
    del model
    torch.cuda.empty_cache()

    # PTQ blocks only
    model = AutoModelForCausalLM.from_pretrained(model_id)
    n = replace_linears_with_mxfp4(model, train_fq=False, include_lm_head=False)
    pack_all_mxfp4_weights(model)
    model.to(device)
    assert_default_tie_scope(model)
    assert_mxfp4_weights_on_grid(model)
    ppl = eval_ppl(model, tok, cfg, device, use_autocast=True)
    rows.append({"variant": "ptq_blocks", "ppl": ppl, "num_linears": n, "include_lm_head": False})
    print(f"[ablate] ptq_blocks ppl={ppl:.4f} n={n}")
    del model
    torch.cuda.empty_cache()

    # PTQ + untied lm_head
    model = AutoModelForCausalLM.from_pretrained(model_id)
    n = replace_linears_with_mxfp4(model, train_fq=False, include_lm_head=True)
    pack_all_mxfp4_weights(model)
    model.to(device)
    ppl = eval_ppl(model, tok, cfg, device, use_autocast=True)
    rows.append({"variant": "ptq_incl_lm_head", "ppl": ppl, "num_linears": n, "include_lm_head": True})
    print(f"[ablate] ptq_incl_lm_head ppl={ppl:.4f} n={n}")

    out = {"model_id": model_id, "rows": rows}
    path = out_dir / "lm_head_ablation.json"
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    md = [
        "# lm_head ablation (pretrained)",
        "",
        f"Model: `{model_id}`",
        "",
        "| Variant | PPL | # Mxfp4Linear |",
        "|---------|-----|---------------|",
    ]
    for r in rows:
        md.append(f"| {r['variant']} | {r['ppl']:.4f} | {r['num_linears']} |")
    (out_dir / "lm_head_ablation.md").write_text("\n".join(md) + "\n")
    print(f"[ablate] Wrote {path}")


if __name__ == "__main__":
    main()
