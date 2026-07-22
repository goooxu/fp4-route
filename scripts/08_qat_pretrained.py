#!/usr/bin/env python3
"""R3': QAT-from-pretrained — short MXFP4 fake-quant finetune then MXFP4 infer."""

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

from mxfp4_lib.asserts import assert_default_tie_scope
from mxfp4_lib.quant import set_scale_mode
from mxfp4_lib.replace import replace_linears_with_mxfp4, revert_mxfp4_to_linear
from mxfp4_lib.train_loop import train_loop
from mxfp4_lib.util import ensure_dir, hf_env, load_cfg, set_seed


@torch.no_grad()
def eval_ppl(model, tok, cfg, device) -> float:
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
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
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
    ap.add_argument("--steps", type=int, default=None)
    ap.add_argument("--model-id", default=None)
    args = ap.parse_args()
    cfg = load_cfg(args.config)
    root = cfg["_root"]
    cfg["paths"]["results"] = str(ensure_dir(root / "results" / "qat_pretrained"))
    cfg["paths"]["ckpt_mxfp4"] = str(ensure_dir(root / "checkpoints" / "ckpt_qat_pretrained"))

    hf_env(cfg)
    set_seed(cfg["seed"])
    set_scale_mode(cfg.get("quant", {}).get("scale_mode", "rtn"))
    model_id = args.model_id or cfg["arch_model_id"]
    steps = args.steps or int(cfg.get("qat_steps", 500))

    cfg["train"]["lr"] = float(cfg.get("qat_lr", 1e-5))
    cfg["train"]["min_lr"] = float(cfg.get("qat_lr", 1e-5)) * 0.1
    cfg["train"]["warmup_steps"] = min(50, max(1, steps // 10))
    tokens_per_step = (
        int(cfg["train"]["batch_size"])
        * int(cfg["train"]["grad_accum"])
        * int(cfg["data"]["seq_len"])
    )
    cfg["train"]["target_tokens"] = steps * tokens_per_step
    cfg["eval"]["val_every_steps"] = 0
    cfg["eval"]["val_tokens"] = 0

    print(f"[qat] Loading pretrained {model_id}")
    model = AutoModelForCausalLM.from_pretrained(model_id)
    tok = AutoTokenizer.from_pretrained(model_id)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    init_tok = ensure_dir(root / "checkpoints" / "qat_tok")
    tok.save_pretrained(init_tok)
    cfg["paths"]["init_model"] = str(init_tok)

    n = replace_linears_with_mxfp4(model, train_fq=True, include_lm_head=False)
    print(f"[qat] FQ on {n} block Linears; steps={steps}")

    def _pre_save(m):
        revert_mxfp4_to_linear(m)
        if getattr(m.config, "tie_word_embeddings", False) and hasattr(m, "tie_weights"):
            m.tie_weights()

    meta = train_loop(
        model,
        cfg,
        out_dir=cfg["paths"]["ckpt_mxfp4"],
        log_name="train_qat.jsonl",
        max_steps=steps,
        pre_save=_pre_save,
    )
    (Path(cfg["paths"]["ckpt_mxfp4"]) / "USE_MXFP4").write_text("qat_pretrained\n")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AutoModelForCausalLM.from_pretrained(cfg["paths"]["ckpt_mxfp4"])
    replace_linears_with_mxfp4(model, train_fq=True, include_lm_head=False)
    model.to(device)
    assert_default_tie_scope(model)
    ppl = eval_ppl(model, tok, cfg, device)
    out = {
        "route": "R3_prime_qat_pretrained",
        "model_id": model_id,
        "qat_steps": steps,
        "num_linears": n,
        "ppl": ppl,
        "train_meta": meta,
    }
    out_path = Path(cfg["paths"]["results"]) / "qat_pretrained.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"[qat] WikiText-2 PPL={ppl:.4f} → {out_path}")


if __name__ == "__main__":
    main()
