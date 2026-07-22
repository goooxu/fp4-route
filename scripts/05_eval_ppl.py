#!/usr/bin/env python3
"""Evaluate R1/R2/R3: WikiText-2 PPL + smoke generation; write summary."""

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
from mxfp4_lib.replace import count_mxfp4_linears, replace_linears_with_mxfp4
from mxfp4_lib.util import ensure_dir, hf_env, load_cfg, set_seed


def load_model_for_route(path: Path, route: str, device: torch.device):
    model = AutoModelForCausalLM.from_pretrained(path)
    tok = AutoTokenizer.from_pretrained(path)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    use_mx = (path / "USE_MXFP4").exists()
    mode = (path / "USE_MXFP4").read_text().strip() if use_mx else ""

    if route == "R1":
        model = model.to(device=device, dtype=torch.float16)
        infer_dtype = "fp16"
        # Re-tie after dtype cast if needed
        if getattr(model.config, "tie_word_embeddings", False) and hasattr(model, "tie_weights"):
            model.tie_weights()
        assert_default_tie_scope(model, expect_tied=True)
    elif route == "R2":
        n = replace_linears_with_mxfp4(model, train_fq=False, include_lm_head=False)
        model = model.to(device)
        infer_dtype = f"mxfp4_ptq(W4A4_blocks),linears={n}"
        assert_default_tie_scope(model, expect_tied=True)
        on_grid = assert_mxfp4_weights_on_grid(model)
        print(f"[eval] R2 grid-ok linears={on_grid} replaced={n} mx_modules={count_mxfp4_linears(model)}")
    elif route == "R3":
        n = replace_linears_with_mxfp4(model, train_fq=True, include_lm_head=False)
        model = model.to(device)
        infer_dtype = f"mxfp4_fq(W4A4_blocks),linears={n},marker={mode}"
        assert_default_tie_scope(model, expect_tied=True)
    else:
        raise ValueError(route)

    model.eval()
    return model, tok, infer_dtype


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
    max_tokens = cfg["eval"].get("max_eval_tokens")
    if max_tokens:
        input_ids = input_ids[:, : int(max_tokens)]

    nll_sum = 0.0
    n_tokens = 0
    total = input_ids.size(1)
    starts = list(range(0, max(1, total - 1), stride))
    for begin in tqdm(starts, desc="ppl"):
        end = min(begin + seq_len, total)
        chunk = input_ids[:, begin:end]
        if chunk.size(1) < 2:
            continue
        trg_len = end - begin if begin == 0 else end - (begin + max(0, seq_len - stride))
        labels = chunk.clone()
        if begin != 0:
            ignore = chunk.size(1) - trg_len
            labels[:, :ignore] = -100
        # R1: true FP16 forward (no bf16 autocast). R2/R3: bf16 autocast OK.
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


@torch.no_grad()
def smoke_gen(model, tok, prompt: str, max_new: int, device) -> str:
    inputs = tok(prompt, return_tensors="pt").to(device)
    out = model.generate(
        **inputs,
        max_new_tokens=max_new,
        do_sample=False,
        use_cache=False,
        pad_token_id=tok.eos_token_id,
    )
    return tok.decode(out[0], skip_special_tokens=True)


def read_final_train_loss(results_dir: Path, log_name: str):
    path = results_dir / log_name
    if not path.exists():
        return None, None, None
    last_loss = None
    last_val = None
    best_val = None
    with open(path) as f:
        for line in f:
            row = json.loads(line)
            if "loss" in row:
                last_loss = row["loss"]
            if "val_loss" in row:
                last_val = row["val_loss"]
            if "best_val_loss" in row:
                best_val = row["best_val_loss"]
    return last_loss, last_val, best_val


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--routes", default="R1,R2,R3")
    args = ap.parse_args()
    cfg = load_cfg(args.config, seed=args.seed)
    hf_env(cfg)
    set_seed(cfg["seed"])
    set_scale_mode(cfg.get("quant", {}).get("scale_mode", "rtn"))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    results_dir = ensure_dir(cfg["paths"]["results"])

    route_paths = {
        "R1": Path(cfg["paths"]["ckpt_bf16"]),
        "R2": Path(cfg["paths"]["ckpt_bf16_mxfp4_ptq"]),
        "R3": Path(cfg["paths"]["ckpt_mxfp4"]),
    }
    train_logs = {"R1": "train_bf16.jsonl", "R2": "train_bf16.jsonl", "R3": "train_mxfp4.jsonl"}
    train_dtypes = {"R1": "bf16", "R2": "bf16+ptq_blocks", "R3": "mxfp4_fq_blocks"}

    rows = []
    gens = []
    for route in [r.strip() for r in args.routes.split(",") if r.strip()]:
        path = route_paths[route]
        print(f"[eval] {route} from {path}")
        model, tok, infer_dtype = load_model_for_route(path, route, device)
        use_ac = route != "R1"
        ppl = eval_ppl(model, tok, cfg, device, use_autocast=use_ac)
        text = smoke_gen(
            model,
            tok,
            cfg["eval"]["gen_prompt"],
            cfg["eval"]["gen_max_new_tokens"],
            device,
        )
        fl, fv, bv = read_final_train_loss(results_dir, train_logs[route])
        row = {
            "route": route,
            "train_dtype": train_dtypes[route],
            "infer_dtype": infer_dtype,
            "ppl": ppl,
            "train_loss_final": fl,
            "val_loss_final": fv,
            "val_loss_best": bv,
            "seed": cfg["seed"],
        }
        rows.append(row)
        gens.append({"route": route, "prompt": cfg["eval"]["gen_prompt"], "text": text})
        print(f"[eval] {route} ppl={ppl:.4f}")
        del model
        torch.cuda.empty_cache()

    with open(results_dir / "metrics.json", "w") as f:
        json.dump(rows, f, indent=2)
    with open(results_dir / "generations.jsonl", "w") as f:
        for g in gens:
            f.write(json.dumps(g) + "\n")

    arch = cfg.get("arch_model_id", "unknown")
    lines = [
        "# MXFP4 Route Compare Results",
        "",
        f"**Setup**: architecture from `{arch}` via `from_config` (random weights); "
        f"train=FineWeb-Edu; eval=WikiText-2; seed={cfg['seed']}; "
        "quant scope=block Linears only (embed+lm_head BF16).",
        "",
        "| Route | Train | Infer | WikiText-2 PPL | Train loss | Val loss (final/best) |",
        "|-------|-------|-------|----------------|------------|----------------------|",
    ]
    for r in rows:
        lines.append(
            f"| {r['route']} | {r['train_dtype']} | `{r['infer_dtype']}` | {r['ppl']:.4f} | "
            f"{r['train_loss_final']} | {r['val_loss_final']}/{r['val_loss_best']} |"
        )
    lines += [
        "",
        "## Notes",
        "",
        "- R1: BF16 train → **FP16** infer (no bf16 autocast)",
        "- R2: BF16 train → MXFP4 PTQ on **block Linears only** (W4A4)",
        "- R3: MXFP4 fake-quant train (blocks) → MXFP4 infer",
        "- Scope: quality (PPL) only; throughput/latency out of scope.",
        "",
    ]
    summary = "\n".join(lines)
    (results_dir / "summary.md").write_text(summary)
    print(summary)


if __name__ == "__main__":
    main()
