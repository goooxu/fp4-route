#!/usr/bin/env python3
"""WikiText-2 PPL for routes R1–R5.

R1: BF16 train → BF16 infer
R2: same BF16 ckpt → TE NVFP4 infer
R3: TE NVFP4 train → TE NVFP4 infer
R4: same BF16 ckpt → TE MXFP8 infer
R5: TE MXFP8 train → TE MXFP8 infer
"""

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
    get_recipe,
    make_te_autocast_ctx,
    replace_linears_with_te,
    te_available,
)
from mxfp4_lib.util import ensure_dir, hf_env, load_cfg, set_seed

# Accept legacy names
_ROUTE_ALIASES = {
    "r1": "R1",
    "bf16": "R1",
    "r2": "R2",
    "nvfp4_ptq": "R2",
    "r3": "R3",
    "nvfp4": "R3",
    "r4": "R4",
    "mxfp8_ptq": "R4",
    "r5": "R5",
    "mxfp8": "R5",
}


def _pad_te_seq(chunk: torch.Tensor, labels: torch.Tensor, *, multiple: int = 32, pad_id: int = 0):
    """Pad seq dim so TE low-precision accepts activations.

    TE constraint is on flattened leading dims of [B, S, H] → (B*S) % block == 0.
    With B=1 this means S % multiple == 0.
    - NVFP4 block is 16
    - MXFP8 block is 32 (stricter)

    Default multiple=32 covers both. Padded labels use -100 (excluded from NLL).
    """
    s = chunk.size(1)
    if s % multiple == 0:
        return chunk, labels
    pad = multiple - (s % multiple)
    chunk = torch.nn.functional.pad(chunk, (0, pad), value=pad_id)
    labels = torch.nn.functional.pad(labels, (0, pad), value=-100)
    return chunk, labels


def _te_seq_multiple(recipe_key: str | None) -> int:
    """Seq-length pad multiple for a TE recipe (NVFP4=16, MXFP8=32)."""
    if not recipe_key:
        return 32
    k = recipe_key.lower()
    if "mxfp8" in k or "fp8" in k:
        return 32
    if "nvfp4" in k or "fp4" in k:
        return 16
    return 32


@torch.no_grad()
def eval_ppl(
    model,
    tok,
    cfg,
    device,
    *,
    te_ctx=None,
    use_fp16: bool = False,
    te_pad_multiple: int = 32,
) -> float:
    from contextlib import nullcontext

    ds = load_from_disk(str(Path(cfg["paths"]["data_dir"]) / "wikitext2"))["test"]
    texts = [t for t in ds["text"] if t and t.strip()]
    eos = tok.eos_token or ""
    big = eos.join(texts)
    enc = tok(big, return_tensors="pt", add_special_tokens=False)
    input_ids = enc["input_ids"].to(device)
    seq_len = cfg["data"]["seq_len"]
    stride = cfg["eval"]["stride"]
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else (tok.eos_token_id or 0)
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
        # TE: pad trailing tokens so B*S is divisible by recipe block size
        if te_ctx is not None:
            chunk, labels = _pad_te_seq(
                chunk, labels, multiple=int(te_pad_multiple), pad_id=pad_id
            )
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
    """Return model, tok, te_ctx, use_fp16, train_dtype, infer_dtype, te_pad_multiple."""
    paths = cfg["paths"]

    if route == "R1":
        path = Path(paths["ckpt_bf16"])
        model = AutoModelForCausalLM.from_pretrained(path, torch_dtype=torch.bfloat16)
        tok = AutoTokenizer.from_pretrained(path)
        model.to(device)
        return model, tok, None, False, "bf16", "bf16", 1

    te_routes = {
        "R2": ("ckpt_bf16", "bf16", "nvfp4"),
        "R3": ("ckpt_nvfp4", "te_nvfp4", "nvfp4"),
        "R4": ("ckpt_bf16", "bf16", "mxfp8"),
        "R5": ("ckpt_mxfp8", "te_mxfp8", "mxfp8"),
    }
    if route in te_routes:
        if not te_available():
            raise SystemExit("Transformer Engine required for R2–R5")
        path_key, train_dtype, recipe_key = te_routes[route]
        path = Path(paths.get(path_key) or "")
        if not path or not path.exists():
            raise SystemExit(f"{route} weights missing: {path} (path key {path_key})")
        model = AutoModelForCausalLM.from_pretrained(path, torch_dtype=torch.bfloat16)
        tok = AutoTokenizer.from_pretrained(path)
        recipe, rname = get_recipe(recipe_key, probe=True)
        n = replace_linears_with_te(model, include_lm_head=False)
        model.to(device)
        te_ctx = make_te_autocast_ctx(recipe)
        infer_dtype = f"te_{rname.replace('BlockScaling', '').lower()}(blocks={n})"
        return model, tok, te_ctx, False, train_dtype, infer_dtype, _te_seq_multiple(recipe_key)

    raise SystemExit(f"unknown route {route!r}; use R1,R2,R3,R4,R5")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/main_360m.yaml")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument(
        "--routes",
        default="R1,R2,R3,R4,R5",
        help="Comma list: R1,R2,R3,R4,R5",
    )
    args = ap.parse_args()
    cfg = load_cfg(args.config, seed=args.seed)
    hf_env(cfg)
    set_seed(cfg["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    results_dir = ensure_dir(Path(cfg["paths"]["results"]))
    metrics = []

    for route_arg in [r.strip() for r in args.routes.split(",") if r.strip()]:
        route = _ROUTE_ALIASES.get(route_arg.lower(), route_arg.upper())
        if route not in ("R1", "R2", "R3", "R4", "R5"):
            raise SystemExit(f"unknown route {route_arg!r}; use R1,R2,R3,R4,R5")
        print(f"[eval] route={route}")
        model, tok, te_ctx, use_fp16, train_dtype, infer_dtype, te_pad = _load_route(
            cfg, route, device
        )
        if te_ctx is not None:
            print(f"[eval] TE seq pad multiple={te_pad}")
        ppl = eval_ppl(
            model,
            tok,
            cfg,
            device,
            te_ctx=te_ctx,
            use_fp16=use_fp16,
            te_pad_multiple=te_pad,
        )
        row = {
            "route": route,
            "train_dtype": train_dtype,
            "infer_dtype": infer_dtype,
            "ppl": ppl,
            "seed": cfg["seed"],
            "te_pad_multiple": te_pad if te_ctx is not None else None,
        }
        metrics.append(row)
        print(f"[eval] {route} ppl={ppl:.4f} train={train_dtype} infer={infer_dtype}")
        del model
        torch.cuda.empty_cache()

    with open(results_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    lines = [
        "# R1–R5 Route Compare (TE NVFP4 + MXFP8)",
        "",
        "| Route | Train | Infer | WikiText-2 PPL |",
        "|-------|-------|-------|----------------|",
    ]
    for m in metrics:
        lines.append(
            f"| {m['route']} | {m['train_dtype']} | `{m['infer_dtype']}` | {m['ppl']:.4f} |"
        )
    lines += [
        "",
        "## Notes",
        "",
        "- **R1**: BF16 train → BF16 infer",
        "- **R2**: same BF16 ckpt → TE NVFP4 infer",
        "- **R3**: TE NVFP4 train → TE NVFP4 infer",
        "- **R4**: same BF16 ckpt → TE MXFP8 infer",
        "- **R5**: TE MXFP8 train → TE MXFP8 infer",
    ]
    (results_dir / "summary.md").write_text("\n".join(lines) + "\n")
    print(f"[eval] wrote {results_dir / 'metrics.json'}")


if __name__ == "__main__":
    main()
