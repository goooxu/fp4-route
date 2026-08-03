#!/usr/bin/env python3
"""Generate qualitative text samples for R1 / R2 / R3 (for report).

Usage (NGC container, 1 GPU):
  python scripts/13_generate_samples.py --seed 42 --max-new-tokens 80
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from mxfp4_lib.te_linear import (
    get_preferred_recipe,
    make_te_autocast_ctx,
    replace_linears_with_te,
    te_available,
)
from mxfp4_lib.util import ensure_dir, hf_env, load_cfg, set_seed

DEFAULT_PROMPTS = [
    "The history of artificial intelligence began",
    "In mathematics, a prime number is",
    "Photosynthesis is the process by which",
    "The capital of France is",
    "Once upon a time, in a small village,",
]


def _pad_to_multiple(ids: torch.Tensor, attn: torch.Tensor, *, multiple: int, pad_id: int):
    """Pad seq dim so B*S is divisible by `multiple` (TE NVFP4 / FP8 constraint)."""
    s = ids.size(1)
    if s % multiple == 0:
        return ids, attn
    pad = multiple - (s % multiple)
    ids = torch.nn.functional.pad(ids, (0, pad), value=pad_id)
    attn = torch.nn.functional.pad(attn, (0, pad), value=0)
    return ids, attn


@torch.no_grad()
def _sample_next(logits_last: torch.Tensor, *, temperature: float, top_p: float) -> torch.Tensor:
    """Sample one token from last-position logits [B, V]."""
    logits = logits_last / max(temperature, 1e-5)
    probs = torch.softmax(logits, dim=-1)
    # nucleus (top-p)
    sorted_probs, sorted_idx = torch.sort(probs, descending=True, dim=-1)
    cumsum = torch.cumsum(sorted_probs, dim=-1)
    mask = cumsum > top_p
    mask[..., 0] = False
    sorted_probs = sorted_probs.masked_fill(mask, 0.0)
    sorted_probs = sorted_probs / sorted_probs.sum(dim=-1, keepdim=True).clamp_min(1e-12)
    pick = torch.multinomial(sorted_probs, num_samples=1)
    return sorted_idx.gather(-1, pick)


@torch.no_grad()
def generate_one(model, tok, prompt: str, *, max_new_tokens: int, te_ctx=None, dtype_name: str):
    """Greedy/sample generate.

    TE NVFP4 requires B*S divisible by 16 on each forward; HF ``generate`` can hit
    illegal intermediate lengths, so R2/R3 use a padded step-by-step loop.
    """
    from contextlib import nullcontext

    device = next(model.parameters()).device
    enc = tok(prompt, return_tensors="pt")
    input_ids = enc["input_ids"].to(device)
    attn = enc.get("attention_mask")
    if attn is None:
        attn = torch.ones_like(input_ids)
    else:
        attn = attn.to(device)
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else (tok.eos_token_id or 0)
    eos_id = tok.eos_token_id
    prompt_len = input_ids.size(1)

    if te_ctx is None:
        with nullcontext():
            out = model.generate(
                input_ids=input_ids,
                attention_mask=attn,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=0.8,
                top_p=0.9,
                pad_token_id=pad_id,
                eos_token_id=eos_id,
            )
        cont = tok.decode(out[0][prompt_len:], skip_special_tokens=True)
        full = tok.decode(out[0], skip_special_tokens=True)
        return {"prompt": prompt, "continuation": cont.strip(), "full": full.strip(), "infer": dtype_name}

    # TE path: pad every forward so leading dims product (B*S) % 16 == 0
    ids = input_ids
    mask = attn
    for _ in range(max_new_tokens):
        ids_p, mask_p = _pad_to_multiple(ids, mask, multiple=16, pad_id=pad_id)
        with te_ctx():
            logits = model(input_ids=ids_p, attention_mask=mask_p).logits
        # last *real* token position (before pad)
        last = ids.size(1) - 1
        next_id = _sample_next(logits[:, last, :], temperature=0.8, top_p=0.9)
        ids = torch.cat([ids, next_id], dim=1)
        mask = torch.cat([mask, torch.ones((mask.size(0), 1), device=device, dtype=mask.dtype)], dim=1)
        if eos_id is not None and int(next_id.item()) == int(eos_id):
            break

    cont = tok.decode(ids[0][prompt_len:], skip_special_tokens=True)
    full = tok.decode(ids[0], skip_special_tokens=True)
    return {"prompt": prompt, "continuation": cont.strip(), "full": full.strip(), "infer": dtype_name}


def load_route(cfg, route: str, device: torch.device):
    paths = cfg["paths"]
    if route == "R1":
        path = Path(paths["ckpt_bf16"])
        model = AutoModelForCausalLM.from_pretrained(path, torch_dtype=torch.bfloat16)
        tok = AutoTokenizer.from_pretrained(path)
        model.to(device).eval()
        return model, tok, None, "bf16"

    if not te_available():
        raise SystemExit("TE required for R2/R3")
    if route == "R2":
        path = Path(paths["ckpt_bf16"])
        train_note = "bf16_ckpt"
    else:
        path = Path(paths["ckpt_nvfp4"])
        train_note = "nvfp4_ckpt"
    model = AutoModelForCausalLM.from_pretrained(path, torch_dtype=torch.bfloat16)
    tok = AutoTokenizer.from_pretrained(path)
    model.to(device)
    recipe, rname = get_preferred_recipe()
    n = replace_linears_with_te(model, include_lm_head=False)
    model.eval()
    te_ctx = make_te_autocast_ctx(recipe)
    return model, tok, te_ctx, f"te_{rname}({train_note},blocks={n})"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/main_360m.yaml")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--routes", default="R1,R2,R3")
    ap.add_argument("--max-new-tokens", type=int, default=80)
    ap.add_argument("--gen-seed", type=int, default=0, help="sampling RNG seed for reproducibility")
    args = ap.parse_args()

    cfg = load_cfg(args.config, seed=args.seed)
    hf_env(cfg)
    set_seed(args.gen_seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise SystemExit("CUDA required")

    prompts = list(DEFAULT_PROMPTS)
    # optional config prompt first
    gen_prompt = (cfg.get("eval") or {}).get("gen_prompt")
    if gen_prompt and gen_prompt not in prompts:
        prompts = [gen_prompt] + prompts

    # load_cfg with isolate_seed_paths already points results at .../seed_<N>
    res = Path(cfg["paths"]["results"])
    if res.name == f"seed_{args.seed}":
        out_dir = ensure_dir(res)
    else:
        out_dir = ensure_dir(res / f"seed_{args.seed}")
    all_rows = []

    for route in [r.strip().upper() for r in args.routes.split(",") if r.strip()]:
        print(f"[gen] loading {route} ...", flush=True)
        model, tok, te_ctx, infer = load_route(cfg, route, device)
        if tok.pad_token_id is None:
            tok.pad_token = tok.eos_token
        for p in prompts:
            torch.manual_seed(args.gen_seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(args.gen_seed)
            row = generate_one(
                model, tok, p, max_new_tokens=args.max_new_tokens, te_ctx=te_ctx, dtype_name=infer
            )
            row["route"] = route
            row["model_seed"] = args.seed
            row["gen_seed"] = args.gen_seed
            all_rows.append(row)
            print(f"\n=== {route} | {infer} ===\nPROMPT: {p}\nOUTPUT: {row['continuation']}\n", flush=True)
        del model
        torch.cuda.empty_cache()

    json_path = out_dir / "generation_samples.json"
    with open(json_path, "w") as f:
        json.dump(all_rows, f, indent=2, ensure_ascii=False)

    # markdown table-friendly dump
    md_lines = [
        f"# Generation samples (model seed={args.seed}, gen_seed={args.gen_seed}, max_new_tokens={args.max_new_tokens})",
        "",
        "Sampling: `do_sample=True`, temperature=0.8, top_p=0.9.",
        "",
    ]
    # group by prompt
    by_prompt = {}
    for r in all_rows:
        by_prompt.setdefault(r["prompt"], []).append(r)
    for prompt, rows in by_prompt.items():
        md_lines.append(f"## Prompt")
        md_lines.append("")
        md_lines.append(f"> {prompt}")
        md_lines.append("")
        for r in rows:
            md_lines.append(f"**{r['route']}** (`{r['infer']}`):")
            md_lines.append("")
            md_lines.append(f"{r['continuation']}")
            md_lines.append("")
    md_path = out_dir / "generation_samples.md"
    md_path.write_text("\n".join(md_lines) + "\n", encoding="utf-8")
    print(f"[gen] wrote {json_path}")
    print(f"[gen] wrote {md_path}")


if __name__ == "__main__":
    main()
