"""Shared causal LM training loop over packed WikiText tokens."""

from __future__ import annotations

import json
import time
from pathlib import Path

import torch
from datasets import load_from_disk
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from mxfp4_lib.util import append_jsonl, cosine_lr, ensure_dir


class PackedLMDataset(Dataset):
    def __init__(self, token_ids: torch.Tensor, seq_len: int):
        # HF CausalLM shifts labels internally — feed identical input_ids/labels.
        n = token_ids.numel() // seq_len
        usable = n * seq_len
        self.x = token_ids[:usable].view(n, seq_len)

    def __len__(self):
        return self.x.size(0)

    def __getitem__(self, idx):
        ids = self.x[idx]
        return {"input_ids": ids, "labels": ids}


def tokenize_wikitext(data_dir: Path, tok: AutoTokenizer, split: str = "train") -> torch.Tensor:
    ds = load_from_disk(str(Path(data_dir) / "wikitext2"))[split]
    texts = [t for t in ds["text"] if t and t.strip()]
    eos = tok.eos_token or ""
    big = eos.join(texts)
    enc = tok(big, return_tensors="pt", add_special_tokens=False)
    return enc["input_ids"].view(-1)


def build_train_loader(cfg: dict, tok: AutoTokenizer) -> DataLoader:
    ids = tokenize_wikitext(cfg["paths"]["data_dir"], tok, "train")
    ds = PackedLMDataset(ids, cfg["data"]["seq_len"])
    return DataLoader(
        ds,
        batch_size=cfg["train"]["batch_size"],
        shuffle=True,
        drop_last=True,
        num_workers=2,
        pin_memory=True,
        generator=torch.Generator().manual_seed(cfg["seed"]),
    )


def train_loop(
    model: AutoModelForCausalLM,
    cfg: dict,
    out_dir: str,
    log_name: str,
    max_steps: int | None = None,
    pre_save=None,
) -> dict:
    """Train for `max_steps` optimizer steps (each = grad_accum micro-batches)."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.train()

    tok = AutoTokenizer.from_pretrained(cfg["paths"]["init_model"])
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    loader = build_train_loader(cfg, tok)

    tcfg = cfg["train"]
    max_steps = max_steps if max_steps is not None else tcfg["max_steps"]
    accum = int(tcfg["grad_accum"])
    opt = torch.optim.AdamW(
        model.parameters(),
        lr=tcfg["lr"],
        weight_decay=tcfg["weight_decay"],
        betas=(0.9, 0.95),
    )

    log_path = Path(cfg["paths"]["results"]) / log_name
    if log_path.exists():
        log_path.unlink()

    use_bf16 = bool(tcfg.get("bf16", True)) and device.type == "cuda"
    step = 0
    micro_loss_sum = 0.0
    log_loss_sum = 0.0
    log_count = 0
    t0 = time.time()
    data_iter = iter(loader)
    pbar = tqdm(total=max_steps, desc=f"train->{Path(out_dir).name}")
    opt.zero_grad(set_to_none=True)
    last_logged_loss = None

    while step < max_steps:
        lr = cosine_lr(step, tcfg["warmup_steps"], max_steps, tcfg["lr"], tcfg["min_lr"])
        for g in opt.param_groups:
            g["lr"] = lr

        micro_losses = []
        for _ in range(accum):
            try:
                batch = next(data_iter)
            except StopIteration:
                data_iter = iter(loader)
                batch = next(data_iter)

            input_ids = batch["input_ids"].to(device, non_blocking=True)
            labels = batch["labels"].to(device, non_blocking=True)
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=use_bf16):
                out = model(input_ids=input_ids, labels=labels)
                loss = out.loss / accum
            loss.backward()
            micro_losses.append(float(out.loss.detach().item()))

        torch.nn.utils.clip_grad_norm_(model.parameters(), tcfg["max_grad_norm"])
        opt.step()
        opt.zero_grad(set_to_none=True)

        step += 1
        step_loss = sum(micro_losses) / len(micro_losses)
        log_loss_sum += step_loss
        log_count += 1
        pbar.update(1)

        if step % tcfg["log_every"] == 0:
            avg = log_loss_sum / max(1, log_count)
            last_logged_loss = avg
            log_loss_sum = 0.0
            log_count = 0
            row = {"step": step, "loss": avg, "lr": lr, "seconds": time.time() - t0}
            append_jsonl(log_path, row)
            pbar.set_postfix(loss=f"{avg:.4f}", lr=f"{lr:.2e}")

    pbar.close()

    outp = ensure_dir(out_dir)
    if pre_save is not None:
        pre_save(model)
    model.cpu()
    model.save_pretrained(outp, safe_serialization=True)
    tok.save_pretrained(outp)

    meta = {
        "max_steps": max_steps,
        "final_loss_proxy": last_logged_loss,
        "seconds": time.time() - t0,
        "device": str(device),
    }
    with open(Path(outp) / "train_meta.json", "w") as f:
        json.dump(meta, f, indent=2)
    print(f"[train] Saved to {outp} meta={meta}")
    return meta
