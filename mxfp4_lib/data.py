"""FineWeb-Edu packing + WikiText-2 helpers for train/eval."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from datasets import load_dataset, load_from_disk
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer


class PackedLMDataset(Dataset):
    def __init__(self, token_ids: torch.Tensor, seq_len: int):
        n = token_ids.numel() // seq_len
        usable = n * seq_len
        self.x = token_ids[:usable].view(n, seq_len)

    def __len__(self):
        return self.x.size(0)

    def __getitem__(self, idx):
        ids = self.x[idx].long()
        return {"input_ids": ids, "labels": ids}


def tokenize_wikitext(data_dir: Path, tok: AutoTokenizer, split: str = "train") -> torch.Tensor:
    ds = load_from_disk(str(Path(data_dir) / "wikitext2"))[split]
    texts = [t for t in ds["text"] if t and t.strip()]
    eos = tok.eos_token or ""
    big = eos.join(texts)
    enc = tok(big, return_tensors="pt", add_special_tokens=False)
    return enc["input_ids"].view(-1)


def _cache_path(data_dir: Path, name: str, target_tokens: int, seed: int) -> Path:
    # Prefer compact int32 memmap (.npy); keep .pt as legacy fallback
    # Optional node-local override (cold NFS nodes): LOCAL_TRAIN_NPY / LOCAL_VAL_NPY
    import os

    if name == "train":
        local = os.environ.get("LOCAL_TRAIN_NPY", "").strip()
        if local and Path(local).exists():
            return Path(local)
    if name == "val":
        local = os.environ.get("LOCAL_VAL_NPY", "").strip()
        if local and Path(local).exists():
            return Path(local)
    return Path(data_dir) / "fineweb_edu" / f"{name}_tok{target_tokens}_seed{seed}.npy"


def _legacy_pt_path(data_dir: Path, name: str, target_tokens: int, seed: int) -> Path:
    return Path(data_dir) / "fineweb_edu" / f"{name}_tok{target_tokens}_seed{seed}.pt"


def build_fineweb_token_buffer(
    cfg: dict,
    tok: AutoTokenizer,
    *,
    target_tokens: int,
    name: str,
    seed: int,
) -> torch.Tensor:
    """
    Stream FineWeb-Edu, tokenize, write int32 tokens to a memmap until target_tokens.
    Low-RAM: does not keep all chunks in a Python list.
    """
    data_dir = Path(cfg["paths"]["data_dir"])
    cache = _cache_path(data_dir, name, target_tokens, seed)
    legacy = _legacy_pt_path(data_dir, name, target_tokens, seed)
    if cache.exists():
        print(f"[data] Loading memmap tokens from {cache}")
        arr = np.load(cache, mmap_mode="r")
        # Keep int32 memmap; Dataset casts to long per batch
        return torch.from_numpy(arr)
    if legacy.exists():
        print(f"[data] Loading cached tokens from {legacy}")
        return torch.load(legacy, map_location="cpu", weights_only=True)

    cache.parent.mkdir(parents=True, exist_ok=True)
    ds_name = cfg["data"].get("train_dataset", "HuggingFaceFW/fineweb-edu")
    ds_config = cfg["data"].get("train_dataset_config", None)
    text_col = cfg["data"].get("text_column", "text")
    print(f"[data] Streaming {ds_name} config={ds_config} until {target_tokens} tokens ({name})")

    kwargs = {"path": ds_name, "split": "train", "streaming": True}
    if ds_config:
        kwargs["name"] = ds_config
    ds = load_dataset(**kwargs)
    ds = ds.shuffle(seed=seed, buffer_size=int(cfg["data"].get("stream_shuffle_buffer", 10_000)))

    tmp = cache.with_suffix(".npy.tmp")
    mm = np.lib.format.open_memmap(tmp, mode="w+", dtype=np.int32, shape=(target_tokens,))
    n_tok = 0
    last_report = 0
    eos = tok.eos_token or ""
    try:
        tok.model_max_length = 10**9
    except Exception:
        pass
    batch_texts: list[str] = []
    flush_every = int(cfg["data"].get("tokenize_flush_docs", 256))
    report_every = max(1_000_000, target_tokens // 20)

    def _flush_ids(id_lists: list[list[int]]) -> None:
        nonlocal n_tok, last_report
        for ids in id_lists:
            if not ids:
                continue
            take = min(len(ids), target_tokens - n_tok)
            if take <= 0:
                return
            mm[n_tok : n_tok + take] = np.asarray(ids[:take], dtype=np.int32)
            n_tok += take
            if n_tok - last_report >= report_every:
                print(f"[data] {name}: {n_tok}/{target_tokens} tokens...", flush=True)
                last_report = n_tok
                mm.flush()
            if n_tok >= target_tokens:
                return

    for ex in ds:
        t = ex.get(text_col) or ""
        if not str(t).strip():
            continue
        batch_texts.append(str(t).strip() + eos)
        if len(batch_texts) < flush_every:
            continue
        enc = tok(batch_texts, add_special_tokens=False, return_attention_mask=False)
        batch_texts = []
        _flush_ids(enc["input_ids"])
        if n_tok >= target_tokens:
            break

    if batch_texts and n_tok < target_tokens:
        enc = tok(batch_texts, add_special_tokens=False, return_attention_mask=False)
        _flush_ids(enc["input_ids"])

    if n_tok <= 0:
        raise RuntimeError("FineWeb-Edu stream produced zero tokens")

    mm.flush()
    del mm
    # Truncate file to actual length if short (rare)
    arr = np.load(tmp, mmap_mode="r")
    actual = int(min(n_tok, target_tokens))
    if actual < target_tokens:
        final = np.lib.format.open_memmap(cache, mode="w+", dtype=np.int32, shape=(actual,))
        final[:] = arr[:actual]
        final.flush()
        del final
        tmp.unlink(missing_ok=True)
    else:
        tmp.replace(cache)

    meta = {"name": name, "target_tokens": target_tokens, "actual": actual, "seed": seed, "dtype": "int32"}
    with open(cache.with_suffix(".json"), "w") as f:
        json.dump(meta, f, indent=2)
    print(f"[data] Cached {actual} tokens → {cache}")
    arr = np.load(cache, mmap_mode="r")
    return torch.from_numpy(arr)


def build_train_loader(
    cfg: dict,
    tok: AutoTokenizer,
    *,
    rank: int = 0,
    world_size: int = 1,
) -> DataLoader:
    tcfg = cfg["train"]
    target = int(tcfg.get("target_tokens") or 0)
    seq_len = int(cfg["data"]["seq_len"])
    if target > 0:
        ids = build_fineweb_token_buffer(
            cfg, tok, target_tokens=target, name="train", seed=cfg["seed"]
        )
    else:
        ids = tokenize_wikitext(cfg["paths"]["data_dir"], tok, "train")

    ds = PackedLMDataset(ids, seq_len)
    sampler = None
    shuffle = True
    if world_size > 1:
        from torch.utils.data.distributed import DistributedSampler

        sampler = DistributedSampler(ds, num_replicas=world_size, rank=rank, shuffle=True, seed=cfg["seed"])
        shuffle = False

    # Allow env override for cold-node resumes (FP4_NUM_WORKERS=0 avoids NFS storms)
    import os

    nw = int(os.environ.get("FP4_NUM_WORKERS", tcfg.get("num_workers", 2)))
    kwargs = dict(
        batch_size=int(tcfg["batch_size"]),
        shuffle=shuffle,
        sampler=sampler,
        drop_last=True,
        num_workers=nw,
        pin_memory=bool(tcfg.get("pin_memory", True)),
        generator=torch.Generator().manual_seed(cfg["seed"] + rank),
    )
    if nw > 0:
        kwargs["persistent_workers"] = bool(tcfg.get("persistent_workers", True))
        kwargs["prefetch_factor"] = int(tcfg.get("prefetch_factor", 4))
    return DataLoader(ds, **kwargs)


def build_val_loader(cfg: dict, tok: AutoTokenizer) -> DataLoader | None:
    vtok = int(cfg.get("eval", {}).get("val_tokens") or 0)
    if vtok <= 0:
        return None
    ids = build_fineweb_token_buffer(
        cfg,
        tok,
        target_tokens=vtok,
        name="val",
        seed=cfg["seed"] + 10_000,
    )
    ds = PackedLMDataset(ids, cfg["data"]["seq_len"])
    nw = min(2, int(cfg["train"].get("num_workers", 2)))
    kwargs = dict(
        batch_size=int(cfg["train"]["batch_size"]),
        shuffle=False,
        drop_last=False,
        num_workers=nw,
        pin_memory=bool(cfg["train"].get("pin_memory", True)),
    )
    if nw > 0:
        kwargs["persistent_workers"] = True
        kwargs["prefetch_factor"] = int(cfg["train"].get("prefetch_factor", 2))
    return DataLoader(ds, **kwargs)


def steps_for_target_tokens(cfg: dict, world_size: int = 1) -> int:
    tcfg = cfg["train"]
    target = int(tcfg.get("target_tokens") or 0)
    if target <= 0:
        return int(tcfg["max_steps"])
    per_step = (
        int(tcfg["batch_size"])
        * int(tcfg["grad_accum"])
        * int(cfg["data"]["seq_len"])
        * int(world_size)
    )
    return max(1, (target + per_step - 1) // per_step)


__all__ = [
    "PackedLMDataset",
    "tokenize_wikitext",
    "build_fineweb_token_buffer",
    "build_train_loader",
    "build_val_loader",
    "steps_for_target_tokens",
]
