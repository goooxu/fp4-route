"""Shared helpers for experiment scripts."""

from __future__ import annotations

import json
import os
import random
from pathlib import Path

import numpy as np
import torch
import yaml


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def load_cfg(path: str | Path | None = None, seed: int | None = None) -> dict:
    root = project_root()
    cfg_path = Path(path) if path else root / "configs" / "train.yaml"
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)
    if seed is not None:
        cfg["seed"] = int(seed)
    # Resolve relative paths against project root unless absolute
    cfg["_root"] = root
    cfg["_config_path"] = str(cfg_path.resolve())
    paths = cfg.setdefault("paths", {})
    for k, v in list(paths.items()):
        p = Path(v)
        if not p.is_absolute():
            paths[k] = str((root / p).resolve())
    # Per-seed checkpoint isolation when seed overridden via multi-run
    seed_tag = cfg.get("paths_seed_subdir")
    if seed_tag or cfg.get("isolate_seed_paths"):
        s = cfg["seed"]
        for key in (
            "init_model",
            "ckpt_bf16",
            "ckpt_nvfp4",
            "ckpt_mxfp8",
            "results",
        ):
            if key in paths:
                base = Path(paths[key])
                # results/seed_42, checkpoints/seed_42/...
                if key == "results":
                    paths[key] = str((base / f"seed_{s}").resolve())
                else:
                    # checkpoints/ckpt_bf16 -> checkpoints/seed_42/ckpt_bf16
                    paths[key] = str((base.parent / f"seed_{s}" / base.name).resolve())
    return cfg


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def save_jsonl(path: str | Path, rows: list[dict]) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


def append_jsonl(path: str | Path, row: dict) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    with open(path, "a") as f:
        f.write(json.dumps(row) + "\n")


def hf_env(cfg: dict) -> None:
    cache = cfg["paths"]["hf_cache"]
    ensure_dir(cache)
    os.environ.setdefault("HF_HOME", cache)
    os.environ.setdefault("HF_DATASETS_CACHE", str(Path(cache) / "datasets"))
    os.environ.setdefault("TRANSFORMERS_CACHE", str(Path(cache) / "transformers"))
    # Never pull model weights accidentally via default hub behavior in scripts
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")


def cosine_lr(step: int, warmup: int, max_steps: int, lr: float, min_lr: float) -> float:
    if step < warmup:
        return lr * float(step + 1) / float(max(1, warmup))
    progress = float(step - warmup) / float(max(1, max_steps - warmup))
    progress = min(1.0, max(0.0, progress))
    import math

    return min_lr + 0.5 * (lr - min_lr) * (1.0 + math.cos(math.pi * progress))
