"""Throughput benchmarking helpers (train / infer tokens/s)."""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import torch


@dataclass
class BenchResult:
    backend: str
    phase: str  # train | infer
    tokens_per_sec: float
    ms_per_step: float
    batch_size: int
    seq_len: int
    nproc: int
    warmup: int
    measure: int
    peak_mem_gb: float
    device_name: str
    notes: str = ""
    extra: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def cuda_sync() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def peak_mem_gb() -> float:
    if not torch.cuda.is_available():
        return 0.0
    return float(torch.cuda.max_memory_allocated()) / (1024**3)


def reset_peak_mem() -> None:
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()


def timed_loop(
    step_fn: Callable[[], None],
    *,
    warmup: int,
    measure: int,
) -> float:
    """Return average wall seconds per measured step (CUDA-synced)."""
    for _ in range(max(0, warmup)):
        step_fn()
    cuda_sync()
    t0 = time.perf_counter()
    for _ in range(measure):
        step_fn()
    cuda_sync()
    t1 = time.perf_counter()
    return (t1 - t0) / max(1, measure)


def make_train_step(
    model: torch.nn.Module,
    opt: torch.optim.Optimizer,
    batch: Dict[str, torch.Tensor],
    *,
    amp_dtype: Optional[torch.dtype] = None,
    te_ctx: Optional[Callable] = None,
) -> Callable[[], None]:
    """One optimizer step: forward + loss + backward + step."""

    def step() -> None:
        opt.zero_grad(set_to_none=True)
        ids = batch["input_ids"]
        labels = batch["labels"]

        def _fwd():
            if amp_dtype is not None and te_ctx is None:
                with torch.autocast(device_type="cuda", dtype=amp_dtype):
                    out = model(input_ids=ids, labels=labels)
                    return out.loss
            out = model(input_ids=ids, labels=labels)
            return out.loss

        if te_ctx is not None:
            with te_ctx():
                loss = _fwd()
            # TE: backward outside autocast
            loss.backward()
        else:
            loss = _fwd()
            loss.backward()
        opt.step()

    return step


@torch.no_grad()
def make_infer_step(
    model: torch.nn.Module,
    batch: Dict[str, torch.Tensor],
    *,
    use_fp16: bool = False,
    te_ctx: Optional[Callable] = None,
) -> Callable[[], None]:
    def step() -> None:
        ids = batch["input_ids"]

        def _fwd():
            if use_fp16 and te_ctx is None:
                with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=False):
                    # true fp16 weights path: model already cast
                    return model(input_ids=ids)
            return model(input_ids=ids)

        if te_ctx is not None:
            with te_ctx():
                _fwd()
        else:
            _fwd()

    return step


def tokens_per_step(batch_size: int, seq_len: int, nproc: int = 1) -> int:
    return int(batch_size * seq_len * nproc)


def run_bench(
    *,
    backend: str,
    phase: str,
    step_fn: Callable[[], None],
    batch_size: int,
    seq_len: int,
    nproc: int = 1,
    warmup: int = 10,
    measure: int = 50,
    notes: str = "",
    extra: Optional[Dict[str, Any]] = None,
) -> BenchResult:
    reset_peak_mem()
    # one dry step to allocate
    step_fn()
    cuda_sync()
    reset_peak_mem()
    sec = timed_loop(step_fn, warmup=warmup, measure=measure)
    tps = tokens_per_step(batch_size, seq_len, nproc) / max(sec, 1e-9)
    dev = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
    return BenchResult(
        backend=backend,
        phase=phase,
        tokens_per_sec=float(tps),
        ms_per_step=float(sec * 1000.0),
        batch_size=batch_size,
        seq_len=seq_len,
        nproc=nproc,
        warmup=warmup,
        measure=measure,
        peak_mem_gb=peak_mem_gb(),
        device_name=dev,
        notes=notes,
        extra=extra or {},
    )


def write_result(path: Path, result: BenchResult) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(result.to_dict(), f, indent=2)
    print(f"[bench] wrote {path}")
    print(
        f"[bench] {result.backend}/{result.phase}: "
        f"{result.tokens_per_sec:.1f} tok/s  {result.ms_per_step:.2f} ms/step  "
        f"batch={result.batch_size} seq={result.seq_len} peak_mem={result.peak_mem_gb:.2f} GB"
    )


def make_lm_batch(
    batch_size: int,
    seq_len: int,
    vocab_size: int,
    device: torch.device,
) -> Dict[str, torch.Tensor]:
    ids = torch.randint(0, vocab_size, (batch_size, seq_len), device=device, dtype=torch.long)
    labels = ids.clone()
    return {"input_ids": ids, "labels": labels}


__all__ = [
    "BenchResult",
    "cuda_sync",
    "make_infer_step",
    "make_lm_batch",
    "make_train_step",
    "run_bench",
    "tokens_per_step",
    "write_result",
]
