#!/usr/bin/env python3
"""Throughput microbench: bf16 / TE NVFP4.

Examples:
  python scripts/12_bench_throughput.py --backend bf16 --phase train --batch-size 64
  python scripts/12_bench_throughput.py --backend te_fp4 --phase train --sweep
  torchrun --standalone --nproc_per_node=4 scripts/12_bench_throughput.py \\
      --backend te_fp4 --phase train --sweep --ddp
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from transformers import AutoModelForCausalLM

from mxfp4_lib.bench import (
    BenchResult,
    make_infer_step,
    make_lm_batch,
    make_train_step,
    run_bench,
    write_result,
)
from mxfp4_lib.model_config import build_model_from_arch
from mxfp4_lib.util import ensure_dir, hf_env, load_cfg, set_seed


def _has_weights(path: Path) -> bool:
    if not path.is_dir() or not (path / "config.json").exists():
        return False
    if (path / "model.safetensors").exists():
        return True
    if (path / "model.safetensors.index.json").exists():
        return True
    if list(path.glob("pytorch_model*.bin")):
        return True
    return False


def _dist_info() -> Tuple[bool, int, int, int]:
    """Return (enabled, rank, world_size, local_rank)."""
    if not dist.is_available():
        return False, 0, 1, 0
    # torchrun sets these even before init
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world = int(os.environ["WORLD_SIZE"])
        local = int(os.environ.get("LOCAL_RANK", rank))
        if not dist.is_initialized():
            dist.init_process_group(backend="nccl")
        torch.cuda.set_device(local)
        return True, rank, world, local
    return False, 0, 1, 0


def _is_main(rank: int) -> bool:
    return rank == 0


def _resolve_route(route: str | None, backend: str, weights: str | None, phase: str):
    """Map R1/R2/R3 to backend + weight source."""
    if not route:
        return backend, (weights or "auto"), None
    r = route.upper()
    if r == "R1":
        # train + infer both use bf16 compute
        return "bf16", "bf16", "R1"
    if r == "R2":
        return "te_fp4", "bf16", "R2"
    if r == "R3":
        return "te_fp4", "nvfp4", "R3"
    raise SystemExit(f"unknown route {route}")


def _build_model(
    cfg,
    backend: str,
    device: torch.device,
    *,
    weights: str = "auto",
    route_label: str | None = None,
):
    """weights: auto|bf16|nvfp4|random — checkpoint source; backend: compute path."""
    init = Path(cfg["paths"].get("init_model", "checkpoints/init_model"))
    ckpt_bf16 = Path(cfg["paths"].get("ckpt_bf16", ""))
    ckpt_nvfp4 = Path(cfg["paths"].get("ckpt_nvfp4", ""))
    notes = []
    rank = int(os.environ.get("RANK", "0"))
    wsrc = (weights or "auto").lower()

    load_path = None
    if wsrc == "bf16" and _has_weights(ckpt_bf16):
        load_path = ckpt_bf16
    elif wsrc == "nvfp4" and _has_weights(ckpt_nvfp4):
        load_path = ckpt_nvfp4
    elif wsrc == "auto":
        if _has_weights(ckpt_bf16):
            load_path = ckpt_bf16
        elif _has_weights(ckpt_nvfp4):
            load_path = ckpt_nvfp4
        elif _has_weights(init):
            load_path = init

    if load_path is not None and wsrc != "random":
        if _is_main(rank):
            print(f"[bench] load weights from {load_path}")
        model = AutoModelForCausalLM.from_pretrained(load_path)
        notes.append(f"weights={load_path}")
    else:
        if _is_main(rank):
            print("[bench] from_config random init")
        model, _ = build_model_from_arch(cfg)
        notes.append("weights=from_config_random")
    if route_label:
        notes.append(f"route={route_label}")

    te_ctx = None
    recipe_name = None

    if backend == "bf16":
        model = model.to(device=device, dtype=torch.bfloat16)
    elif backend == "fp16":
        model = model.to(device=device, dtype=torch.float16)
    elif backend == "te_fp4":
        from mxfp4_lib.te_linear import (
            get_preferred_recipe,
            make_te_autocast_ctx,
            replace_linears_with_te,
        )

        model = model.to(device=device, dtype=torch.bfloat16)
        recipe, recipe_name = get_preferred_recipe()
        n = replace_linears_with_te(model, include_lm_head=False)
        notes.append(f"te_linears={n},recipe={recipe_name}")
        te_ctx = make_te_autocast_ctx(recipe)
    else:
        raise SystemExit(f"Unknown backend: {backend}")

    return model, te_ctx, ";".join(notes), recipe_name


def _candidate_batches(start: int, max_bs: int) -> List[int]:
    """Geometric-ish batch sizes from start up to max_bs."""
    out = []
    b = max(1, start)
    while b <= max_bs:
        out.append(b)
        if b < 32:
            b *= 2
        elif b < 128:
            b += 16
        elif b < 256:
            b += 32
        else:
            b += 64
    if max_bs not in out:
        out.append(max_bs)
    return sorted(set(out))


def _try_bench_one(
    *,
    cfg,
    backend: str,
    phase: str,
    batch: int,
    seq: int,
    warmup: int,
    measure: int,
    device: torch.device,
    world_size: int,
    use_ddp: bool,
    local_rank: int,
    weights: str = "auto",
    route_label: str | None = None,
) -> Optional[BenchResult]:
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    try:
        model, te_ctx, notes, recipe_name = _build_model(
            cfg, backend, device, weights=weights, route_label=route_label
        )
        model = model.to(device)
        if use_ddp:
            model = DDP(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=False)
        raw = model.module if isinstance(model, DDP) else model
        vocab = int(getattr(raw.config, "vocab_size", 49152))
        batch_t = make_lm_batch(batch, seq, vocab, device)

        if phase == "train":
            model.train()
            params = model.parameters()
            opt = torch.optim.AdamW(params, lr=float((cfg.get("train") or {}).get("lr", 3e-4)), weight_decay=0.0)
            amp = torch.bfloat16 if backend == "bf16" else None
            if backend in ("te_fp4", "fp16"):
                amp = None
            # DDP wraps model: forward via model() still works
            step_fn = make_train_step(model, opt, batch_t, amp_dtype=amp, te_ctx=te_ctx)
        else:
            model.eval()
            step_fn = make_infer_step(
                model,
                batch_t,
                use_fp16=(backend == "fp16"),
                te_ctx=te_ctx,
            )

        backend_label = backend
        if backend == "te_fp4" and recipe_name:
            backend_label = f"te_{recipe_name.replace('BlockScaling', '').lower()}"
        if route_label:
            backend_label = f"{route_label}_{backend_label}"

        # Synchronize ranks before timing
        if use_ddp and dist.is_initialized():
            dist.barrier()

        result = run_bench(
            backend=backend_label,
            phase=phase,
            step_fn=step_fn,
            batch_size=batch,
            seq_len=seq,
            nproc=world_size,
            warmup=warmup,
            measure=measure,
            notes=notes + (f";ddp={world_size}" if use_ddp else ""),
            extra={
                "recipe": recipe_name,
                "route": route_label,
                "weights": weights,
                "config": "bench",
                "per_gpu_batch": batch,
                "global_batch": batch * world_size,
                "ddp": use_ddp,
            },
        )
        del model, batch_t
        if phase == "train":
            del opt
        torch.cuda.empty_cache()
        return result
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        return None
    except RuntimeError as e:
        msg = str(e).lower()
        # OOM or NVML/driver glitches at near-full memory — treat as soft fail
        soft = (
            "out of memory" in msg
            or "nvml" in msg
            or "cuda error" in msg
            or "devicesunavailable" in msg
            or "device-side assert" in msg
        )
        torch.cuda.empty_cache()
        if soft:
            print(f"[bench] soft-fail at bs={batch}: {type(e).__name__}: {e}", flush=True)
            return None
        raise


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/bench_360m.yaml")
    ap.add_argument("--backend", default=None, choices=["bf16", "fp16", "te_fp4"])
    ap.add_argument("--phase", required=True, choices=["train", "infer"])
    ap.add_argument("--route", default=None, choices=["R1", "R2", "R3", "r1", "r2", "r3"])
    ap.add_argument(
        "--weights",
        default=None,
        choices=["auto", "bf16", "nvfp4", "random"],
        help="Checkpoint source (default from --route)",
    )
    ap.add_argument("--batch-size", type=int, default=None)
    ap.add_argument("--seq-len", type=int, default=None)
    ap.add_argument("--warmup", type=int, default=None)
    ap.add_argument("--measure", type=int, default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--sweep", action="store_true", help="Sweep batch sizes until OOM; keep best tok/s")
    ap.add_argument("--max-batch", type=int, default=512, help="Upper bound for --sweep")
    ap.add_argument("--start-batch", type=int, default=None, help="First batch in sweep")
    ap.add_argument("--ddp", action="store_true", help="Expect torchrun; use DDP (nproc from env)")
    args = ap.parse_args()

    if not args.route and not args.backend:
        raise SystemExit("Need --backend and/or --route")

    cfg = load_cfg(args.config)
    hf_env(cfg)
    set_seed(cfg.get("seed", 42))
    if not torch.cuda.is_available():
        raise SystemExit("CUDA required for throughput bench")

    backend = args.backend or "bf16"
    weights = args.weights or "auto"
    route_label = None
    if args.route:
        backend, weights, route_label = _resolve_route(args.route, backend, args.weights, args.phase)

    use_ddp, rank, world, local_rank = _dist_info()
    if args.ddp and not use_ddp:
        raise SystemExit("--ddp requires torchrun (RANK/WORLD_SIZE env)")
    if not args.ddp:
        use_ddp, rank, world, local_rank = False, 0, 1, 0

    device = torch.device(f"cuda:{local_rank}")
    tcfg = cfg.get("train") or {}
    bcfg = cfg.get("bench") or {}
    if tcfg.get("allow_tf32", True):
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    if tcfg.get("cudnn_benchmark", True):
        torch.backends.cudnn.benchmark = True

    seq = int(args.seq_len or cfg.get("data", {}).get("seq_len", 512))
    warmup = int(args.warmup if args.warmup is not None else bcfg.get("warmup", 5))
    measure = int(args.measure if args.measure is not None else bcfg.get("measure", 20))

    if args.sweep:
        start = int(args.start_batch or 32)
        batches = _candidate_batches(start, int(args.max_batch))
    else:
        batches = [int(args.batch_size or tcfg.get("batch_size", 64))]

    best: Optional[BenchResult] = None
    all_ok: List[dict] = []

    for batch in batches:
        if _is_main(rank):
            print(
                f"[bench] try route={route_label or '-'} backend={backend} "
                f"weights={weights} phase={args.phase} per_gpu_bs={batch} world={world}"
            )
        res = _try_bench_one(
            cfg=cfg,
            backend=backend,
            phase=args.phase,
            batch=batch,
            seq=seq,
            warmup=warmup,
            measure=measure,
            device=device,
            world_size=world,
            use_ddp=use_ddp,
            local_rank=local_rank,
            weights=weights,
            route_label=route_label,
        )
        # If any rank OOMs, all should treat as fail — OOM is local
        oom_flag = 0 if res is not None else 1
        if use_ddp and dist.is_initialized():
            t = torch.tensor([oom_flag], device=device)
            dist.all_reduce(t, op=dist.ReduceOp.MAX)
            oom_flag = int(t.item())
        if oom_flag:
            if _is_main(rank):
                print(f"[bench] OOM at per_gpu_bs={batch}; stop sweep")
            break
        assert res is not None
        all_ok.append(res.to_dict())
        if best is None or res.tokens_per_sec > best.tokens_per_sec:
            best = res
        if _is_main(rank):
            print(
                f"[bench] OK bs={batch} global_bs={batch * world} "
                f"{res.tokens_per_sec:.1f} tok/s peak={res.peak_mem_gb:.1f}GB"
            )

    if best is None:
        if use_ddp and dist.is_initialized():
            dist.destroy_process_group()
        raise SystemExit(f"No successful batch for {route_label or backend}/{args.phase}")

    if _is_main(rank):
        out_dir = ensure_dir(Path(cfg["paths"]["results"]))
        tag = f"n{world}"
        out_path = (
            Path(args.out)
            if args.out
            else out_dir
            / f"bench_{best.backend}_{args.phase}_{tag}_bs{best.batch_size}.json"
        )
        best.extra["sweep"] = all_ok if args.sweep else None
        best.extra["max_mem_batch"] = all_ok[-1]["batch_size"] if all_ok else best.batch_size
        best.extra["best_tok_s_batch"] = best.batch_size
        write_result(out_path, best)
        # also write full sweep summary
        if args.sweep:
            summary_path = out_dir / f"sweep_{best.backend}_{args.phase}_{tag}.json"
            with open(summary_path, "w") as f:
                json.dump(
                    {
                        "backend": best.backend,
                        "phase": args.phase,
                        "world_size": world,
                        "seq_len": seq,
                        "points": all_ok,
                        "best_tokens_per_sec": best.tokens_per_sec,
                        "best_batch": best.batch_size,
                        "max_batch_no_oom": all_ok[-1]["batch_size"],
                    },
                    f,
                    indent=2,
                )
            print(f"[bench] sweep summary {summary_path}")

    if use_ddp and dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
