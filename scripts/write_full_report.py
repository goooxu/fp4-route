#!/usr/bin/env python3
"""Aggregate PPL metrics + throughput benches into full_report.md."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from mxfp4_lib.util import load_cfg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/main_360m.yaml")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    cfg = load_cfg(args.config, seed=args.seed)
    root = Path(cfg["_root"])
    res_dir = Path(cfg["paths"]["results"])
    perf_dir = root / "results" / "perf"

    lines = [
        f"# Full R1/R2/R3 report (seed={args.seed})",
        "",
        f"Config: `{args.config}`",
        "",
        "## Quality (WikiText-2 PPL)",
        "",
    ]
    metrics_path = res_dir / "metrics.json"
    if metrics_path.exists():
        metrics = json.loads(metrics_path.read_text())
        lines += [
            "| Route | Train | Infer | PPL |",
            "|-------|-------|-------|-----|",
        ]
        for m in metrics:
            lines.append(
                f"| {m.get('route')} | {m.get('train_dtype')} | `{m.get('infer_dtype')}` | **{m.get('ppl'):.4f}** |"
            )
    else:
        lines.append("_metrics.json not found_")

    lines += ["", "## Throughput (tokens/s)", "", "| File / route | Phase | nGPU | bs/gpu | tok/s | mem GB |", "|--------------|-------|-----:|-------:|------:|-------:|"]
    if perf_dir.is_dir():
        for p in sorted(perf_dir.glob("bench_R*.json")):
            d = json.loads(p.read_text())
            lines.append(
                f"| `{p.name}` | {d.get('phase')} | {d.get('nproc')} | {d.get('batch_size')} | "
                f"**{d.get('tokens_per_sec', 0):.1f}** | {d.get('peak_mem_gb', 0):.1f} |"
            )
    else:
        lines.append("_no perf results_")

    # Steady train throughput from jsonl if present
    lines += ["", "## Train log steady tokens/s (from jsonl)", ""]
    for name in ("train_bf16.jsonl", "train_nvfp4.jsonl"):
        path = res_dir / name
        if not path.exists():
            # also under seed results sometimes only metrics
            continue
        vals = []
        for line in path.read_text().splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            if "tokens_per_s" in row:
                vals.append(float(row["tokens_per_s"]))
        if vals:
            # last 20%
            tail = vals[max(0, int(len(vals) * 0.8)) :]
            avg = sum(tail) / len(tail)
            lines.append(f"- `{name}`: steady avg **{avg:.1f}** tok/s (n={len(tail)} log points)")
        else:
            lines.append(f"- `{name}`: no tokens_per_s fields")

    # Also look under results for train logs (train_loop writes next to results often)
    for log_name, label in (
        (res_dir / "train_bf16.jsonl", "R1 BF16"),
        (res_dir / "train_nvfp4.jsonl", "R3 NVFP4"),
    ):
        pass

    out = res_dir / "full_report.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n")
    print(f"[report] wrote {out}")


if __name__ == "__main__":
    main()
