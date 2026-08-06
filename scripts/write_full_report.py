#!/usr/bin/env python3
"""Aggregate PPL metrics + throughput benches into full_report.md.

Standalone of mxfp4_lib / TE so it can run on the login node (may be Python 3.6).
"""

from __future__ import print_function

import argparse
import json
from pathlib import Path


def _steady_toks(path):
    if not path.exists():
        return None
    vals = []
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if "tokens_per_s" in row:
            vals.append(float(row["tokens_per_s"]))
    if not vals:
        return None
    tail = vals[max(0, int(len(vals) * 0.8)) :]
    return sum(tail) / len(tail), len(tail)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/main_360m.yaml", help="unused path hint")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument(
        "--root",
        default=None,
        help="repo root (default: parent of scripts/)",
    )
    args = ap.parse_args()
    root = Path(args.root) if args.root else Path(__file__).resolve().parents[1]
    res_dir = root / "results" / "main_360m" / ("seed_%d" % args.seed)
    perf_dir = root / "results" / "perf"
    seed_tag = "seed%d" % args.seed

    lines = [
        "# Full R1–R5 report (seed=%d)" % args.seed,
        "",
        "Config: `%s`" % args.config,
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
                "| %s | %s | `%s` | **%.4f** |"
                % (
                    m.get("route"),
                    m.get("train_dtype"),
                    m.get("infer_dtype"),
                    float(m.get("ppl")),
                )
            )
    else:
        lines.append("_metrics.json not found_")

    lines += [
        "",
        "## Throughput (tokens/s)",
        "",
        "_Benches matching `*%s*` under `results/perf/`._" % seed_tag,
        "",
        "| File / route | Phase | nGPU | bs/gpu | tok/s | mem GB |",
        "|--------------|-------|-----:|-------:|------:|-------:|",
    ]
    if perf_dir.is_dir():
        paths = []
        seen = set()
        for p in sorted(perf_dir.glob("bench_R*.json")):
            if seed_tag not in p.name:
                continue
            if p.name in seen:
                continue
            seen.add(p.name)
            paths.append(p)
        if not paths:
            lines.append("_no perf results for %s_" % seed_tag)
        for p in paths:
            d = json.loads(p.read_text())
            lines.append(
                "| `%s` | %s | %s | %s | **%.1f** | %.1f |"
                % (
                    p.name,
                    d.get("phase"),
                    d.get("nproc"),
                    d.get("batch_size"),
                    float(d.get("tokens_per_sec") or 0),
                    float(d.get("peak_mem_gb") or 0),
                )
            )
    else:
        lines.append("_no perf results_")

    lines += ["", "## Train log steady tokens/s (from jsonl)", ""]
    any_jsonl = False
    for name in ("train_bf16.jsonl", "train_nvfp4.jsonl", "train_mxfp8.jsonl"):
        path = res_dir / name
        stats = _steady_toks(path)
        if stats is None:
            if path.exists():
                lines.append("- `%s`: no tokens_per_s fields" % name)
                any_jsonl = True
            continue
        avg, n = stats
        lines.append(
            "- `%s`: steady avg **%.1f** tok/s (n=%d log points)" % (name, avg, n)
        )
        any_jsonl = True
    if not any_jsonl:
        lines.append("_no train_*.jsonl under results_")

    out = res_dir / "full_report.md"
    try:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text("\n".join(lines) + "\n")
        print("[report] wrote %s" % out)
    except OSError as e:
        # NFS root_squash / ownership: write beside scripts as fallback is unhelpful;
        # print content path and error so the resume container can rewrite.
        print("[report] WARN could not write %s: %s" % (out, e))
        fallback = root / "logs" / ("full_report_seed%d.md" % args.seed)
        try:
            fallback.parent.mkdir(parents=True, exist_ok=True)
            fallback.write_text("\n".join(lines) + "\n")
            print("[report] wrote fallback %s" % fallback)
        except OSError as e2:
            print("[report] fallback failed: %s" % e2)
            raise e


if __name__ == "__main__":
    main()
