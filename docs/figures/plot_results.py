#!/usr/bin/env python3
"""Generate SVG bar charts for the technical report from on-disk metrics/bench JSON.

No matplotlib required — pure SVG. Run from repo root:

  python docs/figures/plot_results.py
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent


def _bar_svg(title, series, ylabel, out_name, width=720, height=420, y_fmt="{:.1f}"):
    """series: list of (label, value, color)."""
    margin_l, margin_r, margin_t, margin_b = 70, 24, 48, 90
    plot_w = width - margin_l - margin_r
    plot_h = height - margin_t - margin_b
    vmax = max(v for _, v, _ in series) * 1.15 or 1.0
    n = len(series)
    gap = 16
    bar_w = (plot_w - gap * (n + 1)) / max(n, 1)

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        f'<text x="{width/2}" y="28" text-anchor="middle" font-family="DejaVu Sans,Arial,sans-serif" '
        f'font-size="16" font-weight="600" fill="#111827">{title}</text>',
        # axes
        f'<line x1="{margin_l}" y1="{margin_t}" x2="{margin_l}" y2="{margin_t+plot_h}" '
        f'stroke="#374151" stroke-width="1.5"/>',
        f'<line x1="{margin_l}" y1="{margin_t+plot_h}" x2="{margin_l+plot_w}" y2="{margin_t+plot_h}" '
        f'stroke="#374151" stroke-width="1.5"/>',
        f'<text x="18" y="{margin_t+plot_h/2}" text-anchor="middle" transform='
        f'"rotate(-90 18 {margin_t+plot_h/2})" font-family="DejaVu Sans,Arial,sans-serif" '
        f'font-size="12" fill="#374151">{ylabel}</text>',
    ]

    # y grid
    for i in range(5):
        yv = vmax * i / 4
        y = margin_t + plot_h * (1 - i / 4)
        parts.append(
            f'<line x1="{margin_l}" y1="{y:.1f}" x2="{margin_l+plot_w}" y2="{y:.1f}" '
            f'stroke="#e5e7eb" stroke-width="1"/>'
        )
        parts.append(
            f'<text x="{margin_l-8}" y="{y+4:.1f}" text-anchor="end" '
            f'font-family="DejaVu Sans,Arial,sans-serif" font-size="11" fill="#6b7280">'
            f"{y_fmt.format(yv)}</text>"
        )

    for i, (label, val, color) in enumerate(series):
        x = margin_l + gap + i * (bar_w + gap)
        bh = plot_h * (val / vmax)
        y = margin_t + plot_h - bh
        parts.append(
            f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_w:.1f}" height="{bh:.1f}" '
            f'rx="4" fill="{color}"/>'
        )
        parts.append(
            f'<text x="{x+bar_w/2:.1f}" y="{y-6:.1f}" text-anchor="middle" '
            f'font-family="DejaVu Sans,Arial,sans-serif" font-size="11" fill="#111827">'
            f"{y_fmt.format(val)}</text>"
        )
        # multi-line labels
        for j, line in enumerate(label.split("\n")):
            parts.append(
                f'<text x="{x+bar_w/2:.1f}" y="{margin_t+plot_h+18+j*14:.1f}" text-anchor="middle" '
                f'font-family="DejaVu Sans,Arial,sans-serif" font-size="11" fill="#374151">{line}</text>'
            )

    parts.append("</svg>")
    path = OUT / out_name
    path.write_text("\n".join(parts) + "\n", encoding="utf-8")
    print(f"wrote {path}")
    return path


def load_ppl(seed):
    p = ROOT / "results" / "main_360m" / ("seed_%d" % seed) / "metrics.json"
    data = json.loads(p.read_text())
    return {m["route"]: float(m["ppl"]) for m in data}


def load_toks(path):
    if not path.exists():
        return None
    d = json.loads(path.read_text())
    return float(d.get("tokens_per_sec") or 0)


def main():
    colors = {"R1": "#2563eb", "R2": "#d97706", "R3": "#059669"}
    for seed in (42, 43):
        ppl = load_ppl(seed)
        _bar_svg(
            f"WikiText-2 PPL（seed={seed}，越低越好）",
            [
                (f"R1\nBF16", ppl["R1"], colors["R1"]),
                (f"R2\nBF16→NVFP4", ppl["R2"], colors["R2"]),
                (f"R3\nNVFP4", ppl["R3"], colors["R3"]),
            ],
            ylabel="PPL",
            out_name=f"fig06_ppl_seed{seed}.svg",
            y_fmt="{:.2f}",
        )

    # throughput: seed42 representative microbench
    perf = ROOT / "results" / "perf"
    pairs = [
        ("R1 train 1GPU", load_toks(perf / "bench_R1_train_n1_seed42_best.json") or load_toks(perf / "bench_R1_train_n1_best.json"), colors["R1"]),
        ("R1 infer 1GPU", load_toks(perf / "bench_R1_infer_n1_seed42_best.json") or 526460, colors["R1"]),
        ("R3 train 1GPU", load_toks(perf / "bench_R3_train_n1_seed42_best.json") or load_toks(perf / "bench_R3_train_n1_best.json"), colors["R3"]),
        ("R2 infer 1GPU", load_toks(perf / "bench_R2_infer_n1_seed42_best.json") or load_toks(perf / "bench_R2_infer_n1_best.json"), colors["R2"]),
        ("R3 infer 1GPU", load_toks(perf / "bench_R3_infer_n1_seed42_best.json") or load_toks(perf / "bench_R3_infer_n1_best.json"), colors["R3"]),
        ("R1 train 4GPU", load_toks(perf / "bench_R1_train_n4_seed42_best.json") or load_toks(perf / "bench_R1_train_n4_best.json"), colors["R1"]),
        ("R3 train 4GPU", load_toks(perf / "bench_R3_train_n4_seed42_best.json") or load_toks(perf / "bench_R3_train_n4_best.json"), colors["R3"]),
    ]
    series = [(lab.replace(" ", "\n"), v / 1000.0, c) for lab, v, c in pairs if v]
    _bar_svg(
        "吞吐 microbench（seed42 权重，k tokens/s，越高越好）",
        series,
        ylabel="k tokens/s",
        out_name="fig07_throughput.svg",
        width=900,
        height=440,
        y_fmt="{:.0f}",
    )


if __name__ == "__main__":
    main()
