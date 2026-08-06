#!/usr/bin/env python3
"""Generate SVG bar charts for the technical report from on-disk metrics/bench JSON.

No matplotlib required — pure SVG. Run from repo root:

  python docs/figures/plot_results.py

Compatible with Python 3.6+ (login node may be 3.6; GPU image is newer).
"""
from __future__ import print_function

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent

COLORS = {
    "R1": "#2563eb",
    "R2": "#d97706",
    "R3": "#059669",
    "R4": "#7c3aed",
    "R5": "#db2777",
}


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
        '<svg xmlns="http://www.w3.org/2000/svg" width="{}" height="{}" '
        'viewBox="0 0 {} {}">'.format(width, height, width, height),
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        '<text x="{}" y="28" text-anchor="middle" font-family="DejaVu Sans,Arial,sans-serif" '
        'font-size="16" font-weight="600" fill="#111827">{}</text>'.format(width / 2, title),
        '<line x1="{}" y1="{}" x2="{}" y2="{}" stroke="#374151" stroke-width="1.5"/>'.format(
            margin_l, margin_t, margin_l, margin_t + plot_h
        ),
        '<line x1="{}" y1="{}" x2="{}" y2="{}" stroke="#374151" stroke-width="1.5"/>'.format(
            margin_l, margin_t + plot_h, margin_l + plot_w, margin_t + plot_h
        ),
        '<text x="18" y="{}" text-anchor="middle" transform="rotate(-90 18 {})" '
        'font-family="DejaVu Sans,Arial,sans-serif" font-size="12" fill="#374151">{}</text>'.format(
            margin_t + plot_h / 2, margin_t + plot_h / 2, ylabel
        ),
    ]

    for i in range(5):
        yv = vmax * i / 4.0
        y = margin_t + plot_h * (1 - i / 4.0)
        parts.append(
            '<line x1="{}" y1="{:.1f}" x2="{}" y2="{:.1f}" stroke="#e5e7eb" stroke-width="1"/>'.format(
                margin_l, y, margin_l + plot_w, y
            )
        )
        parts.append(
            '<text x="{}" y="{:.1f}" text-anchor="end" '
            'font-family="DejaVu Sans,Arial,sans-serif" font-size="11" fill="#6b7280">{}</text>'.format(
                margin_l - 8, y + 4, y_fmt.format(yv)
            )
        )

    for i, (label, val, color) in enumerate(series):
        x = margin_l + gap + i * (bar_w + gap)
        bh = plot_h * (val / vmax)
        y = margin_t + plot_h - bh
        parts.append(
            '<rect x="{:.1f}" y="{:.1f}" width="{:.1f}" height="{:.1f}" rx="4" fill="{}"/>'.format(
                x, y, bar_w, bh, color
            )
        )
        parts.append(
            '<text x="{:.1f}" y="{:.1f}" text-anchor="middle" '
            'font-family="DejaVu Sans,Arial,sans-serif" font-size="11" fill="#111827">{}</text>'.format(
                x + bar_w / 2, y - 6, y_fmt.format(val)
            )
        )
        for j, line in enumerate(label.split("\n")):
            parts.append(
                '<text x="{:.1f}" y="{:.1f}" text-anchor="middle" '
                'font-family="DejaVu Sans,Arial,sans-serif" font-size="11" fill="#374151">{}</text>'.format(
                    x + bar_w / 2, margin_t + plot_h + 18 + j * 14, line
                )
            )

    parts.append("</svg>")
    path = OUT / out_name
    path.write_text("\n".join(parts) + "\n", encoding="utf-8")
    print("wrote {}".format(path))
    return path


def load_ppl(seed):
    p = ROOT / "results" / "main_360m" / ("seed_%d" % seed) / "metrics.json"
    data = json.loads(p.read_text())
    return {m["route"]: float(m["ppl"]) for m in data}


def load_toks(path):
    if not path.exists():
        return None
    d = json.loads(path.read_text())
    v = d.get("tokens_per_sec")
    return float(v) if v is not None else None


def ppl_series(ppl):
    order = [
        ("R1", "R1\nBF16"),
        ("R2", "R2\nBF16→NVFP4"),
        ("R3", "R3\nNVFP4"),
        ("R4", "R4\nBF16→MXFP8"),
        ("R5", "R5\nMXFP8"),
    ]
    out = []
    for key, label in order:
        if key in ppl:
            out.append((label, ppl[key], COLORS[key]))
    return out


def main():
    for seed in (42, 43):
        try:
            ppl = load_ppl(seed)
        except Exception as e:
            print("skip seed{} ppl: {}".format(seed, e))
            continue
        series = ppl_series(ppl)
        if not series:
            print("skip seed{}: empty ppl".format(seed))
            continue
        _bar_svg(
            "WikiText-2 PPL（seed={} · mix73 · 26.07，越低越好）".format(seed),
            series,
            ylabel="PPL",
            out_name="fig06_ppl_seed{}.svg".format(seed),
            width=820,
            height=440,
            y_fmt="{:.2f}",
        )

    try:
        p42, p43 = load_ppl(42), load_ppl(43)
        mean = {k: (p42[k] + p43[k]) / 2.0 for k in p42 if k in p43}
        series = ppl_series(mean)
        if series:
            _bar_svg(
                "WikiText-2 PPL（seed42+43 均值 · mix73，越低越好）",
                series,
                ylabel="PPL",
                out_name="fig06_ppl_mean42_43.svg",
                width=820,
                height=440,
                y_fmt="{:.2f}",
            )
    except Exception as e:
        print("skip mean ppl:", e)

    perf = ROOT / "results" / "perf"
    pairs = [
        ("R1 train\n1GPU", load_toks(perf / "bench_R1_train_n1_seed42_best.json"), COLORS["R1"]),
        ("R1 infer\n1GPU", load_toks(perf / "bench_R1_infer_n1_seed42_best.json"), COLORS["R1"]),
        ("R3 train\n1GPU", load_toks(perf / "bench_R3_train_n1_seed42_best.json"), COLORS["R3"]),
        ("R2 infer\n1GPU", load_toks(perf / "bench_R2_infer_n1_seed42_best.json"), COLORS["R2"]),
        ("R3 infer\n1GPU", load_toks(perf / "bench_R3_infer_n1_seed42_best.json"), COLORS["R3"]),
        ("R5 train\n1GPU", load_toks(perf / "bench_R5_train_n1_seed42_best.json"), COLORS["R5"]),
        ("R4 infer\n1GPU", load_toks(perf / "bench_R4_infer_n1_seed42_best.json"), COLORS["R4"]),
        ("R5 infer\n1GPU", load_toks(perf / "bench_R5_infer_n1_seed42_best.json"), COLORS["R5"]),
        ("R1 train\n4GPU", load_toks(perf / "bench_R1_train_n4_seed42_best.json"), COLORS["R1"]),
        ("R3 train\n4GPU", load_toks(perf / "bench_R3_train_n4_seed42_best.json"), COLORS["R3"]),
        ("R5 train\n4GPU", load_toks(perf / "bench_R5_train_n4_seed42_best.json"), COLORS["R5"]),
    ]
    series = [(lab, v / 1000.0, c) for lab, v, c in pairs if v]
    if series:
        _bar_svg(
            "吞吐 microbench（seed42 mix73，k tokens/s，越高越好）",
            series,
            ylabel="k tokens/s",
            out_name="fig07_throughput.svg",
            width=1100,
            height=460,
            y_fmt="{:.0f}",
        )


if __name__ == "__main__":
    main()
