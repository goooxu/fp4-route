#!/usr/bin/env python3
"""Probe Transformer Engine / hardware FP4 (NVFP4/MXFP8) capability on this GPU."""

from __future__ import annotations

import json
import sys
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from mxfp4_lib.util import ensure_dir, load_cfg


def main() -> int:
    ap_cfg = "configs/main_360m.yaml"
    if len(sys.argv) > 1 and sys.argv[1].startswith("--"):
        # allow --config PATH
        import argparse

        p = argparse.ArgumentParser()
        p.add_argument("--config", default="configs/main_360m.yaml")
        args = p.parse_args()
        ap_cfg = args.config
    cfg = load_cfg(ap_cfg)
    out_dir = ensure_dir(Path(cfg["_root"]) / "results" / "perf")
    out: dict = {
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
        "device_name": None,
        "capability": None,
        "transformer_engine": None,
        "recipes": {},
        "te_linear_smoke": {},
        "errors": [],
    }
    if torch.cuda.is_available():
        out["device_name"] = torch.cuda.get_device_name(0)
        out["capability"] = list(torch.cuda.get_device_capability(0))

    try:
        import transformer_engine as te_pkg

        out["transformer_engine"] = getattr(te_pkg, "__version__", "unknown")
    except Exception as e:
        out["errors"].append(f"import transformer_engine: {type(e).__name__}: {e}")
        path = out_dir / "capability.json"
        try:
            with open(path, "w") as f:
                json.dump(out, f, indent=2)
        except OSError:
            path = Path("/tmp/capability.json")
            with open(path, "w") as f:
                json.dump(out, f, indent=2)
        print(json.dumps(out, indent=2))
        print(f"[probe] wrote {path} (TE missing)")
        return 2

    try:
        import transformer_engine.pytorch as te
        from transformer_engine.common.recipe import (
            DelayedScaling,
            Format,
            MXFP8BlockScaling,
            NVFP4BlockScaling,
        )
    except Exception as e:
        out["errors"].append(f"import te.pytorch/recipe: {type(e).__name__}: {e}")
        path = out_dir / "capability.json"
        with open(path, "w") as f:
            json.dump(out, f, indent=2)
        print(json.dumps(out, indent=2))
        return 2

    recipes = {
        "DelayedScaling": DelayedScaling(fp8_format=Format.HYBRID),
        "MXFP8BlockScaling": MXFP8BlockScaling(fp8_format=Format.E4M3),
        "NVFP4BlockScaling": NVFP4BlockScaling(),
    }

    device = torch.device("cuda")
    for name, recipe in recipes.items():
        rec_out: dict = {"ok_forward": False, "ok_backward": False, "error": None}
        try:
            linear = te.Linear(512, 512, bias=True).to(device).bfloat16()
            x = torch.randn(64, 512, device=device, dtype=torch.bfloat16)
            with te.autocast(enabled=True, recipe=recipe):
                y = linear(x)
            rec_out["ok_forward"] = True
            y.mean().backward()
            rec_out["ok_backward"] = True
            del linear, x, y
            torch.cuda.empty_cache()
        except Exception as e:
            rec_out["error"] = f"{type(e).__name__}: {e}"
            rec_out["traceback"] = traceback.format_exc(limit=5)
        out["recipes"][name] = rec_out
        print(f"[probe] recipe {name}: {rec_out}")

    # Prefer NVFP4 for labeling
    preferred = None
    for cand in ("NVFP4BlockScaling", "MXFP8BlockScaling", "DelayedScaling"):
        if out["recipes"].get(cand, {}).get("ok_forward"):
            preferred = cand
            break
    out["preferred_recipe"] = preferred
    out["te_linear_smoke"] = {
        "preferred": preferred,
        "hw_fp4_available": preferred in ("NVFP4BlockScaling",) if preferred else False,
        "hw_lowprec_available": preferred is not None,
    }

    path = out_dir / "capability.json"
    try:
        with open(path, "w") as f:
            json.dump(out, f, indent=2)
        print(f"[probe] wrote {path}")
    except OSError as e:
        # Docker root vs NFS user ownership
        alt = Path("/tmp/capability.json")
        with open(alt, "w") as f:
            json.dump(out, f, indent=2)
        print(f"[probe] could not write {path} ({e}); wrote {alt}")
    print(json.dumps(out, indent=2))
    return 0 if preferred else 3


if __name__ == "__main__":
    raise SystemExit(main())
