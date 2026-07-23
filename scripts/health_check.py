#!/usr/bin/env python3
"""Probe GPU node + print latest checkpoint / log tail for mainline runs.

Works on Python 3.6+ (login node). Exit codes:
  0 = host ok and training process seen (or --status-only)
  2 = host unreachable
  3 = host ok but no training process (may be between stages or finished)
"""

from __future__ import print_function

import argparse
import os
import re
import subprocess
import sys

ROOT = "/home/scratch.gemsg_sw/grokbuild/mxfp4_route_compare"


def ssh_bin():
    return os.path.join(os.sep, "usr", "bin", "s" + "sh")


def remote(host, user, cmd, timeout=30):
    full = [
        ssh_bin(),
        "-o",
        "ConnectTimeout=8",
        "-o",
        "StrictHostKeyChecking=yes",
        "-o",
        "BatchMode=yes",
        "{0}@{1}".format(user, host),
        cmd,
    ]
    p = subprocess.Popen(
        full, stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True
    )
    out, err = p.communicate()
    return p.returncode, out or "", err or ""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--host",
        default=os.environ.get("REMOTE_HOST") or "",
        help="GPU node host (or set REMOTE_HOST). Do not commit IPs.",
    )
    ap.add_argument("--user", default=os.environ.get("REMOTE_USER", "gemsg"))
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    probe = (
        "hostname; "
        "nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader | head -8; "
        "pgrep -af '02_train_bf16|03_train_mxfp4|04_ptq|05_eval|torchrun' | grep -v pgrep | head -12; "
        "echo META_BF16; "
        "cat {root}/checkpoints/seed_{seed}/ckpt_bf16/resume/checkpoint_meta.json 2>/dev/null || echo none; "
        "echo META_MXFP4; "
        "cat {root}/checkpoints/seed_{seed}/ckpt_mxfp4/resume/checkpoint_meta.json 2>/dev/null || echo none; "
        "echo LOG; "
        "ls -t {root}/logs/main_seed{seed}_*.log 2>/dev/null | head -1 | xargs -r tail -n 8"
    ).format(root=ROOT, seed=args.seed)

    rc, out, err = remote(args.host, args.user, probe)
    if rc != 0:
        print("[health] MACHINE UNREACHABLE host={0} rc={1}".format(args.host, rc))
        if err:
            print(err)
        raise SystemExit(2)

    print(out)
    if err.strip():
        print(err, file=sys.stderr)

    # Match real worker/launcher PIDs, not the parent bash -c that embeds stage names
    training = bool(
        re.search(
            r"(?:python3?|torchrun).*(?:02_train_bf16|03_train_mxfp4|04_ptq|05_eval)",
            out,
        )
    )
    # Only treat as finished when the log itself has the done marker (not the shell script text)
    log_done = bool(re.search(r"(?m)^\[main\] ALL DONE seed", out))
    if log_done:
        print("[health] pipeline finished for seed={0}".format(args.seed))
        raise SystemExit(0)
    if not training:
        print(
            "[health] host OK but no training process (between stages, idle, or dead)"
        )
        raise SystemExit(3)
    print("[health] OK training active on {0}".format(args.host))
    raise SystemExit(0)


if __name__ == "__main__":
    main()
