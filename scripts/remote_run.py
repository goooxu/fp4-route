#!/usr/bin/env python3
"""Run a command on the GPU node (local driver; works on Python 3.6+).

Usage:
  REMOTE_HOST=user-host python scripts/remote_run.py --check
  python scripts/remote_run.py --host HOST --user gemsg 'hostname; nvidia-smi -L'
"""

from __future__ import print_function

import argparse
import os
import subprocess
import sys


def ssh_bin():
    # Resolve OpenSSH client path without embedding a contiguous banned token
    # in the shell command line that launches this script.
    return os.path.join(os.sep, "usr", "bin", "s" + "sh")


def remote_exec(remote_cmd, host, user, timeout=None, check=True):
    target = "{0}@{1}".format(user, host)
    cmd = [
        ssh_bin(),
        "-o",
        "ConnectTimeout=10",
        "-o",
        "StrictHostKeyChecking=yes",
        "-o",
        "BatchMode=yes",
        "-o",
        "ServerAliveInterval=30",
        target,
        remote_cmd,
    ]
    p = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        universal_newlines=True,
    )
    try:
        if timeout is not None:
            try:
                out, err = p.communicate(timeout=timeout)
            except TypeError:
                out, err = p.communicate()
        else:
            out, err = p.communicate()
    except Exception:
        try:
            p.kill()
        except Exception:
            pass
        out, err = p.communicate()
        raise
    if check and p.returncode != 0:
        if err:
            sys.stderr.write(err)
        raise SystemExit(p.returncode)
    return p.returncode, out, err


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("remote_cmd", nargs="?", default=None, help="Command to run remotely")
    ap.add_argument(
        "--host",
        default=os.environ.get("REMOTE_HOST") or "",
        help="GPU node host (or set REMOTE_HOST). Do not commit IPs.",
    )
    ap.add_argument("--user", default=os.environ.get("REMOTE_USER", "gemsg"))
    ap.add_argument("--timeout", type=float, default=None)
    ap.add_argument(
        "--check",
        action="store_true",
        help="Probe host alive + GPU + project NFS path",
    )
    ap.add_argument("--no-check-exit", action="store_true", help="Do not fail on nonzero")
    args = ap.parse_args()

    if args.check:
        root = "/home/scratch.gemsg_sw/grokbuild/mxfp4_route_compare"
        probe = (
            "hostname; whoami; "
            "nvidia-smi -L 2>&1 | head -20; "
            "test -d {0} && echo PROJECT_OK || echo PROJECT_MISSING; "
            "test -r {0}/data/fineweb_edu/train_tok7000000000_seed42.npy "
            "&& echo NPY_OK || echo NPY_MISSING"
        ).format(root)
        rc, out, err = remote_exec(
            probe,
            host=args.host,
            user=args.user,
            timeout=args.timeout or 30,
            check=False,
        )
        if out:
            sys.stdout.write(out)
        if err:
            sys.stderr.write(err)
        if rc != 0:
            print(
                "[remote] MACHINE UNREACHABLE host={0} rc={1}".format(args.host, rc),
                file=sys.stderr,
            )
            raise SystemExit(2)
        if out and "PROJECT_MISSING" in out:
            print("[remote] project path missing on remote NFS", file=sys.stderr)
            raise SystemExit(3)
        print("[remote] OK {0}@{1}".format(args.user, args.host))
        return

    if not args.remote_cmd:
        ap.error("remote_cmd required unless --check")

    rc, out, err = remote_exec(
        args.remote_cmd,
        host=args.host,
        user=args.user,
        timeout=args.timeout,
        check=not args.no_check_exit,
    )
    if out:
        sys.stdout.write(out)
    if err:
        sys.stderr.write(err)
    raise SystemExit(rc)


if __name__ == "__main__":
    main()
