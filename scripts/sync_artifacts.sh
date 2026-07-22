#!/usr/bin/env bash
# Sync durable experiment artifacts between a remote node and a local backup dir.
# Usage:
#   REMOTE=user@host REMOTE_DIR=/tmp/fp4_route LOCAL_DIR=./artifact_backup \
#     bash scripts/sync_artifacts.sh pull|push
#
# Does NOT put host IPs in the repo — pass REMOTE via env.
# Default REMOTE_DIR is /tmp/fp4_route (many nodes lack writable /raid/tmp).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
REMOTE="${REMOTE:?set REMOTE=user@host}"
REMOTE_DIR="${REMOTE_DIR:-/tmp/fp4_route}"
LOCAL_DIR="${LOCAL_DIR:-$ROOT/artifact_backup}"
MODE="${1:?usage: sync_artifacts.sh pull|push}"

mkdir -p "$LOCAL_DIR"

# Large but essential: token caches, checkpoints (incl. resume/), results
INCLUDES=(
  --include 'data/'
  --include 'data/fineweb_edu/***'
  --include 'data/tokenizer/***'
  --include 'data/wikitext2/***'
  --include 'checkpoints/***'
  --include 'results/***'
  --include 'configs/***'
  --include 'mxfp4_lib/***'
  --include 'scripts/***'
  --include 'tests/***'
  --include 'requirements.txt'
  --include 'README.md'
  --include 'EXPERIMENT_SUMMARY.md'
  --include '.gitignore'
  --exclude '*'
)

case "$MODE" in
  pull)
    echo "[sync] PULL $REMOTE:$REMOTE_DIR → $LOCAL_DIR"
    rsync -azP --partial --append-verify \
      "${INCLUDES[@]}" \
      "$REMOTE:$REMOTE_DIR/" "$LOCAL_DIR/"
    ;;
  push)
    echo "[sync] PUSH $LOCAL_DIR → $REMOTE:$REMOTE_DIR"
    rsync -azP --partial \
      "${INCLUDES[@]}" \
      "$LOCAL_DIR/" "$REMOTE:$REMOTE_DIR/"
    ;;
  *)
    echo "usage: sync_artifacts.sh pull|push"; exit 1
    ;;
esac
echo "[sync] done"
