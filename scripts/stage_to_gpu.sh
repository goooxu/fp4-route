#!/usr/bin/env bash
# Stream full project venv + code + optional data subsets from NFS (login) to GPU /tmp.
# Usage:
#   REMOTE_HOST=10.x.x.x bash scripts/stage_to_gpu.sh
#   REMOTE_HOST=... bash scripts/stage_to_gpu.sh --with-wikitext --with-init
set -euo pipefail

NFS_ROOT="${NFS_ROOT:-/home/scratch.gemsg_sw/grokbuild/mxfp4_route_compare}"
REMOTE_HOST="${REMOTE_HOST:-}"
REMOTE_USER="${REMOTE_USER:-gemsg}"
SSH_BIN="${SSH_BIN:-/usr/bin/ssh}"
LOCAL_VENV="/tmp/mxfp4_full_venv"
STAGING="/tmp/mxfp4_work"

WITH_WIKITEXT=0
WITH_INIT=0
for a in "$@"; do
  case "$a" in
    --with-wikitext) WITH_WIKITEXT=1 ;;
    --with-init) WITH_INIT=1 ;;
  esac
done

if [[ -z "$REMOTE_HOST" ]]; then
  echo "Set REMOTE_HOST=<gpu-ip>" >&2
  exit 2
fi

ssh_gpu() {
  "$SSH_BIN" -o BatchMode=yes -o ConnectTimeout=15 -o ServerAliveInterval=30 \
    "${REMOTE_USER}@${REMOTE_HOST}" "$@"
}

echo "[stage] host=$REMOTE_HOST $(date -Is)"
echo "[stage] venv -> $LOCAL_VENV"
tar -C "$NFS_ROOT" -cf - venv \
  | ssh_gpu "rm -rf ${LOCAL_VENV} && mkdir -p ${LOCAL_VENV} && tar -C ${LOCAL_VENV} --strip-components=1 -xf - && du -sh ${LOCAL_VENV}"

echo "[stage] code -> $STAGING"
tar -C "$NFS_ROOT" -cf - \
  --exclude=venv --exclude=uv_cache --exclude=checkpoints \
  --exclude=data --exclude=logs --exclude=results \
  --exclude=.git --exclude=hf_cache --exclude=artifact_backup \
  --exclude='*.npy' \
  mxfp4_lib scripts configs tests README.md EXPERIMENT_SUMMARY.md RUN_STATUS.md .gitignore \
  2>/dev/null \
  | ssh_gpu "rm -rf ${STAGING} && mkdir -p ${STAGING} && tar -C ${STAGING} -xf - && ln -sfn ${LOCAL_VENV} ${STAGING}/venv"

if [[ "$WITH_WIKITEXT" == "1" ]]; then
  echo "[stage] wikitext2"
  tar -C "$NFS_ROOT/data" -cf - wikitext2 \
    | ssh_gpu "mkdir -p ${STAGING}/data && tar -C ${STAGING}/data -xf -"
fi

if [[ "$WITH_INIT" == "1" ]] && [[ -d "$NFS_ROOT/checkpoints/seed_42/init_model" ]]; then
  echo "[stage] init_model"
  tar -C "$NFS_ROOT/checkpoints/seed_42" -cf - init_model \
    | ssh_gpu "mkdir -p ${STAGING}/checkpoints/seed_42 && tar -C ${STAGING}/checkpoints/seed_42 -xf -"
elif [[ "$WITH_INIT" == "1" ]] && [[ -d "$NFS_ROOT/checkpoints/init_model" ]]; then
  tar -C "$NFS_ROOT/checkpoints" -cf - init_model \
    | ssh_gpu "mkdir -p ${STAGING}/checkpoints && tar -C ${STAGING}/checkpoints -xf -"
fi

ssh_gpu "test -x ${LOCAL_VENV}/bin/python && ${LOCAL_VENV}/bin/python -c 'import torch; print(torch.__version__, torch.cuda.is_available())'"
echo "[stage] DONE $(date -Is)"
echo "On GPU: export PATH=${LOCAL_VENV}/bin:\$PATH; cd ${STAGING}"
