#!/usr/bin/env bash
# Official SmolLM2 pretrained baseline (FP16 + block MXFP4 PTQ) — one-shot.
#
# Design (cold NFS-proof):
#   1) LOGIN node reads NFS (usually warmer) and streams tar over SSH
#   2) GPU node writes only to local /tmp (never rsyncs multi-GB venv from NFS)
#   3) Single python process: local venv + local HF cache + local code/wikitext
#
# Usage (from login / anywhere with NFS + SSH to GPU):
#   REMOTE_HOST=10.x.x.x bash scripts/run_pretrained_baseline.sh
#   # or on GPU already: bash scripts/run_pretrained_baseline.sh --local-only
set -euo pipefail

NFS_ROOT="${NFS_ROOT:-/home/scratch.gemsg_sw/grokbuild/mxfp4_route_compare}"
REMOTE_HOST="${REMOTE_HOST:-}"
REMOTE_USER="${REMOTE_USER:-gemsg}"
SSH_BIN="${SSH_BIN:-/usr/bin/ssh}"
STAGING="/tmp/mxfp4_pretrained_eval"
LOCAL_VENV="/tmp/mxfp4_full_venv"
HF_HOME_LOCAL="/tmp/hf_cache_mxfp4"
LOCAL_LOG="/tmp/pretrained_baseline_run.log"
MODEL_ID="HuggingFaceTB/SmolLM2-360M"

ssh_gpu() {
  "$SSH_BIN" -o BatchMode=yes -o ConnectTimeout=15 -o ServerAliveInterval=30 \
    "${REMOTE_USER}@${REMOTE_HOST}" "$@"
}

stage_to_gpu() {
  echo "[stage] full venv -> GPU ${LOCAL_VENV} (login reads NFS, GPU writes local) $(date -Is)"
  # shellcheck disable=SC2086
  tar -C "$NFS_ROOT" -cf - venv \
    | ssh_gpu "rm -rf ${LOCAL_VENV} && mkdir -p ${LOCAL_VENV} && tar -C ${LOCAL_VENV} --strip-components=1 -xf - && du -sh ${LOCAL_VENV}"
  echo "[stage] venv done $(date -Is)"

  echo "[stage] code + wikitext -> GPU ${STAGING} $(date -Is)"
  tar -C "$NFS_ROOT" -cf - \
    --exclude=venv --exclude=uv_cache --exclude=checkpoints \
    --exclude=data/fineweb_edu --exclude=logs --exclude=results \
    --exclude=.git --exclude=hf_cache --exclude=artifact_backup \
    --exclude='*.npy' \
    . \
    | ssh_gpu "rm -rf ${STAGING} && mkdir -p ${STAGING} && tar -C ${STAGING} -xf - && test -f ${STAGING}/scripts/07_eval_pretrained.py && du -sh ${STAGING}"
  # ensure wikitext present
  tar -C "$NFS_ROOT/data" -cf - wikitext2 \
    | ssh_gpu "mkdir -p ${STAGING}/data && tar -C ${STAGING}/data -xf - && du -sh ${STAGING}/data/wikitext2"
  echo "[stage] code done $(date -Is)"

  echo "[stage] HF model cache $(date -Is)"
  ssh_gpu "bash -s" <<EOS
set -euo pipefail
mkdir -p ${HF_HOME_LOCAL}/hub
if [[ ! -d ${HF_HOME_LOCAL}/hub/models--HuggingFaceTB--SmolLM2-360M ]]; then
  if [[ -d \$HOME/.cache/huggingface/hub/models--HuggingFaceTB--SmolLM2-360M ]]; then
    cp -a \$HOME/.cache/huggingface/hub/models--HuggingFaceTB--SmolLM2-360M ${HF_HOME_LOCAL}/hub/
    echo MODEL_FROM_HOME
  else
    echo "WARN: no local model cache; will need hub download"
  fi
else
  echo MODEL_OK
fi
du -sh ${HF_HOME_LOCAL}
EOS
}

run_on_gpu() {
  echo "[run] launch eval on GPU $(date -Is)"
  ssh_gpu "bash -s" <<'EOS'
set -uo pipefail
STAGING=/tmp/mxfp4_pretrained_eval
LOCAL_VENV=/tmp/mxfp4_full_venv
HF_HOME_LOCAL=/tmp/hf_cache_mxfp4
LOG=/tmp/pretrained_baseline_run.log
NFS=/home/scratch.gemsg_sw/grokbuild/mxfp4_route_compare
exec >"$LOG" 2>&1
echo "[run] host=$(hostname) date=$(date -Is) pid=$$"

export HF_HOME=$HF_HOME_LOCAL
export TRANSFORMERS_CACHE=$HF_HOME
export HF_DATASETS_CACHE=$HF_HOME/datasets
export HF_HUB_CACHE=$HF_HOME/hub
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export PYTHONUNBUFFERED=1
# Prefer local venv entirely — do not touch NFS site-packages
# shellcheck disable=SC1091
source "$LOCAL_VENV/bin/activate"
# Ensure python is the staged one
hash -r
which python
python -c "import sys; print(sys.prefix); import torch,transformers,datasets; print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), torch.cuda.get_device_name(0)); print('transformers', transformers.__version__, transformers.__file__)"

cd "$STAGING"
echo "[run] eval start $(date -Is)"
python -u scripts/07_eval_pretrained.py --config configs/main_360m.yaml
rc=$?
echo "[run] eval exit=$rc $(date -Is)"

mkdir -p "$NFS/results/pretrained" "$NFS/logs"
if [[ -f results/pretrained/pretrained_baseline.json ]]; then
  cp -f results/pretrained/pretrained_baseline.json "$NFS/results/pretrained/"
  cp -f results/pretrained/pretrained_baseline.json "$NFS/results/"
fi
if [[ -f results/pretrained_baseline.json ]]; then
  cp -f results/pretrained_baseline.json "$NFS/results/pretrained/"
  cp -f results/pretrained_baseline.json "$NFS/results/"
fi
cp -f "$LOG" "$NFS/logs/pretrained_baseline_$(date +%Y%m%d_%H%M%S).log" || true
cp -f "$LOG" "$NFS/logs/pretrained_baseline_latest.log" || true
echo RESULTS:
cat "$NFS/results/pretrained/pretrained_baseline.json" 2>/dev/null \
  || cat results/pretrained/pretrained_baseline.json 2>/dev/null || true
echo "[run] ALL DONE rc=$rc $(date -Is)"
exit $rc
EOS
}

main() {
  if [[ "${1:-}" == "--local-only" ]]; then
    # Already on GPU with staged dirs
    REMOTE_HOST=localhost
    bash -c "$(declare -f); run_on_gpu" 2>/dev/null || true
    # simpler: just exec the remote body locally
    STAGING=/tmp/mxfp4_pretrained_eval
    LOCAL_VENV=/tmp/mxfp4_full_venv
    export HF_HOME=/tmp/hf_cache_mxfp4
    export TRANSFORMERS_CACHE=$HF_HOME HF_DATASETS_CACHE=$HF_HOME/datasets HF_HUB_CACHE=$HF_HOME/hub
    export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0} PYTHONUNBUFFERED=1
    # shellcheck disable=SC1091
    source "$LOCAL_VENV/bin/activate"
    cd "$STAGING"
    exec python -u scripts/07_eval_pretrained.py --config configs/main_360m.yaml
  fi

  if [[ -z "$REMOTE_HOST" ]]; then
    echo "Set REMOTE_HOST=gpu-ip (or pass --local-only on the GPU node)" >&2
    exit 2
  fi

  echo "[main] REMOTE=${REMOTE_USER}@${REMOTE_HOST} NFS=$NFS_ROOT $(date -Is)"
  stage_to_gpu
  run_on_gpu
  rc=$?
  echo "[main] finished rc=$rc $(date -Is)"
  # print results from NFS if present
  if [[ -f "$NFS_ROOT/results/pretrained/pretrained_baseline.json" ]]; then
    cat "$NFS_ROOT/results/pretrained/pretrained_baseline.json"
  fi
  exit $rc
}

main "$@"
