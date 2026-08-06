#!/usr/bin/env bash
# Download pinned K-EXAONE sources. Resumable: hf download skips complete files.
set -euo pipefail

MODELS_DIR="${MODELS_DIR:-/workspace/models}"
HF_REV_BF16="61e6d578eb102b578e5704e2916ac841df9eca0a"
HF_REV_GGUF="5bd0394e4f42c00df63e207b9c434387523a6b77"
WORKERS="${WORKERS:-8}"

log() { echo "[$(date -u +%H:%M:%S)] $*"; }

dl() {  # dl <repo> <revision> <local-subdir> <include-pattern...>
  local repo="$1" rev="$2" sub="$3"; shift 3
  local args=()
  for p in "$@"; do args+=(--include "$p"); done
  log "START $repo ($*)"
  hf download "$repo" --revision "$rev" \
    "${args[@]}" \
    --max-workers "$WORKERS" \
    --local-dir "$MODELS_DIR/$sub"
  log "DONE  $repo ($*)"
}

case "${1:-all}" in
  gguf-bf16)   dl LGAI-EXAONE/K-EXAONE-236B-A23B-GGUF "$HF_REV_GGUF" K-EXAONE-236B-A23B-GGUF '*BF16*' ;;
  gguf-q4km)   dl LGAI-EXAONE/K-EXAONE-236B-A23B-GGUF "$HF_REV_GGUF" K-EXAONE-236B-A23B-GGUF '*Q4_K_M*' ;;
  safetensors) dl LGAI-EXAONE/K-EXAONE-236B-A23B      "$HF_REV_BF16" K-EXAONE-236B-A23B      '*.safetensors' ;;
  *) echo "usage: $0 {gguf-bf16|gguf-q4km|safetensors}"; exit 2 ;;
esac
