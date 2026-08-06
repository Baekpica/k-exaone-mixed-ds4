#!/usr/bin/env bash
# Retry a large artifact upload. The Xet backend timed out on the first attempt
# while llama-imatrix was saturating the same network filesystem; Xet dedups, so
# a retry resumes rather than restarting. HF_HUB_DISABLE_XET=1 falls back to
# classic LFS multipart if Xet keeps failing.
set -euo pipefail
cd "$(dirname "$0")/.."
ART="${1:?usage: $0 <artifact.gguf> [repo]}"
REPO="${2:-Baekpica/K-EXAONE-236B-A23B-Mixed-Quant-GGUF}"

for attempt in 1 2 3; do
  echo "[attempt $attempt] xet enabled"
  if hf upload "$REPO" "$ART" "$(basename "$ART")" --repo-type model \
       --commit-message "mixed-quant artifact"; then
    echo "[ok] uploaded"; exit 0
  fi
  echo "[attempt $attempt] failed; retrying in 60s"
  sleep 60
done

echo "[fallback] retrying with Xet disabled (classic LFS multipart)"
HF_HUB_DISABLE_XET=1 hf upload "$REPO" "$ART" "$(basename "$ART")" \
  --repo-type model --commit-message "mixed-quant artifact"
