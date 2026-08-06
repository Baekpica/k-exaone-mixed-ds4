#!/usr/bin/env bash
# Upload the large artifacts with Xet disabled.
#
# The Xet backend timed out three times on this host, always inside
# new_upload_commit, with zero bytes transferred. HF_HUB_DISABLE_XET=1 falls
# back to classic LFS multipart, which is slower but does not depend on the
# Xet control plane.
set -euo pipefail
cd "$(dirname "$0")/.."

REPO="${REPO:-Baekpica/K-EXAONE-236B-A23B-Mixed-Quant-GGUF}"
export HF_HUB_DISABLE_XET=1

for ART in "$@"; do
  BN=$(basename "$ART")
  echo "[$(TZ=Asia/Seoul date '+%H:%M:%S')] uploading $BN ($(du -h "$ART" | cut -f1))"
  for attempt in 1 2 3 4 5; do
    if hf upload "$REPO" "$ART" "$BN" --repo-type model \
         --commit-message "mixed-quant artifact: $BN"; then
      echo "[$(TZ=Asia/Seoul date '+%H:%M:%S')] OK $BN"
      break
    fi
    echo "[$(TZ=Asia/Seoul date '+%H:%M:%S')] attempt $attempt failed for $BN; retry in 30s"
    sleep 30
  done
done
echo "[$(TZ=Asia/Seoul date '+%H:%M:%S')] all uploads finished"
