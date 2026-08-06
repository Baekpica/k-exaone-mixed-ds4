#!/usr/bin/env bash
# Split each artifact below the Hub's 50 GB per-file limit, then upload.
#
# The Hub rejected both 85.6 GiB and 87.8 GiB uploads with
# "Max individual file size for files: 50.0GB". 40 G shards give three parts per
# artifact with headroom, and llama.cpp loads a split GGUF by pointing at part
# 00001 -- no merge step needed on the consumer side.
set -euo pipefail
cd "$(dirname "$0")/.."

REPO="${REPO:-Baekpica/K-EXAONE-236B-A23B-Mixed-Quant-GGUF}"
SPLIT_DIR="${SPLIT_DIR:-/workspace/artifacts/split}"
MAX="${MAX:-40G}"
export HF_HUB_DISABLE_XET=1

mkdir -p "$SPLIT_DIR"
ts() { TZ=Asia/Seoul date '+%H:%M:%S'; }

for ART in "$@"; do
  BASE=$(basename "$ART" .gguf)
  PREFIX="$SPLIT_DIR/$BASE"

  if ! ls "$PREFIX"-*-of-*.gguf >/dev/null 2>&1; then
    echo "[$(ts)] splitting $BASE at $MAX"
    llama.cpp/build-sm120/bin/llama-gguf-split --split --split-max-size "$MAX" \
      "$ART" "$PREFIX"
  else
    echo "[$(ts)] shards already present for $BASE"
  fi
  ls -la "$PREFIX"-*-of-*.gguf | awk '{printf "  %.2f GiB  %s\n", $5/2^30, $9}'

  for SH in "$PREFIX"-*-of-*.gguf; do
    BN=$(basename "$SH")
    echo "[$(ts)] uploading $BN"
    for attempt in 1 2 3 4 5; do
      if hf upload "$REPO" "$SH" "$BN" --repo-type model \
           --commit-message "artifact shard $BN"; then
        echo "[$(ts)] OK $BN"
        break
      fi
      echo "[$(ts)] attempt $attempt failed for $BN; retry in 30s"
      sleep 30
    done
  done
done
echo "[$(ts)] done"
