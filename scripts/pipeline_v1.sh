#!/usr/bin/env bash
# Stage 2: imatrix -> v1 artifact -> verify.
# Waits for both the pilot build and the Q8_0 download, then runs unattended.
set -euo pipefail
cd /workspace/k-exaone-mixed-ds4

Q8=/workspace/models/K-EXAONE-236B-A23B-GGUF/K-EXAONE-236B-A23B-Q8_0.gguf
IMAT=/workspace/artifacts/k-exaone-236b.imatrix
PILOT=/workspace/artifacts/K-EXAONE-236B-A23B-MXQ-Q2K-Q4Edge-Q8Dense-MTPQ8-pilot-v1.gguf

ts() { TZ=Asia/Seoul date '+%H:%M:%S KST'; }

echo "[$(ts)] waiting for Q8_0 download"
while [ ! -f "$Q8" ]; do sleep 30; done
echo "[$(ts)] Q8_0 ready: $(du -h "$Q8" | cut -f1)"

# Don't contend with the pilot build for the network filesystem.
echo "[$(ts)] waiting for the pilot build to finish"
while pgrep -f "llama-quantize.*pilot" >/dev/null; do sleep 30; done
echo "[$(ts)] pilot build done"

echo "[$(ts)] imatrix run"
# 3000 chunks x 512 tokens ~= 1.5M tokens. Generous for an imatrix and bounded
# in wall clock; partial results are written every 25 chunks either way.
CHUNKS=3000 ./scripts/gen_imatrix.sh 2>&1 | tail -40
echo "[$(ts)] imatrix done: $(ls -la "$IMAT" | awk '{print $5}') bytes"

echo "[$(ts)] building v1"
./scripts/build_mixed_gguf.sh v1 "$IMAT"
echo "[$(ts)] v1 complete"
