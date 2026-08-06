#!/usr/bin/env bash
set -euo pipefail
cd /workspace/k-exaone-mixed-ds4
GGUF=/workspace/models/K-EXAONE-236B-A23B-GGUF/K-EXAONE-236B-A23B-BF16.gguf

echo "[wait] for BF16 download to land"
while [ ! -f "$GGUF" ]; do sleep 20; done
sleep 5
echo "[wait] done: $(du -h "$GGUF" | cut -f1)"

# Q8_0 for the imatrix run, in parallel with the pilot build: it fits entirely
# in 4x95.6 GiB of VRAM whereas BF16 does not.
nohup ./scripts/download_sources.sh gguf-q8 > logs/dl-gguf-q8.log 2>&1 &

echo "[build] pilot"
./scripts/build_mixed_gguf.sh pilot 2>&1 | tee logs/build-pilot.log
