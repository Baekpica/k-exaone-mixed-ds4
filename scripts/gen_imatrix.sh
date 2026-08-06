#!/usr/bin/env bash
# Generate the importance matrix required by IQ2_XXS.
#
# Source model: Q8_0 rather than BF16. BF16 is 441.7 GiB and does not fit the
# 4 x 95.6 GiB = 382 GiB of VRAM on this host, so it would have to run partly on
# CPU. Q8_0 is 234.7 GiB, fits entirely across the four GPUs, and is effectively
# lossless for activation statistics.
#
# Writes partial results periodically, so a run can be stopped early and the
# imatrix so far is still usable.
set -euo pipefail
cd "$(dirname "$0")/.."

MODEL="${MODEL:-/workspace/models/K-EXAONE-236B-A23B-GGUF/K-EXAONE-236B-A23B-Q8_0.gguf}"
CALIB="${CALIB:-fixtures/calibration.txt}"
OUT="${OUT:-/workspace/artifacts/k-exaone-236b.imatrix}"
CTX="${CTX:-512}"
NGL="${NGL:-99}"
CHUNKS="${CHUNKS:-0}"          # 0 = consume the whole calibration file
SAVE_EVERY="${SAVE_EVERY:-25}"

test -f "$MODEL" || { echo "model missing: $MODEL"; exit 1; }
test -f "$CALIB" || { echo "calibration corpus missing: $CALIB"; exit 1; }

mkdir -p "$(dirname "$OUT")"

echo "[imatrix] model  $MODEL ($(du -h "$MODEL" | cut -f1))"
echo "[imatrix] calib  $CALIB ($(du -h "$CALIB" | cut -f1))"
echo "[imatrix] out    $OUT"

ARGS=(
  -m "$MODEL"
  -f "$CALIB"
  -o "$OUT"
  -c "$CTX"
  -ngl "$NGL"
  --output-frequency "$SAVE_EVERY"
  -sm layer                 # split across the 4 GPUs by layer
)
[ "$CHUNKS" -gt 0 ] && ARGS+=(--chunks "$CHUNKS")

exec llama.cpp/build-sm120/bin/llama-imatrix "${ARGS[@]}"
