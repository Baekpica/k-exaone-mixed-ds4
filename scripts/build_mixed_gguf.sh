#!/usr/bin/env bash
# Build a mixed-quant K-EXAONE artifact from the pinned BF16 GGUF.
#
#   ./scripts/build_mixed_gguf.sh pilot          # Q2_K experts, no imatrix
#   ./scripts/build_mixed_gguf.sh v1 <imatrix>   # IQ2_XXS experts, needs imatrix
#
# Resumable: re-running skips an artifact that already exists (--resume).
set -euo pipefail
cd "$(dirname "$0")/.."

SRC="${SRC:-/workspace/models/K-EXAONE-236B-A23B-GGUF/K-EXAONE-236B-A23B-BF16.gguf}"
OUTDIR="${OUTDIR:-/workspace/artifacts}"
THREADS="${THREADS:-$(nproc)}"
VARIANT="${1:-pilot}"

mkdir -p "$OUTDIR"

case "$VARIANT" in
  pilot)
    NAME="K-EXAONE-236B-A23B-MXQ-Q2K-Q4Edge-Q8Dense-MTPQ8-pilot-v1.gguf"
    ARGS=(--variant v1-pilot-noimatrix)
    ;;
  v1)
    IMAT="${2:?usage: $0 v1 <imatrix.dat>}"
    NAME="K-EXAONE-236B-A23B-MXQ-IQ2XXS-Q3K-Q4Edge-Q8Dense-MTPQ8-v1.gguf"
    ARGS=(--variant v1 --imatrix "$IMAT")
    ;;
  *) echo "usage: $0 {pilot|v1 <imatrix>}"; exit 2 ;;
esac

test -f "$SRC" || { echo "source GGUF missing: $SRC"; exit 1; }

echo "[build] variant=$VARIANT threads=$THREADS"
echo "[build] src=$SRC"
echo "[build] out=$OUTDIR/$NAME"

python3 converter/src/build_mixed.py \
  --source-gguf "$SRC" \
  --out "$OUTDIR/$NAME" \
  --threads "$THREADS" \
  --resume \
  "${ARGS[@]}"

echo "[verify] checking artifact against recipe"
python3 converter/src/verify_gguf.py "$OUTDIR/$NAME" \
  --variant "$([ "$VARIANT" = pilot ] && echo v1-pilot-noimatrix || echo v1)" \
  --report "manifests/verify-$VARIANT.json"
