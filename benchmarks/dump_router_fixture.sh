#!/usr/bin/env bash
# Dump per-layer router decisions from a high-precision model.
#
# This is the fixture the work order asks for in D2-3: selected expert indices
# and their weights, per layer, for a fixed prompt. Router semantics are the
# easiest thing to get subtly wrong when porting -- where sigmoid is applied,
# whether normalization happens before or after top-k, where the 2.5 scaling
# lands, how exp_probs_b is folded in. Any of those produces a *different set of
# experts*, which no end-to-end output check reliably catches.
#
# It has to be generated here, on the development host: the reference is the
# 234.7 GiB Q8_0 (or 441.7 GiB BF16) model, and neither fits a 128 GB DGX Spark.
set -euo pipefail
cd "$(dirname "$0")/.."

MODEL="${MODEL:-/workspace/models/K-EXAONE-236B-A23B-GGUF/K-EXAONE-236B-A23B-Q8_0.gguf}"
OUT="${OUT:-fixtures/layer-checksums}"
PROMPT="${PROMPT:-한국의 수도는}"
NGL="${NGL:-99}"

mkdir -p "$OUT"
LABEL=$(basename "$MODEL" .gguf)

echo "[dump] model=$MODEL"
echo "[dump] prompt=$PROMPT"

# ffn_moe_topk        -- selected expert indices  (i32, [n_expert_used, n_tokens])
# ffn_moe_weights_norm-- weights after top-k normalization
# ffn_moe_probs       -- sigmoid gate output before selection
llama.cpp/build-sm120/bin/llama-eval-callback \
  -m "$MODEL" -ngl "$NGL" -sm layer -c 512 -n 1 --temp 0 \
  -p "$PROMPT" \
  --tensor-filter 'ffn_moe_topk|ffn_moe_weights_norm|ffn_moe_probs' \
  > "$OUT/router-$LABEL.raw" 2>&1

echo "[dump] raw -> $OUT/router-$LABEL.raw ($(du -h "$OUT/router-$LABEL.raw" | cut -f1))"
python3 benchmarks/parse_router_dump.py "$OUT/router-$LABEL.raw" \
  --out "$OUT/router-$LABEL.json" --prompt "$PROMPT"
