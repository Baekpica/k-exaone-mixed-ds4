#!/usr/bin/env bash
# Assemble and push the DGX Spark handoff bundle to the private HF bucket.
#
# What goes in is decided by one question: can the sm_121 stage regenerate this
# cheaply? The importance matrix cannot -- it needs the 234.7 GiB Q8_0 model,
# which does not fit a 128 GB Spark, and took ~40 minutes of four-GPU time. Same
# for the Q8_0 greedy reference and the router fixture. Those are the reason
# this bundle exists; the manifests and reports come along because they are
# small and pin everything together.
set -euo pipefail
cd "$(dirname "$0")/.."

BUCKET="${BUCKET:-Baekpica/k-exaone-spark-handoff}"
STAGE=$(mktemp -d /workspace/artifacts/.handoff-XXXXXX)
trap 'rm -rf "$STAGE"' EXIT

mkdir -p "$STAGE"/{manifests,fixtures,benchmarks,reports,ds4}

echo "[stage] manifests"
cp manifests/*.json manifests/*.yaml "$STAGE/manifests/" 2>/dev/null || true

echo "[stage] fixtures -- not regenerable on a 128 GB device"
cp fixtures/prompts.jsonl                    "$STAGE/fixtures/"
cp fixtures/greedy-reference-q8.json         "$STAGE/fixtures/" 2>/dev/null || true
cp fixtures/calibration.composition.json     "$STAGE/fixtures/" 2>/dev/null || true
cp -r fixtures/layer-checksums               "$STAGE/fixtures/" 2>/dev/null || true
# the corpus itself, so the imatrix is reproducible
cp fixtures/calibration.txt                  "$STAGE/fixtures/" 2>/dev/null || true

echo "[stage] benchmark results"
cp benchmarks/results/*.json "$STAGE/benchmarks/" 2>/dev/null || true

echo "[stage] reports"
cp reports/*.md "$STAGE/reports/"

echo "[stage] ds4 patch (in case the fork is not reachable)"
git -C ds4 format-patch upstream/main --stdout > "$STAGE/ds4/exaone-moe.patch" 2>/dev/null || true
git -C ds4 log --oneline upstream/main..HEAD > "$STAGE/ds4/commits.txt" 2>/dev/null || true

echo "[stage] importance matrix ($(du -h /workspace/artifacts/k-exaone-236b.imatrix | cut -f1))"
cp /workspace/artifacts/k-exaone-236b.imatrix "$STAGE/"

cat > "$STAGE/README.md" <<'EOF'
# K-EXAONE -> DGX Spark handoff

Produced on the RTX PRO 6000 (`sm_120`) development host. Nothing here has run
on GB10.

- `k-exaone-236b.imatrix` — importance matrix used to build v1. 775 chunks,
  396,800 tokens, every one of 128 experts activated in all 47 MoE layers
  (lowest per-expert count 1,159). Regenerating it needs the 234.7 GiB Q8_0
  model, which does not fit 128 GB.
- `fixtures/greedy-reference-q8.json` — 32 fixtures run on the official Q8_0
  build. This is the correctness oracle for anything downstream.
- `fixtures/layer-checksums/` — per-layer router decisions (selected expert
  indices and weights) from the Q8_0 reference.
- `fixtures/calibration.txt` + `calibration.composition.json` — the corpus and
  its provenance, so the imatrix is reproducible.
- `manifests/` — pinned revisions, tensor inventory, size projections,
  per-tensor verification reports, imatrix coverage.
- `benchmarks/` — raw Phase C results for both artifacts, failures included.
- `reports/DGX-SPARK-HANDOFF.md` — start here.
- `ds4/exaone-moe.patch` — the ds4 loader work as a patch against upstream/main.

Artifacts themselves are public:
`Baekpica/K-EXAONE-236B-A23B-Mixed-Quant-GGUF`.
EOF

echo "[bucket] $BUCKET"
du -sh "$STAGE"
hf buckets cp -r "$STAGE"/* "hf://buckets/$BUCKET/" 2>&1 | tail -5 || \
  hf buckets sync "$STAGE" "hf://buckets/$BUCKET/" 2>&1 | tail -5
echo "[done] https://huggingface.co/buckets/$BUCKET"
