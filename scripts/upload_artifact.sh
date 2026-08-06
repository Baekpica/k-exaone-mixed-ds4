#!/usr/bin/env bash
# Publish a mixed-quant artifact to the Hub (public, under the logged-in user).
#
#   ./scripts/upload_artifact.sh <artifact.gguf> <repo-name>
#
# Uploads the GGUF plus everything needed to reproduce or audit it: recipe,
# tensor-type file, build manifest, verification report, projection, and the
# upstream LICENSE. K-EXAONE is licensed under the k-exaone license, so the
# license file and attribution travel with the artifact.
set -euo pipefail
cd "$(dirname "$0")/.."

ART="${1:?usage: $0 <artifact.gguf> <repo-name>}"
REPO_NAME="${2:?usage: $0 <artifact.gguf> <repo-name>}"
OWNER="$(hf auth whoami 2>/dev/null | sed -n 's/^user=\([^ ]*\).*/\1/p')"
REPO="${OWNER}/${REPO_NAME}"

test -f "$ART" || { echo "artifact missing: $ART"; exit 1; }

STAGE=$(mktemp -d /workspace/artifacts/.upload-XXXXXX)
trap 'rm -rf "$STAGE"' EXIT

echo "[stage] collecting provenance"
cp manifests/quant-recipe-v1.yaml               "$STAGE/"
cp manifests/tensor-inventory.json              "$STAGE/" 2>/dev/null || true
cp manifests/projection-*.json                  "$STAGE/" 2>/dev/null || true
cp manifests/verify-*.json                      "$STAGE/" 2>/dev/null || true
cp manifests/artifact-manifest.json             "$STAGE/" 2>/dev/null || true
cp manifests/build-manifest.json                "$STAGE/" 2>/dev/null || true
cp "${ART%.gguf}.manifest.json"                 "$STAGE/" 2>/dev/null || true
cp "${ART%.gguf}.tensor-types.txt"              "$STAGE/" 2>/dev/null || true
cp /workspace/models/K-EXAONE-236B-A23B/LICENSE "$STAGE/LICENSE"
cp reports/MODEL_CARD.md                        "$STAGE/README.md"
cp fixtures/calibration.composition.json        "$STAGE/" 2>/dev/null || true

echo "[repo] $REPO (public)"
hf repos create "$REPO" --repo-type model --exist-ok >/dev/null

echo "[upload] provenance files"
hf upload "$REPO" "$STAGE" . --repo-type model \
  --commit-message "recipe, manifests, verification report, license"

echo "[upload] $(basename "$ART") ($(du -h "$ART" | cut -f1)) -- this takes a while"
hf upload "$REPO" "$ART" "$(basename "$ART")" --repo-type model \
  --commit-message "mixed-quant artifact"

echo "[done] https://huggingface.co/$REPO"
