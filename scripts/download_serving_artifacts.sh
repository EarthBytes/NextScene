#!/bin/sh
# Download transformer checkpoint tarball before boot (Render builds from git without /models).
set -e

MODEL_DIR="${TRANSFORMER_MODEL_PATH:-models/transformer-full-v2}"
CHECKPOINT="${MODEL_DIR}/best.pt"

if [ -f "$CHECKPOINT" ]; then
  echo "Serving checkpoint already present: $CHECKPOINT"
  exit 0
fi

if [ -z "${SERVING_ARTIFACT_URL:-}" ]; then
  echo "SERVING_ARTIFACT_URL not set; skipping model download."
  exit 0
fi

echo "Downloading serving artifacts from SERVING_ARTIFACT_URL ..."
curl -fsSL "$SERVING_ARTIFACT_URL" -o /tmp/serving-artifacts.tar.gz
mkdir -p "$(dirname "$MODEL_DIR")"
tar -xzf /tmp/serving-artifacts.tar.gz -C /app
rm -f /tmp/serving-artifacts.tar.gz

if [ ! -f "$CHECKPOINT" ]; then
  echo "Download finished but checkpoint missing at $CHECKPOINT" >&2
  exit 1
fi

echo "Serving artifacts ready at $MODEL_DIR"
