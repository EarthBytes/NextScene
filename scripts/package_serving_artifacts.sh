#!/bin/sh
# Package transformer checkpoint for upload (GitHub Release, S3, etc.).
set -e

MODEL_DIR="${1:-models/transformer-full-v2}"
OUTPUT="${2:-serving-artifacts.tar.gz}"

if [ ! -f "${MODEL_DIR}/best.pt" ]; then
  echo "Missing ${MODEL_DIR}/best.pt — train first: PYTHONPATH=backend python scripts/train_transformer.py" >&2
  exit 1
fi

tar -czf "$OUTPUT" -C "$(dirname "$MODEL_DIR")" "$(basename "$MODEL_DIR")"
echo "Created $OUTPUT ($(du -h "$OUTPUT" | cut -f1))"
echo "Archive contains: $(basename "$MODEL_DIR")/best.pt"
echo "Set TRANSFORMER_MODEL_PATH=${MODEL_DIR} on Render."
echo ""
echo "Upload and set on Render:"
echo "  gh release create serving-v1 $OUTPUT --title 'Serving artifacts'"
echo "  SERVING_ARTIFACT_URL=<release asset URL>"
