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

echo "Downloading serving artifacts ..."
if [ -n "${GITHUB_TOKEN:-}" ]; then
  if ! curl -fsSL \
    -H "Authorization: Bearer ${GITHUB_TOKEN}" \
    -H "Accept: application/octet-stream" \
    "$SERVING_ARTIFACT_URL" -o /tmp/serving-artifacts.tar.gz; then
    DOWNLOAD_FAILED=1
  fi
else
  if ! curl -fsSL "$SERVING_ARTIFACT_URL" -o /tmp/serving-artifacts.tar.gz; then
    DOWNLOAD_FAILED=1
  fi
fi

if [ "${DOWNLOAD_FAILED:-0}" -eq 1 ]; then
  echo "Failed to download serving artifacts (HTTP error)." >&2
  echo "If the GitHub repo is private, set GITHUB_TOKEN on Render with read access to releases." >&2
  echo "Or host serving-artifacts.tar.gz at a public HTTPS URL and point SERVING_ARTIFACT_URL there." >&2
  exit 1
fi
mkdir -p "$(dirname "$MODEL_DIR")"
tar -xzf /tmp/serving-artifacts.tar.gz -C /app
rm -f /tmp/serving-artifacts.tar.gz

if [ ! -f "$CHECKPOINT" ]; then
  echo "Download finished but checkpoint missing at $CHECKPOINT" >&2
  exit 1
fi

echo "Serving artifacts ready at $MODEL_DIR"
