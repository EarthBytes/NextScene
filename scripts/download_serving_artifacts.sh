#!/bin/sh
# Download transformer checkpoint tarball before boot (Render builds from git without /models).
set -e

MODEL_DIR="${TRANSFORMER_MODEL_PATH:-models/transformer-full-v2}"
CHECKPOINT="${MODEL_DIR}/best.pt"
GITHUB_TOKEN="${GITHUB_TOKEN:-${GH_TOKEN:-}}"

if [ -f "$CHECKPOINT" ]; then
  echo "Serving checkpoint already present: $CHECKPOINT"
  exit 0
fi

resolve_github_release_asset_url() {
  GITHUB_REPO="${GITHUB_REPO:-EarthBytes/Generative-Recommendation-System}"
  GITHUB_RELEASE_TAG="${GITHUB_RELEASE_TAG:?GITHUB_RELEASE_TAG is required}"
  GITHUB_ASSET_NAME="${GITHUB_ASSET_NAME:-serving-artifacts.tar.gz}"

  if [ -z "$GITHUB_TOKEN" ]; then
    echo "GITHUB_TOKEN is required to download from a private GitHub release." >&2
    exit 1
  fi

  python3 - "$GITHUB_REPO" "$GITHUB_RELEASE_TAG" "$GITHUB_ASSET_NAME" <<'PY'
import json
import sys
import urllib.error
import urllib.request

repo, tag, asset_name = sys.argv[1:4]
token = __import__("os").environ.get("GITHUB_TOKEN") or __import__("os").environ.get("GH_TOKEN")
url = f"https://api.github.com/repos/{repo}/releases/tags/{tag}"
req = urllib.request.Request(
    url,
    headers={
        "Authorization": f"Bearer {token}",
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
    },
)
try:
    with urllib.request.urlopen(req) as resp:
        release = json.load(resp)
except urllib.error.HTTPError as exc:
    print(f"GitHub API error {exc.code} for {url}", file=sys.stderr)
    if exc.code == 404:
        print("Check GITHUB_REPO, GITHUB_RELEASE_TAG, and token access to this repo.", file=sys.stderr)
    sys.exit(1)

for asset in release.get("assets", []):
    if asset.get("name") == asset_name:
        print(asset["url"])
        sys.exit(0)

print(f"Asset {asset_name!r} not found on release {tag!r}", file=sys.stderr)
sys.exit(1)
PY
}

download_url=""
if [ -n "${GITHUB_RELEASE_TAG:-}" ]; then
  echo "Resolving GitHub release ${GITHUB_RELEASE_TAG} (authenticated) ..."
  download_url="$(resolve_github_release_asset_url)"
elif [ -n "${SERVING_ARTIFACT_URL:-}" ]; then
  download_url="$SERVING_ARTIFACT_URL"
  case "$download_url" in
    *github.com/*/releases/download/*)
      echo "SERVING_ARTIFACT_URL uses a browser GitHub URL, which returns 404 for private repos." >&2
      echo "Set GITHUB_RELEASE_TAG=serving-v1 instead, or use the GitHub API asset URL." >&2
      exit 1
      ;;
  esac
else
  echo "SERVING_ARTIFACT_URL not set; skipping model download."
  exit 0
fi

echo "Downloading serving artifacts ..."
if [ -n "$GITHUB_TOKEN" ]; then
  if ! curl -fsSL \
    -H "Authorization: Bearer ${GITHUB_TOKEN}" \
    -H "Accept: application/octet-stream" \
    "$download_url" -o /tmp/serving-artifacts.tar.gz; then
    DOWNLOAD_FAILED=1
  fi
else
  if ! curl -fsSL "$download_url" -o /tmp/serving-artifacts.tar.gz; then
    DOWNLOAD_FAILED=1
  fi
fi

if [ "${DOWNLOAD_FAILED:-0}" -eq 1 ]; then
  echo "Failed to download serving artifacts (HTTP error)." >&2
  echo "Private repo: set GITHUB_TOKEN plus GITHUB_RELEASE_TAG=serving-v1 on Render." >&2
  echo "Public hosting: set SERVING_ARTIFACT_URL to a public HTTPS URL." >&2
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
