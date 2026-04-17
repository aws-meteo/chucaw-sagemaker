#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODEL_FILE="${ROOT_DIR}/model.joblib"
INFERENCE_FILE="${ROOT_DIR}/inference/inference.py"
REQ_FILE="${ROOT_DIR}/inference/requirements.txt"
TARBALL_FILE="${ROOT_DIR}/model.tar.gz"

for required in "$MODEL_FILE" "$INFERENCE_FILE" "$REQ_FILE"; do
  if [[ ! -f "$required" ]]; then
    echo "ERROR: required file missing: $required" >&2
    exit 1
  fi
done

TMP_DIR="$(mktemp -d)"
cleanup() {
  rm -rf "$TMP_DIR"
}
trap cleanup EXIT

mkdir -p "$TMP_DIR/code"
cp "$MODEL_FILE" "$TMP_DIR/model.joblib"
cp "$INFERENCE_FILE" "$TMP_DIR/code/inference.py"
cp "$REQ_FILE" "$TMP_DIR/code/requirements.txt"

tar -czf "$TARBALL_FILE" -C "$TMP_DIR" model.joblib code

if [[ ! -f "$TARBALL_FILE" ]]; then
  echo "ERROR: packaging failed, tarball missing: $TARBALL_FILE" >&2
  exit 1
fi

EXPECTED_LIST=$'code/\ncode/inference.py\ncode/requirements.txt\nmodel.joblib'
ACTUAL_LIST="$(tar -tzf "$TARBALL_FILE" | LC_ALL=C sort)"

if [[ "$ACTUAL_LIST" != "$EXPECTED_LIST" ]]; then
  echo "ERROR: model.tar.gz structure mismatch." >&2
  echo "Expected:" >&2
  echo "$EXPECTED_LIST" >&2
  echo "Actual:" >&2
  echo "$ACTUAL_LIST" >&2
  exit 1
fi

echo "Packaged tarball: $TARBALL_FILE"
echo "Validated structure:"
echo "$ACTUAL_LIST"
