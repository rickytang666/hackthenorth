#!/usr/bin/env bash
# Vendor the repo modules this Truss needs. Re-run after changing contract/,
# train/cohere/model.py, or the adapter. A Truss must be self-contained: it is
# built from this directory alone, with no access to the rest of the repo.
set -euo pipefail
cd "$(dirname "$0")"
ROOT=../..
rm -rf packages/contract packages/train packages/adapter packages/best
cp -r "$ROOT/contract" packages/contract
mkdir -p packages/train/cohere
cp "$ROOT/train/__init__.py" packages/train/
cp "$ROOT/train/cohere/__init__.py" "$ROOT/train/cohere/model.py" packages/train/cohere/
# Directory name is the served model_id suffix, so keep it the checkpoint name.
cp -r "$ROOT/checkpoints/cohere/best" packages/best
find packages -name __pycache__ -type d -exec rm -rf {} + 2>/dev/null || true
echo "vendored $(du -sh packages | cut -f1) into packages/"
