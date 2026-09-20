#!/usr/bin/env bash
set -euo pipefail

repo_root=$(git rev-parse --show-toplevel)
branch_name=$(git -C "$repo_root" symbolic-ref --short HEAD)
expected_commit=$(git -C "$repo_root" rev-parse HEAD)
task_tmp=$(mktemp -d "${TMPDIR:-/tmp}/voicebridge-clean-clone.XXXXXX")
clone_dir="$task_tmp/repo"
mock_pid=""

cleanup() {
  if [[ -n "$mock_pid" ]]; then
    kill "$mock_pid" 2>/dev/null || true
    wait "$mock_pid" 2>/dev/null || true
  fi
  rm -rf "$task_tmp"
}
trap cleanup EXIT

echo "[1/5] cloning $branch_name at ${expected_commit:0:12}"
git clone --quiet --no-local --branch "$branch_name" "$repo_root" "$clone_dir"
actual_commit=$(git -C "$clone_dir" rev-parse HEAD)
[[ "$actual_commit" == "$expected_commit" ]] || {
  echo "clean clone resolved to $actual_commit, expected $expected_commit" >&2
  exit 1
}

echo "[2/5] syncing locked dependencies"
uv sync --frozen --project "$clone_dir" --quiet

echo "[3/5] importing contract modules"
(
  cd "$clone_dir"
  uv run --frozen python - <<'PY'
import importlib

modules = (
    "contract.confidence",
    "contract.decode",
    "contract.evaluate",
    "contract.manifest",
    "contract.mock_asr",
    "contract.normalize",
    "contract.predictions",
)
for module in modules:
    importlib.import_module(module)
print(f"imported {len(modules)} contract modules")
PY
)

echo "[4/5] verifying manifest hash contracts"
reference_hashes="$clone_dir/contract/MANIFEST_HASHES"
[[ -s "$reference_hashes" ]] || {
  echo "contract/MANIFEST_HASHES is missing or empty" >&2
  exit 1
}
while IFS= read -r hash_file; do
  cmp -s "$reference_hashes" "$hash_file" || {
    echo "MANIFEST_HASHES copy differs: ${hash_file#"$clone_dir"/}" >&2
    exit 1
  }
done < <(find "$clone_dir" -path '*/MANIFEST_HASHES' -type f -not -path '*/.venv/*' | sort)

uv run --project "$clone_dir" --frozen python - "$reference_hashes" <<'PY'
import re
import sys
from pathlib import Path

path = Path(sys.argv[1])
rows = path.read_text().splitlines()
if not rows or any(not re.fullmatch(r"[0-9a-f]{64}  [^ ]+\.jsonl", row) for row in rows):
    raise SystemExit("MANIFEST_HASHES contains an invalid row")
if len({row.split("  ", 1)[1] for row in rows}) != len(rows):
    raise SystemExit("MANIFEST_HASHES contains duplicate manifest names")
print(f"verified {len(rows)} manifest hash declarations")
PY

if [[ -n "${VOICEBRIDGE_DATA:-}" ]]; then
  while read -r expected_hash manifest_name; do
    manifest_path="$VOICEBRIDGE_DATA/manifests/$manifest_name"
    [[ -f "$manifest_path" ]] || {
      echo "missing manifest under VOICEBRIDGE_DATA: $manifest_name" >&2
      exit 1
    }
    actual_hash=$(shasum -a 256 "$manifest_path" | awk '{print $1}')
    [[ "$actual_hash" == "$expected_hash" ]] || {
      echo "manifest hash mismatch: $manifest_name" >&2
      exit 1
    }
  done < "$reference_hashes"
  echo "verified manifest bytes under VOICEBRIDGE_DATA"
else
  echo "VOICEBRIDGE_DATA unset; verified tracked declarations and package copies only"
fi

echo "[5/5] running a Protocol 1 mock round trip"
mock_port=$(uv run --project "$clone_dir" --frozen python - <<'PY'
import socket

with socket.socket() as sock:
    sock.bind(("127.0.0.1", 0))
    print(sock.getsockname()[1])
PY
)
(
  cd "$clone_dir"
  uv run --frozen python -m contract.mock_asr --port "$mock_port"
) >"$task_tmp/mock.log" 2>&1 &
mock_pid=$!

uv run --project "$clone_dir" --frozen python - "$mock_port" <<'PY'
import socket
import sys
import time

port = int(sys.argv[1])
for _ in range(100):
    try:
        with socket.create_connection(("127.0.0.1", port), timeout=0.1):
            break
    except OSError:
        time.sleep(0.05)
else:
    raise SystemExit("mock ASR did not start")
PY

(
  cd "$clone_dir"
  uv run --frozen python -m bench.latency \
    --url "ws://127.0.0.1:$mock_port/v1/stream" \
    --runs 1 --seconds 0.1 --out "$task_tmp/latency.jsonl" >"$task_tmp/latency.stdout"
)
uv run --project "$clone_dir" --frozen python - "$task_tmp/latency.jsonl" <<'PY'
import json
import sys
from pathlib import Path

row = json.loads(Path(sys.argv[1]).read_text().splitlines()[-1])
if row.get("model_id") != "mock-asr":
    raise SystemExit("mock round trip returned the wrong model_id")
if row.get("first_partial_p50_ms") is None or row.get("final_p50_ms") is None:
    raise SystemExit("mock round trip did not return complete timing")
print(f"mock round trip passed in {row['final_p50_ms']:.1f} ms")
PY

kill "$mock_pid" 2>/dev/null || true
wait "$mock_pid" 2>/dev/null || true
mock_pid=""

[[ -z "$(git -C "$clone_dir" status --porcelain)" ]] || {
  echo "clean clone gained tracked or unignored files during verification" >&2
  git -C "$clone_dir" status --short >&2
  exit 1
}
echo "clean-clone check passed"
