#!/usr/bin/env bash
# Fresh clone to a temp dir, install, and verify the contract holds.
# This is what fails at 03:00 if nobody wrote it: a repo that works only
# because of files living outside git.
set -euo pipefail

BRANCH="${1:-main}"
TMP=$(mktemp -d)
trap 'rm -rf "$TMP"' EXIT
echo "clean clone of $BRANCH into $TMP"

git clone -q --depth 1 --branch "$BRANCH" \
  "$(git -C "$(dirname "$0")/.." remote get-url origin)" "$TMP/repo"
cd "$TMP/repo"

echo "1. uv sync"
uv sync -q
echo "   ok"

echo "2. contract imports"
uv run python -c "
import contract.normalize, contract.predictions, contract.evaluate
import contract.confidence, contract.decode, contract.manifest, contract.mock_asr
print('   ok')
"

echo "3. normalizer worked examples"
uv run python -c "
from contract.normalize import normalize
assert normalize(\"Don't give her the insulin.\") == 'dont give her the insulin'
assert normalize('  THE  QUICK   brown fox  ') == 'the quick brown fox'
print('   ok')
"

echo "4. evaluator on synthetic predictions"
uv run python -c "
from contract.evaluate import score
rows=[{'audio_filepath':'a','speaker_id':'X','text':'do not call the nurse',
       'prediction':'do call the nurse','model_id':'m','latency_ms':1.0,'duration':1.0}]
s=score(rows); assert s['wer']>0 and s['critical_error_rate']>0
print(f\"   ok, WER {s['wer']:.3f}, critical {s['critical_error_rate']:.3f}\")
"

echo "5. verifier refuses text outside the candidate set"
uv run python -c "
from app.verifier.verify import enforce, Verdict
v=enforce(Verdict('accept', text='invented'), ['do not call','do call'])
assert v.action=='clarify'
print('   ok')
"

echo "6. mock ASR round trip over Protocol 1"
uv run python -m contract.mock_asr --port 8799 > /tmp/cc_mock.log 2>&1 &
MOCK=$!
trap 'kill $MOCK 2>/dev/null; rm -rf "$TMP"' EXIT
for _ in $(seq 1 30); do sleep 1; grep -q listening /tmp/cc_mock.log && break; done
uv run python -m bench.latency --url ws://127.0.0.1:8799 --runs 2 --seconds 1.0 --out /dev/null \
  | grep -q first_partial_p50_ms && echo "   ok"

echo "7. manifests are NOT in the clone (data must live outside git)"
test ! -d data/voicebridge && test ! -d data/torgo-dataset && echo "   ok"

echo
echo "CLEAN CLONE PASSES. A fresh machine needs only: uv sync, source env.sh,"
echo "then the TORGO download and prepare_torgo to regenerate the manifests."
