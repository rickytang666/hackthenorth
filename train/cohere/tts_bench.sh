#!/usr/bin/env bash
# Item #2: time OpenVoice on the serving GPU, not a laptop CPU.
# This decides whether chunked streaming TTS gets built or deleted from the plan,
# so it must run on the hardware the renderer will actually be served on.
set -euo pipefail

# The training image has no git, and both MeloTTS (a git dependency) and
# fetch_openvoice.sh need it.
apt-get update -qq && apt-get install -y -qq git
pip install -q uv
uv sync --frozen

# Reference clips come from TORGO M02, the sealed test speaker, because that is
# whose voice beat 5 speaks in.
mkdir -p "$VOICEBRIDGE_DATA"
uv run hf download abnerh/TORGO-database --repo-type dataset --local-dir "$VOICEBRIDGE_DATA/torgo-dataset"
uv run python -m data.prepare_torgo --source "$VOICEBRIDGE_DATA/torgo-dataset"

cd serve/voice
./fetch_openvoice.sh
uv sync
uv pip install -q "setuptools<81"          # librosa 0.9.1 imports pkg_resources
uv pip uninstall -q unidic 2>/dev/null || true   # force MeCab onto unidic-lite

REFS=$(ls "$VOICEBRIDGE_DATA"/torgo_wav/M02_1_headMic_000*.wav | head -5 | tr '\n' ' ')
echo "references: $REFS"
uv run --no-sync python time_synthesis.py --reference $REFS --runs 5
