#!/usr/bin/env bash
# OpenVoice's pyproject pins gradio 3.48 and faster-whisper 0.9, which drags in
# av==10 and fails to build. Its tone-color conversion needs only the source
# tree plus the deps already in pyproject.toml, so we vendor the source.
set -euo pipefail
cd "$(dirname "$0")"
[ -d vendor/OpenVoice ] || git clone --depth 1 https://github.com/myshell-ai/OpenVoice.git vendor/OpenVoice
mkdir -p checkpoints
uv run hf download myshell-ai/OpenVoiceV2 --local-dir checkpoints/openvoice_v2
echo "vendored: $(pwd)/vendor/OpenVoice"
