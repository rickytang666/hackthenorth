# Source this before anything: `source env.sh`
#
# VOICEBRIDGE_DATA lives OUTSIDE the repo. Two reasons: the TORGO dataset and
# the generated WAVs are ~3 GB and would be uploaded with every Baseten training
# push, and manifests store paths relative to this root so it differs per
# machine anyway. Override it freely; nothing hardcodes the location.
export VOICEBRIDGE_DATA="${VOICEBRIDGE_DATA:-$HOME/voicebridge-data}"
export PYTHONPATH="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd):${PYTHONPATH}"
