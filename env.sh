# Source this before anything: `source env.sh`
# VOICEBRIDGE_DATA is the root every manifest path resolves against. It differs
# per machine, which is exactly why manifests store relative paths.
export VOICEBRIDGE_DATA="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)/data/voicebridge"
export PYTHONPATH="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd):${PYTHONPATH}"
