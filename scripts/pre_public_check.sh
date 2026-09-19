#!/usr/bin/env bash

set -euo pipefail

repo_root="$(git rev-parse --show-toplevel 2>/dev/null)" || {
  echo "pre-public check: run this inside the VoiceBridge Git repository" >&2
  exit 2
}
cd "$repo_root"

failed=0
scanner_path='scripts/pre_public_check.sh'
key_pattern="(sk-ant-[A-Za-z0-9_-]{20,}|sk-[A-Za-z0-9_-]{20,}|hf_[A-Za-z0-9]{20,}|xox[baprs]-[A-Za-z0-9-]{20,}|gh[pousr]_[A-Za-z0-9]{20,}|AKIA[0-9A-Z]{16}|AIza[0-9A-Za-z_-]{30,}|-----BEGIN (RSA |EC |OPENSSH )?PRIVATE KEY-----|((api[_-]?key|access[_-]?token|secret|password|BASETEN_API_KEY|HF_TOKEN)[[:space:]]*[:=][[:space:]]*[\"']?[A-Za-z0-9][A-Za-z0-9_./+=-]{15,}))"

report_matches() {
  local heading="$1"
  local matches="$2"
  if [[ -n "$matches" ]]; then
    echo "$heading" >&2
    printf '%s\n' "$matches" | sed 's/^/  /' >&2
    failed=1
  fi
}

current_matches="$(
  git grep -I -l -i -E "$key_pattern" -- . ":(exclude)$scanner_path" 2>/dev/null || true
)"
report_matches "Potential secret shapes in tracked files:" "$current_matches"

history_matches="$(
  while IFS= read -r revision; do
    git grep -I -l -i -E "$key_pattern" "$revision" -- . ":(exclude)$scanner_path" 2>/dev/null || true
  done < <(git rev-list --all)
)"
history_matches="$(printf '%s\n' "$history_matches" | sed '/^$/d' | sort -u)"
report_matches "Potential secret shapes in Git history:" "$history_matches"

ignore_probes=(
  'CLAUDE.md'
  'AGENTS.md'
  '.workspace/pre-public-probe'
  '.env'
  '.env.local'
)

for probe in "${ignore_probes[@]}"; do
  if ! git check-ignore -q "$probe"; then
    echo "Missing required .gitignore coverage: $probe" >&2
    failed=1
  fi
done

if [[ "$failed" -ne 0 ]]; then
  echo "pre-public check: FAILED" >&2
  exit 1
fi

echo "pre-public check: PASS"
echo "  tracked files: no key-shaped values found"
echo "  Git history: no key-shaped values found"
echo "  ignore rules: CLAUDE.md, AGENTS.md, .workspace/, and .env* covered"
