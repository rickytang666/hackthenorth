"""Extend the frozen safety slice from the dev and test transcripts.

The seed list in critical_terms.txt is generic. This adds the terms that
actually occur in the speakers we score against, so the critical-error rate
measures something real rather than mostly counting zero.

Only terms PRESENT in dev or test are added: a term that never appears cannot
be got wrong, and padding the list would quietly dilute the metric.

    uv run python -m contract.extend_critical_terms --write
"""

import argparse
from pathlib import Path

from contract import manifest
from contract.evaluate import CRITICAL_TERMS_PATH, load_critical_terms
from contract.normalize import tokens

# Categories whose loss changes meaning rather than fluency.
NEGATION = {"no", "not", "dont", "don't", "cant", "cannot", "wont", "never",
            "none", "nothing", "nobody", "nor", "neither", "without", "stop"}
AFFIRM = {"yes", "yeah", "yep", "ok", "okay", "sure", "please"}
NUMBERS = {"zero", "one", "two", "three", "four", "five", "six", "seven", "eight",
           "nine", "ten", "eleven", "twelve", "thirteen", "fourteen", "fifteen",
           "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty",
           "ninety", "hundred", "thousand", "first", "second", "third", "half",
           "quarter", "twice", "double"}
MEDICAL = {"mg", "ml", "dose", "pill", "pills", "tablet", "insulin", "morphine",
           "warfarin", "aspirin", "inhaler", "oxygen", "nurse", "doctor", "pain",
           "help", "hurt", "sick", "medicine", "medication", "emergency"}
TIME = {"now", "before", "after", "today", "tomorrow", "tonight", "morning",
        "evening", "never", "always", "again"}

CANDIDATES = NEGATION | AFFIRM | NUMBERS | MEDICAL | TIME


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--write", action="store_true", help="update critical_terms.txt")
    args = parser.parse_args()

    present = set()
    counts: dict[str, int] = {}
    for name in ("torgo_dys_dev.jsonl", "torgo_dys_test.jsonl"):
        for row in manifest.load(name, resolve=False):
            for token in tokens(row["text"]):
                if token in CANDIDATES:
                    present.add(token)
                    counts[token] = counts.get(token, 0) + 1

    existing = load_critical_terms()
    added = sorted(present - existing)
    unused = sorted(existing - present)

    print(f"terms present in dev+test: {len(present)}")
    for term in sorted(present, key=lambda t: -counts[t]):
        mark = "new" if term in added else "   "
        print(f"  {mark} {counts[term]:4}x  {term}")
    print(f"\nin the list but absent from dev/test ({len(unused)}): {' '.join(unused)}")
    print("  (harmless: they can never be scored, they just never fire)")

    if not args.write:
        print("\ndry run. pass --write to update critical_terms.txt")
        return
    if not added:
        print("\nnothing to add")
        return

    path = Path(CRITICAL_TERMS_PATH)
    lines = path.read_text().rstrip().splitlines()
    lines.append("")
    lines.append("# Added from dev and test transcripts by extend_critical_terms.py.")
    lines.extend(added)
    path.write_text("\n".join(lines) + "\n")
    print(f"\nadded {len(added)}: {' '.join(added)}")


if __name__ == "__main__":
    main()
