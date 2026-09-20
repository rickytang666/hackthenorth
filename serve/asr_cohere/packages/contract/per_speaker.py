"""Per-speaker WER table, base against tuned.

Six of the eight dysarthric speakers are in the tuned model's training split, so
their rows are optimistic by construction and are labelled TRAINED. Only F04
(validation) and M02 (sealed test) are honest generalization numbers. The table
is still worth having: it shows how the gain tracks speaker severity.
"""

import json
import sys
from pathlib import Path

from contract.evaluate import score

TRAINED = {"F01", "F03", "M01", "M03", "M04", "M05"}
ROLE = {"F04": "validation", "M02": "sealed test"}


def by_speaker(path: str) -> dict:
    rows = [json.loads(line) for line in Path(path).open()]
    out: dict[str, list] = {}
    for r in rows:
        out.setdefault(r["speaker_id"], []).append(r)
    return out


def main() -> None:
    base, tuned = by_speaker(sys.argv[1]), by_speaker(sys.argv[2])
    print("\n| Speaker | Role | Clips | Base WER | Tuned WER | Relative | Critical base | Critical tuned |")
    print("|---|---|---:|---:|---:|---:|---:|---:|")
    unseen = []
    for s in sorted(base):
        b, t = score(base[s]), score(tuned[s])
        rel = 100 * (b["wer"] - t["wer"]) / b["wer"] if b["wer"] else 0.0
        role = ROLE.get(s, "TRAINED")
        if s not in TRAINED:
            unseen.append((s, b["wer"], t["wer"]))
        print(f"| {s} | {role} | {len(base[s])} | {b['wer']:.4f} | {t['wer']:.4f} | "
              f"{rel:.1f}% | {b['critical_error_rate']:.4f} | {t['critical_error_rate']:.4f} |")

    order = sorted((score(base[s])["wer"], s) for s in base)
    print("\nDifficulty ranking by FROZEN baseline WER (fair: it saw no one):")
    print("  easiest " + " < ".join(f"{s} {w:.3f}" for w, s in order) + "  hardest")
    if unseen:
        print("\nHonest generalization rows only (never trained on):")
        for s, b, t in unseen:
            print(f"  {s}: {b:.4f} -> {t:.4f}  ({100*(b-t)/b:.1f}% relative)")


if __name__ == "__main__":
    main()
