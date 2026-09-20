"""Scoring. The single ruler both lanes are measured against.

Four numbers, in plain terms:
  1. how many words it got wrong          -> WER, split into words vs sentences
  2. how many words that MATTER it got wrong -> critical error rate
  3. whether it got worse at ordinary speech -> run this on clean_eval too
  4. how fast                                -> RTF and latency percentiles
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import jiwer

from contract.normalize import normalize, tokens

CRITICAL_TERMS_PATH = Path(__file__).parent / "critical_terms.txt"


def load_critical_terms(path: str | Path = CRITICAL_TERMS_PATH) -> set[str]:
    terms = set()
    for line in Path(path).read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            terms.add(normalize(line))
    return terms - {""}


def wer(refs: list[str], hyps: list[str]) -> float:
    """Word error rate over normalized text. Empty reference set scores 0.0."""
    pairs = [(normalize(r), normalize(h)) for r, h in zip(refs, hyps)]
    pairs = [(r, h) for r, h in pairs if r]
    if not pairs:
        return 0.0
    return jiwer.wer([r for r, _ in pairs], [h for _, h in pairs])


def cer(refs: list[str], hyps: list[str]) -> float:
    pairs = [(normalize(r), normalize(h)) for r, h in zip(refs, hyps)]
    pairs = [(r, h) for r, h in pairs if r]
    if not pairs:
        return 0.0
    return jiwer.cer([r for r, _ in pairs], [h for _, h in pairs])


def critical_error_rate(rows: list[dict], terms: set[str] | None = None) -> tuple[float, int, int]:
    """Fraction of critical-term occurrences in the reference that the hypothesis lost.

    A term counts as preserved when it appears at least as often in the
    hypothesis as in the reference. Returns (rate, lost, total).
    """
    terms = load_critical_terms() if terms is None else terms
    lost = total = 0
    for row in rows:
        ref, hyp = tokens(row["text"]), tokens(row["prediction"])
        for term in terms:
            n_ref = ref.count(term)
            if not n_ref:
                continue
            total += n_ref
            lost += max(0, n_ref - hyp.count(term))
    return (lost / total if total else 0.0), lost, total


def is_isolated_word(row: dict) -> bool:
    return len(tokens(row["text"])) == 1


def percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    idx = min(len(ordered) - 1, int(round((p / 100.0) * (len(ordered) - 1))))
    return ordered[idx]


def score(rows: list[dict]) -> dict:
    refs = [r["text"] for r in rows]
    hyps = [r["prediction"] for r in rows]
    words = [r for r in rows if is_isolated_word(r)]
    sentences = [r for r in rows if not is_isolated_word(r)]

    by_speaker = defaultdict(lambda: {"ref": [], "hyp": []})
    for row in rows:
        bucket = by_speaker[row["speaker_id"]]
        bucket["ref"].append(row["text"])
        bucket["hyp"].append(row["prediction"])

    rate, lost, total = critical_error_rate(rows)
    latencies = [float(r["latency_ms"]) for r in rows if r.get("latency_ms") is not None]
    durations = [float(r["duration"]) for r in rows if r.get("duration")]

    return {
        "n": len(rows),
        "model_id": rows[0].get("model_id") if rows else None,
        "wer": wer(refs, hyps),
        "cer": cer(refs, hyps),
        "wer_isolated_word": wer([r["text"] for r in words], [r["prediction"] for r in words]),
        "wer_sentence": wer([r["text"] for r in sentences], [r["prediction"] for r in sentences]),
        "critical_error_rate": rate,
        "critical_lost": lost,
        "critical_total": total,
        "per_speaker_wer": {s: wer(v["ref"], v["hyp"]) for s, v in sorted(by_speaker.items())},
        "latency_p50_ms": percentile(latencies, 50),
        "latency_p95_ms": percentile(latencies, 95),
        "rtf": (sum(latencies) / 1000.0 / sum(durations)) if durations and latencies else None,
    }


_ROWS = [
    ("Held-out speaker WER", "wer", "{:.4f}"),
    ("Isolated-word WER", "wer_isolated_word", "{:.4f}"),
    ("Restricted-sentence WER", "wer_sentence", "{:.4f}"),
    ("Critical error rate", "critical_error_rate", "{:.4f}"),
    ("p50 latency (ms)", "latency_p50_ms", "{:.1f}"),
    ("p95 latency (ms)", "latency_p95_ms", "{:.1f}"),
    ("Real-time factor", "rtf", "{:.4f}"),
]


def scorecard(named: dict[str, dict]) -> str:
    names = list(named)
    width = max([22] + [len(n) for n in names])
    head = "| Metric".ljust(28) + "".join(f"| {n:>{width}} " for n in names) + "|"
    sep = "|" + "-" * 27 + "".join("|" + "-" * (width + 2) for _ in names) + "|"
    lines = [head, sep]
    for label, key, fmt in _ROWS:
        cells = ""
        for n in names:
            value = named[n].get(key)
            cells += f"| {(fmt.format(value) if value is not None else 'n/a'):>{width}} "
        lines.append(f"| {label:<26}" + cells + "|")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Score prediction JSONL files.")
    parser.add_argument("files", nargs="+", help="prediction JSONL paths")
    parser.add_argument("--json", action="store_true", help="emit raw JSON too")
    args = parser.parse_args()

    named = {}
    for path in args.files:
        rows = [json.loads(line) for line in Path(path).open()]
        named[Path(path).stem] = score(rows)
    print(scorecard(named))
    if args.json:
        print(json.dumps(named, indent=2))


if __name__ == "__main__":
    main()
