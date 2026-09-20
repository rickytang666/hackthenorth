"""Pick the demo clips and copy them next to the app.

    uv run python demo/curate_clips.py

Selection rule, applied to the sealed M02 decode only: the tuned model matches
the reference exactly, the frozen baseline does not, and the reference is a
sentence rather than an isolated word. Ranked by how badly the baseline failed.

M02 is the only speaker used. It is the sealed test speaker, so nothing here was
trained on and nothing here selected the checkpoint. F04 is excluded on purpose:
it chose the checkpoint, so calling it unseen would be wrong, and only one F04
clip qualifies anyway.

The WAVs are gitignored like every other WAV in this repo; re-run this to
rebuild them from $VOICEBRIDGE_DATA. app/clips/manifest.json is committed so the
chosen set is a reviewable fact rather than whatever happens to be on a laptop.
"""

import json
import os
import shutil
from pathlib import Path

from contract.confidence import ACCEPT_THRESHOLD
from contract.normalize import normalize

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "app" / "clips"
BASELINE = ROOT / "results" / "test_m02" / "baseline_cohere.jsonl"
TUNED = ROOT / "results" / "test_m02" / "tuned_cohere.jsonl"

# Hand-picked from the ranked candidates, in presentation order. Ranking alone
# would lead with a 13-word clip and put the negation flip fifth.
ORDER = [
    "M02_1_headMic_0141",   # total collapse into nonsense, the opener
    "M02_1_headMic_0135",   # below threshold, so the clarification card fires
    "M02_1_headMic_0185",   # baseline inverts the meaning with a spurious "not"
    "M02_1_headMic_0184",
    "M02_1_headMic_0196",
    "M02_2_headMic_0088",   # below threshold, the second clarification
    "M02_1_headMic_0092",   # longest, 13 words, closer
]
MIN_WORDS = 4


def word_errors(ref: str, hyp: str) -> int:
    r, h = ref.split(), hyp.split()
    prev = list(range(len(h) + 1))
    for i, rw in enumerate(r, 1):
        cur = [i]
        for j, hw in enumerate(h, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (rw != hw)))
        prev = cur
    return prev[-1]


def resolve(recorded: str) -> Path:
    """Re-root a path recorded on the training machine onto this one.

    The decode ran on Baseten with $VOICEBRIDGE_DATA at /tmp, so the absolute
    paths in the JSONL do not exist here. Only the tail below the data root is
    portable.
    """
    root = Path(os.environ.get("VOICEBRIDGE_DATA", Path.home() / "voicebridge-data"))
    parts = Path(recorded).parts
    tail = Path(*parts[parts.index("torgo_wav"):]) if "torgo_wav" in parts else Path(parts[-1])
    for candidate in (Path(recorded), root / tail, root / "torgo_wav" / Path(recorded).name):
        if candidate.exists():
            return candidate
    raise SystemExit(f"missing {recorded}; source env.sh so $VOICEBRIDGE_DATA resolves")


def rows(path: Path) -> dict:
    return {r["utterance_id"]: r for r in map(json.loads, path.open())}


def main() -> None:
    base, tuned = rows(BASELINE), rows(TUNED)
    picked = []
    for uid in ORDER:
        t, b = tuned[uid], base[uid]
        ref, hyp, before = (normalize(t["text"]), normalize(t["prediction"]),
                            normalize(b["prediction"]))
        assert hyp == ref, f"{uid}: tuned no longer matches the reference"
        assert before != ref, f"{uid}: baseline now matches too, clip is pointless"
        assert len(ref.split()) >= MIN_WORDS, f"{uid}: not a sentence"
        src = resolve(t["audio_filepath"])
        shutil.copyfile(src, OUT / f"{uid}.wav")
        errs = word_errors(ref, before)
        picked.append({
            "utterance_id": uid,
            "wav": f"clips/{uid}.wav",
            "speaker_id": t["speaker_id"],
            "truth": ref,
            "baseline": before,
            "tuned": hyp,
            "baseline_word_errors": errs,
            "baseline_wer": round(errs / len(ref.split()), 4),
            "expected_confidence": round(t["confidence"], 4),
            "clarifies": t["confidence"] < ACCEPT_THRESHOLD,
            "duration_s": round(t["duration"], 2),
        })
    (OUT / "manifest.json").write_text(json.dumps(picked, indent=2) + "\n")
    fires = sum(p["clarifies"] for p in picked)
    print(f"{len(picked)} clips -> {OUT}/manifest.json  ({fires} below "
          f"the {ACCEPT_THRESHOLD} threshold, so the clarification card fires there)")


if __name__ == "__main__":
    main()
