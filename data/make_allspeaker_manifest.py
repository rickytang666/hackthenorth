"""One manifest with every dysarthric speaker, headMic only, for a difficulty ranking.

The frozen baseline never saw any TORGO speaker, so its per-speaker WER is a
fair severity ordering. This is NOT an evaluation of the tuned model: six of
these eight speakers are in its training split.
"""

import json
from pathlib import Path

from contract import manifest

rows = []
for name in ("torgo_dys_train.jsonl", "torgo_dys_dev.jsonl", "torgo_dys_test.jsonl"):
    rows += [r for r in manifest.load(name, resolve=False)
             if r["mic"].lower() == "headmic"]
rows.sort(key=lambda r: r["utterance_id"])

out = manifest.manifest_dir() / "torgo_dys_allspeakers.jsonl"
with out.open("w") as f:
    for r in rows:
        f.write(json.dumps(r, sort_keys=True) + "\n")

by = {}
for r in rows:
    by.setdefault(r["speaker_id"], []).append(r)
print(f"wrote {out} with {len(rows)} clips")
for s in sorted(by):
    hours = sum(x["duration"] for x in by[s]) / 3600
    print(f"  {s}  {len(by[s]):5} clips  {hours:.2f} h")
