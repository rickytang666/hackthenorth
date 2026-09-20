# Demo assets

Everything beat 2 and beat 3 need, on disk, so the demo survives dead wifi.

The demo's own clips now live in `app/clips/`, chosen by `demo/curate_clips.py`
and listed in `app/clips/manifest.json`: seven M02 sentences where the tuned
model matches the reference exactly and the frozen baseline does not. Three sit
below the 0.879 accept threshold, so the clarification card fires on those and
nowhere else. The WAVs are gitignored; re-run the script to rebuild them.

`clips/` here is the older four-clip fallback bundle with `cue_card.json`.

M02 is the sealed test speaker: decoded exactly once, on 2026-09-19 at 18:33,
job `wd66me3`. Do not re-decode it.

Sealed results over all 388 test clips:

| Metric | Frozen | Tuned |
|---|---:|---:|
| Word error rate | 0.5668 | 0.3481 |
| Isolated words | 1.0169 | 0.4628 |
| Restricted sentences | 0.3634 | 0.2962 |
| Critical terms lost | 14 / 35 | 10 / 35 |
| p95 latency | 154 ms | 216 ms |

Never describe F04 as held out or unseen. It is the validation speaker that
selected this checkpoint. The script is `.workspace/hackathon-demo-script.md`.
