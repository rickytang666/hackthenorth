# Demo assets

Everything beat 2 and beat 3 need, on disk, so the demo survives dead wifi.

`clips/` holds four M02 recordings in presentation order with `cue_card.json`
giving what was said, what the frozen baseline heard, and what the tuned model
produces. M02 is the sealed test speaker: decoded exactly once, on 2026-09-19 at
18:33, job `wd66me3`. Do not re-decode it.

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
