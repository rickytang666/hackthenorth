# VoiceBridge ownership and lanes

Two builders. This file binds both of you regardless of whether you write code by hand or with an agent. The technical agreement is in [DESIGN.md](DESIGN.md).

| Role | Person | Fill in |
|---|---|---|
| Person A, Parakeet lane and serving optimization, **demo owner** | | |
| Person B, Cohere lane and product path | | |

Person A is the demo owner because A's post-gate work (FP8, serving micro-optimization) is the most droppable thing on the board, and B's post-gate work (the product path) is not. The demo owner flips to full-time submission work at roughly 22:00.

## Phase 0 is joint. Do not branch before it exits.

One person at the keyboard, the other provisioning. Parallelizing code before the seams exist is the most expensive mistake available here: two lanes independently invent two normalizers, two prediction schemas, and two ideas of what a partial hypothesis is, and then the hour-5 comparison is meaningless.

**Exit gate, all six, not a clock:**

1. `contract/evaluate.py` prints the full scorecard table from synthetic prediction files
2. `contract/mock_asr.py` streams protocol-correct partials and `bench/latency.py` reports a number from it
3. Both people independently reproduce the five manifest SHA-256 hashes
4. Both people have a trainable checkpoint loaded and one local backward pass completed
5. `contract/confidence.py` returns a score on a real decode, because NVIDIA's TDT confidence utility is broken and both lanes must threshold identically
6. One 7-word OpenVoice synthesis is timed and written down, which decides whether streaming TTS is built at all

The provisioning lane is real work and is fatal if deferred: Baseten account, CLI, and H100 quota; `HF_TOKEN` plus the Cohere contact-information click-through accepted (Apache 2.0, two minutes, not an approval queue, and pull from `CohereLabs/cohere-transcribe-03-2026` rather than a CC-BY-NC mirror); TORGO downloaded; OpenVoice V2 weights downloaded; one consented enrollment recording; verifier Model API key; TORGO license checked for third-party cloud processing.

## File ownership

Request a change to someone else's file through them. A two-minute ping, not a code review.

```text
hackthenorth/
|
|-- contract/ .................. BOTH, frozen after Phase 0
|   |                            changes are announced, never quiet
|   `-- (all files)              a quiet edit here voids every score
|
|-- data/prepare_torgo.py ...... PHASE 0, then frozen
|                                rerunning it invalidates both lanes
|-- serve/_template/ ........... PHASE 0, then frozen. copy it, do not edit it
|
|-- train/parakeet/ ............ A
|-- serve/asr_parakeet/ ........ A
|-- bench/ ..................... A
|-- DESIGN.md .................. A
|
|-- train/cohere/ .............. B
|-- serve/asr_cohere/ .......... B
|-- serve/voice/ ............... B
|-- app/ ....................... B
|-- .gitignore ................. B
|-- package.json / lock files .. B  (B installs ALL dependencies)
|-- pyproject.toml / lock ...... B
|
|-- OWNERSHIP.md ............... either, append only
|-- .workspace/hackathon-status.md  either, append only
`-- results/ ................... gitignored, no owner needed
```

Standing rules:

- Merge to main every 60 to 90 minutes. Never hold a branch longer.
- Pull main before starting each task, not before pushing.
- Never force-push a shared branch. `--force-with-lease`, own branch only.
- One commit per unit that works. The test is whether reverting it alone leaves a working app.
- After any merge whose demo-path smoke passes: `git tag -f last-good && git push -f origin last-good`.
- Branches are `type/short-description`, never prefixed with a tool or model name.
- A human presses merge, always. No CI auto-merge.

## The only four sync points after Phase 0

Everything else is lane-local. If you find yourself waiting on the other person outside these four, say so in `.workspace/hackathon-status.md` immediately, because it means a seam leaked.

0. **The 200-step kill-rule check (~17:00).** Thirty seconds, not a meeting: each person states whether their tuned model beat its own frozen baseline. A "no" means that lane stops now rather than at 19:45. See DESIGN.md for why an early flat curve is a wiring bug and not slow learning.
1. **The promotion gate (~19:45).** Both dev scorecards filled, winner chosen by the predeclared rule in DESIGN.md, `ASR_WS_URL` flipped. This is a config change, not an integration.
2. **The sealed test decode (~20:45).** One person runs it once, with the winner and its own frozen baseline. Never reopened.
3. **The freeze (04:00).** Demo-path bugfixes only after this.

## Clock

Freeze 04:00 Sunday. Final code and Devpost edits close 08:00. Judging 09:30 at the venue, so no travel budget.

| Wall clock | Person A | Person B | Exit condition |
|---|---|---|---|
| 14:30 to 15:45 | Keyboard: `contract/`, manifests, mock, template, bench | Provisioning list, then verify hashes and run a local backward pass | **The four Phase 0 gates. Branch only now.** |
| 15:45 to 18:45 | Launch Parakeet NeMo job on H100-A, checkpoint every 200 to 250 steps. While it runs: `serve/asr_parakeet/` against the frozen base, latency harness | Launch Cohere LoRA job on H100-B (top 6 encoder blocks plus decoder, ~400 steps). While it runs: `app/` UI, verifier, OpenVoice enrollment and render, all against `mock_asr.py` | Two tuned candidates plus an audio-in, audio-out shell already working on the mock |
| ~17:00, at 200 steps | **Kill-rule check.** Tuned not beating its own frozen baseline on the dev speaker? Stop the job, join the other lane | Same check, same rule | Both lanes still alive, or one dropped and two people on the survivor |
| 18:45 to 19:45 | Decode tuned Parakeet on the full dev set, fill its columns | Decode tuned Cohere on the same dev set, fill its columns | Gate table complete |
| 19:45 to 20:45 | Apply the gate. Deploy your model if it won, otherwise hand A's endpoint config to B and start profiling | Point `ASR_WS_URL` at the winner, finish streamed audio playback | **Product gate: speech, recovered text, confirmation, audible personal voice** |
| 20:45 to 22:45 | Build and benchmark FP8 against BF16 on the frozen winner | Sealed test decode once, then rolling-window session state, WebSocket partials, latency traces | Final held-out number plus one accuracy-approved serving config. BF16 stays the fallback |
| 22:45 to 23:45 | Noise robustness and clean-speech regression. Do not reopen model selection | Critical terms, voice identity, end-to-end p50 and p95 | Frozen deployment, populated scorecard, documented failures |
| 23:45 to 01:45 | Stratified failure analysis. Fix only reproducible serving defects | Blinded intelligibility and identity checks, harden clarification UX, capture before and after evidence | Final quality tables and a stable demo path |
| 01:45 to 02:45 | Freeze model and checkpoint IDs, capture evidence, rehearse | Freeze URLs, prepare recorded fallback inputs | Reliable demo and evidence bundle |
| 02:45 to 04:00 | Buffer for the one thing that breaks | Buffer | Nothing new starts |
| **04:00** | **FREEZE** | **FREEZE** | Demo-path bugfixes only |
| 04:00 to 08:00 | Video, Devpost links and screenshots, rehearsal, sleep | Same | Submitted well before 08:00 |

The 12 aggregate H100-hour budget is unchanged: 3 Parakeet training, 3 Cohere training, 2 dev evaluation, 4 winner endpoint plus FP8 benchmark. Only two GPUs are needed concurrently. Shut down idle workers.

## One-H100 fallback

Keep both models and the full training split, shorten both step budgets, and run the two jobs sequentially: Parakeet 15:45 to 17:45, Cohere 17:45 to 19:45. The kill rule matters more here, not less: a dead first job burns the second job's window. The 20:45 product gate may use the better **frozen** model if neither adapter has cleared evaluation. Phase 0 does not shrink.

## Changing the contract after Phase 0

1. Say it in chat and append it to `.workspace/hackathon-status.md` before editing.
2. Both people re-run `contract/evaluate.py` on their existing prediction files.
3. If numbers move, every prior scorecard row is void and must be re-decoded.

That third step is why the contract is frozen. After the promotion gate, do not touch it at all.

## Escalate immediately, do not queue

- A decision that changes the demo
- A spike assumption that turned out wrong
- A credential, account, or key you do not have
- A change to `contract/` or another person's owned file
- Pace: at this rate it will not be done by the freeze
- Three failed attempts at the same thing
- Usage exhausted mid-task: commit the WIP, write a handoff file, name the receiving tool

Never escalate library choice, naming, or formatting.

## Gates before anything merges

1. Typecheck
2. Production build
3. The artifact runs and produces the demo output
4. No errors on the demo path

For this project, gates 3 and 4 mean: `contract/mock_asr.py` plus `app/` complete one speech-to-personal-voice round trip with a clean browser console. Say which gates ran and what they said, never "should work".

## Open items needing a human decision

- **Devpost team, badge IDs, and sponsor prize selections locked at 14:00 today.** Confirm this was submitted. If it was not, that outranks every line above.
- `.workspace/hackathon-demo-script.md` does not exist yet. The handbook forbids feature work before it does. It is 30 minutes and it is the demo owner's first task.
