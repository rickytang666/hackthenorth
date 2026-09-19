# VoiceBridge ownership and lanes

Two builders. This file binds both of you regardless of whether you write code by hand or with an agent. The technical agreement is in [DESIGN.md](DESIGN.md).

| Role | Person | Fill in |
|---|---|---|
| Person A, Parakeet lane and serving optimization, **demo owner** | | |
| Person B, Cohere lane and product path | | |

Person A is the demo owner because A's post-gate work (FP8, serving micro-optimization) is the most droppable thing on the board, and B's post-gate work (the product path) is not. The demo owner flips to full-time submission work at 22:45, which the clock below reflects.

## Phase 0 is written on one machine. Do not branch before it exits.

One person at the keyboard on one machine, the other getting their own machine and accounts ready. Parallelizing code before the seams exist is the most expensive mistake available here: two lanes independently invent two normalizers, two prediction schemas, and two ideas of what a partial hypothesis is, and then the hour-5 comparison is meaningless.

**Everything in the coding checklist below is generated once, on the Phase 0 machine, and reaches the other person through git.** The other person writes none of it. Their job in the same hour is to be able to pull it and use it immediately, which is the readiness checklist.

### Phase 0 coding checklist, one machine only

Start item 2 first and let it run unattended. The TORGO download plus WAV conversion is the long pole and everything else is written while it works.

| # | File | Does what | Done when |
|---|---|---|---|
| 1 | `data/prepare_torgo.py` | Download TORGO, convert to 16 kHz mono WAV, emit the five manifests | Five JSONL files exist, speaker-disjointness assertions pass, printed row and hour counts look sane |
| 2 | `contract/manifest.py` | Row schema, validator, and path resolution against `$VOICEBRIDGE_DATA` | Loads all five manifests, rejects a row with an absolute path |
| 3 | `contract/MANIFEST_HASHES` | SHA-256 of each of the five manifests | Committed. Both people will check against this |
| 4 | `contract/normalize.py` | The one text normalizer both lanes import | Handles case, punctuation, contractions, numbers. Has three worked examples in a docstring |
| 5 | `contract/predictions.py` | Prediction row schema and writer | Round-trips a row. Rejects a row missing `latency_ms` |
| 6 | `contract/critical_terms.txt` | Frozen safety slice: negation, yes/no, numbers, names, medications | Non-empty, sourced from the dev and test transcripts |
| 7 | `contract/evaluate.py` | WER, per-speaker WER, word vs sentence subsets, critical-error rate, RTF | Prints the full scorecard table from two synthetic prediction files |
| 8 | `contract/confidence.py` | Shared token log-prob scorer plus calibrated threshold | Returns a float on a real decode. NVIDIA's TDT utility is broken, this replaces it |
| 9 | `contract/decode.py` | Shared decode loop taking a `transcribe(paths) -> list[str]` callable | Produces a valid prediction JSONL from a stub callable |
| 10 | `contract/protocol.md` | The three wire protocols, normative | Every field in DESIGN.md's protocol section appears here |
| 11 | `contract/mock_asr.py` | Fake ASR server replaying a fixture JSONL on a timer | Streams protocol-correct partials and one final over a WebSocket |
| 12 | `serve/_template/` | Truss that already speaks Protocol 1 against a stub model | Deploys, or at minimum runs locally and answers a WebSocket |
| 13 | `bench/latency.py` | Drives Protocol 1, writes a scorecard row | Reports a number against `mock_asr.py` |
| 14 | `.gitignore` | Stack-specific half | Done. Last moment it is free to edit |
| 15 | `plans/` and `README.md` | Committed and pushed | The other person can clone and read |

### Readiness checklist, each person on their own machine

Both of you run this list. Neither of you is ready to branch until your own column is clear.

**Accounts and credentials**

| # | Item | Verified by |
|---|---|---|
| 16 | Baseten account, billing active, H100 quota confirmed | `baseten train push` on a hello-world job returns a job ID |
| 17 | `BASETEN_API_KEY` in your shell, not in a file | `echo $BASETEN_API_KEY` is non-empty and the key is in `.env`, which is gitignored |
| 18 | Hugging Face account and `HF_TOKEN` | `hf auth whoami` returns your username |
| 19 | Cohere repo conditions accepted | `hf download CohereLabs/cohere-transcribe-03-2026 --include "*.json"` succeeds. Contact-info click-through, Apache 2.0, two minutes. Never a CC-BY-NC mirror |
| 20 | Verifier LLM API key | One round-trip returns strict JSON |
| 21 | GitHub push access to this repo | You have pushed one commit |

**Data and models on your machine**

| # | Item | Verified by |
|---|---|---|
| 22 | TORGO downloaded, 1.56 GB | `hf download abnerh/TORGO-database --repo-type dataset` completed |
| 23 | `source env.sh`, then the WAVs resolve | `python -c "import contract.manifest as m; m.load_all()"` resolves every path |
| 24 | **The five manifest hashes match `contract/MANIFEST_HASHES`** | You ran `prepare_torgo.py` yourself and got identical hashes. A mismatch means stop, do not train |
| 25 | Your lane's base checkpoint downloaded | Parakeet: `ASRModel.from_pretrained` loads. Cohere: `from_pretrained` loads |
| 26 | OpenVoice V2 weights downloaded | One synthesis produces audible audio |

**Toolchain**

| # | Item | Verified by |
|---|---|---|
| 27 | GPU visible, CUDA working | `torch.cuda.is_available()` is `True` and reports the right device |
| 28 | Lane dependencies installed | A: NeMo imports. B: Transformers plus PEFT import |
| 29 | One backward pass completed locally on your own base model | Loss is finite, `loss.backward()` returns, trainable parameter count is non-zero and plausible |
| 30 | You can run `contract/evaluate.py` and reproduce the printed scorecard | Same numbers as the Phase 0 machine |

**Legal and consent, whoever gets there first**

| # | Item |
|---|---|
| 31 | TORGO license confirmed to permit academic non-profit use and third-party cloud processing on Baseten |
| 32 | One consented voice enrollment recording captured, with the consenting person told it can be deleted |
| 33 | Decided whether the five manifests may be committed. They contain TORGO transcripts and this repo goes public at submission. Default is no: commit `MANIFEST_HASHES` only and have each person regenerate |

**Exit gate, all six, not a clock:**

1. `contract/evaluate.py` prints the full scorecard table from synthetic prediction files
2. `contract/mock_asr.py` streams protocol-correct partials and `bench/latency.py` reports a number from it
3. Both people independently reproduce the five manifest SHA-256 hashes (items 22 to 24)
4. Both people have a trainable checkpoint loaded and one local backward pass completed (item 29)
5. `contract/confidence.py` returns a score on a real decode, because NVIDIA's TDT confidence utility is broken and both lanes must threshold identically
6. One 7-word OpenVoice synthesis is timed and written down, which decides whether streaming TTS is built at all

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
|
|-- train/cohere/ .............. B
|-- serve/asr_cohere/ .......... B
|-- serve/voice/ ............... B
|-- app/ ....................... B
|-- .gitignore ................. PHASE 0 (A), then B
|-- package.json / lock files .. B  (B installs ALL dependencies)
|-- pyproject.toml / lock ...... B
|
|-- README.md .................. either, regenerated at phase boundaries
|-- plans/DESIGN.md ............ A
|-- plans/OWNERSHIP.md ......... either, append only
|-- plans/voicebridge-plan.md .. either, append only. Devpost source
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
| 14:30 to 15:45 | Keyboard: `contract/`, manifests, confidence scorer, mock, template, bench | Provisioning list, then verify hashes, run a local backward pass, and time one 7-word OpenVoice synthesis | **All six Phase 0 gates. Branch only now.** |
| 15:45 to 18:45 | Launch Parakeet NeMo job on H100-A, checkpoint every 200 to 250 steps. While it runs: `serve/asr_parakeet/` against the frozen base, latency harness | Launch Cohere LoRA job on H100-B (top 6 encoder blocks plus decoder, ~400 steps). While it runs: `app/` UI, verifier, OpenVoice enrollment and render, all against `mock_asr.py` | Two tuned candidates plus an audio-in, audio-out shell already working on the mock |
| ~17:00, at 200 steps | **Kill-rule check.** Tuned not beating its own frozen baseline on the dev speaker? Stop the job, join the other lane | Same check, same rule | Both lanes still alive, or one dropped and two people on the survivor |
| 18:45 to 19:45 | Decode tuned Parakeet on the full dev set, fill its columns | Decode tuned Cohere on the same dev set, fill its columns | Gate table complete |
| 19:45 to 20:45 | Apply the gate. Deploy your model if it won, otherwise help B wire the Cohere adapter and start profiling the common endpoint | Point `ASR_WS_URL` at the winner, finish streamed audio playback | **Product gate: speech, recovered text, confirmation, audible personal voice** |
| 20:45 to 22:45 | Build and benchmark FP8 against BF16 on the frozen winner | Sealed test decode once, then rolling-window session state, WebSocket partials, latency traces | Final held-out number plus one accuracy-approved serving config. BF16 stays the fallback |
| 22:45 to 23:45 | **Flip to demo owner.** Noise robustness and clean-speech regression, then hand serving work to B | Critical terms, voice identity, end-to-end p50 and p95 | Frozen deployment, populated scorecard, documented failures |
| 23:45 to 01:45 | Demo script, video plan, Devpost draft, screenshot plan. Stratified failure analysis only if time allows | Blinded intelligibility and identity checks, harden clarification UX, capture before and after evidence | Final quality tables and a stable demo path |
| 01:45 to 02:45 | Freeze model and checkpoint IDs, capture evidence, rehearse | Freeze URLs, prepare recorded fallback inputs | Reliable demo and evidence bundle |
| 02:45 to 04:00 | Buffer for the one thing that breaks | Buffer | Nothing new starts |
| **04:00** | **FREEZE** | **FREEZE** | Demo-path bugfixes only |
| 04:00 to 08:00 | Video, Devpost links and screenshots, rehearsal, sleep | Same | Submitted well before 08:00 |

The 12 aggregate H100-hour budget is unchanged: 3 Parakeet training, 3 Cohere training, 2 dev evaluation, 4 winner endpoint plus FP8 benchmark. Only two GPUs are needed concurrently. Shut down idle workers.

## One-H100 fallback

Keep both models and the full training split, shorten both step budgets, and run the two jobs sequentially: Parakeet 15:45 to 17:45, Cohere 17:45 to 19:45. Three things shift and must shift together:

- Each lane's kill-rule check happens 200 steps into **its own** window, not at a shared 17:00. Cohere has not started at 17:00.
- The dev decode cannot start before 19:45, so the promotion gate slips to **~20:45** and every row after it slides one hour.
- The product gate therefore uses the better **frozen** model if neither adapter has cleared evaluation. This is expected, not a failure.

The kill rule matters more here, not less: a dead first job burns the second job's window. Phase 0 does not shrink.

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
