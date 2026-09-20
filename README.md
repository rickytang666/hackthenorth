# VoiceBridge

Speech recovery for dysarthric speakers, spoken back in the speaker's own voice.

## The problem

Dysarthria is a motor speech disorder: the muscles that produce speech are
weakened by stroke, cerebral palsy, Parkinson's, ALS or brain injury. The person
knows exactly what they want to say. The words come out slurred, slow or
imprecise, and listeners stop understanding them.

Ordinary speech recognition makes this worse rather than better. It does not
fail visibly; it produces fluent, confident, wrong English. On our sealed test
speaker the frozen baseline heard "why yelled a warrior of a silly agent" for
"why yell or worry over silly items", and reported high confidence while doing
it. For someone who depends on that transcript to be understood, a confident
wrong answer is worse than no answer.

VoiceBridge does three things about that:

1. **Recovers the words.** A LoRA adapter over Cohere Transcribe, fine-tuned on
   dysarthric speech, cuts word error rate 38.6% on a speaker it never heard.
2. **Asks when the evidence is weak** instead of guessing, using a calibrated
   confidence threshold, and offers the alternatives it was choosing between.
3. **Speaks the result in the person's own voice**, not a generic synthetic one,
   using a voice print taken from their own recordings.

## How it works

```text
  microphone or clip
         |  20 ms PCM frames over a WebSocket
         v
  +---------------------------+
  |  ASR (Protocol 1)         |   Cohere Transcribe 2.07B + LoRA adapter
  |  Baseten H100             |   partial transcript every 400 ms
  +---------------------------+
         |  text + per-token confidence
         v
  +---------------------------+
  |  confidence gate          |   >= 0.879  ->  accept
  |  contract/confidence.py   |   <  0.879  ->  ask, with beam-search n-best
  +---------------------------+
         |  confirmed text
         v
  +---------------------------+
  |  voice renderer           |   MeloTTS generates speech
  |  (Protocol 2)             |   OpenVoice V2 repaints it in the speaker's timbre
  +---------------------------+
         |
         v  audio, about one second after the sentence ends
```

Three design decisions carry most of the weight.

**Only the encoder matters.** Dysarthria is an acoustic problem, not a language
one, and the encoder holds 91.8% of this model's parameters. The adapter targets
the top 6 of 48 encoder blocks plus the decoder layers: 3.4M trainable weights,
**0.166% of the model**, trained in 10.6 minutes on one H100.

**Confidence is computed here, not supplied.** Neither candidate model ships a
usable confidence signal, so `contract/confidence.py` derives one from token
log-probabilities. The 0.879 accept threshold was calibrated on 244 held-out
decodes for 95% precision on accepted answers.

**Voice conversion, not voice cloning.** OpenVoice transfers timbre only and
leaves articulation to MeloTTS. That is the right tool here: cloning the whole
delivery from a dysarthric reference would reproduce the slurring, which is
precisely what the system exists to remove.

## Results

Speaker M02, held out from training and from every model-selection decision,
then decoded exactly once. 388 clips.

| Metric | Frozen baseline | Fine-tuned | |
|---|---:|---:|---|
| Word error rate | 0.5668 | **0.3481** | **38.6% better** |
| Isolated words | 1.0169 | **0.4628** | **54.5% better** |
| Critical errors (negation, yes/no, numbers, medications) | 0.4000 | **0.2857** | **28.6% better** |
| Clips exactly right | 98 / 388 | **198 / 388** | **2.0x** |

All eight dysarthric speakers improved, and critical errors worsened in none.
The gain grows with severity: **70.9%** on the most impaired speaker against
38.6% on M02. Full tables, the capacity sweep and the live end-to-end timings
are in [docs/RESULTS.md](docs/RESULTS.md).

## Limitations

- 34.8% WER on the sealed speaker is still high. The claim is a large relative
  reduction on an unseen speaker, not that the problem is solved.
- Two speakers are genuinely held out. That is evidence, not proof of
  generalization.
- TORGO is **read prompts**, isolated words and a fixed sentence set. We tested
  the adapter on spontaneous conversational speech from outside the corpus and
  it did not transfer: the frozen baseline scored 0.654 there and the tuned
  model 0.731. It is specialized, not universally better.
- Fine-tuning costs latency. LoRA adds matrix multiplications, moving p95 from
  154 ms to 216 ms per clip. At 39x real time this does not matter, but it is
  not a win.

## Run it

```bash
uv sync
source env.sh                      # VOICEBRIDGE_DATA defaults to ~/voicebridge-data
export BASETEN_API_KEY=...         # the page never sees this; app.serve holds it
```

Build the demo clips once. This needs the TORGO dataset, because the audio is
not redistributed in this repo:

```bash
uv run hf download abnerh/TORGO-database --repo-type dataset --local-dir "$VOICEBRIDGE_DATA/torgo-dataset"
uv run python -m data.prepare_torgo --source "$VOICEBRIDGE_DATA/torgo-dataset"
uv run python -m contract.manifest        # hashes must match contract/MANIFEST_HASHES
uv run python demo/curate_clips.py        # writes app/clips/*.wav
```

Then two processes. The first serves the page on 5173 and proxies the ASR
WebSocket on 8765, adding the Authorization header that a browser cannot set:

```bash
uv run python -m app.serve
```

```bash
cd serve/voice && uv run --no-sync python server.py --port 8770
```

Open <http://127.0.0.1:5173>, pick a clip, press **Recover speech**. The clip
plays while the transcript resolves against it; the recovered sentence is then
spoken back in the speaker's voice.

Verify the whole chain without a browser:

```bash
uv run python scripts/e2e_check.py
```

## Repository map

| Path | What |
|---|---|
| `contract/` | The shared ruler: manifests, normalizer, evaluator, confidence, protocols |
| `data/prepare_torgo.py` | Builds the five manifests. Deterministic, run once |
| `train/cohere/` | LoRA lane: pre-flight, collator, training, decode |
| `serve/asr_cohere/` | Cohere behind Protocol 1, deployed as a Baseten Truss |
| `serve/voice/` | OpenVoice V2 behind Protocol 2. Isolated environment, `numpy<2` |
| `app/` | The page, the static server and the ASR proxy |
| `demo/` | Clip curation and video ingestion |
| `scripts/e2e_check.py` | Drives the real chain end to end, nothing mocked |

## Things that will bite you

Each of these cost hours to find. The comments in the code say more.

- **Cohere does no language identification.** Without the decoder prompt prefix
  from `train/cohere/model.py` it transcribes dysarthric English into Arabic
  script at 0.9 confidence. Training and inference must use the same prefix.
- **The model does not shift labels.** Logits align 1:1 with `decoder_input_ids`,
  so the collator does teacher forcing itself. Skip this and loss still falls,
  and the checkpoint is worthless.
- **The LoRA targets are Fast-Conformer names**, not `q_proj`/`k_proj`. Run
  `train.cohere.preflight` before spending a GPU minute.
- **A WebSocket subprotocol cannot carry an API key.** It must be an RFC 6455
  token and `Api-Key <key>` contains a space. Baseten accepts header auth only,
  which is why `app/serve.py` proxies.
- **fp16 overflows this model's attention mask.** Use bfloat16 on MPS.
- **Cold start is 33 seconds** at `min_replica: 0`. Warm the endpoint first.

## Data and licensing

Built with the [TORGO database](https://huggingface.co/datasets/abnerh/TORGO-database)
of dysarthric and control speech. No TORGO audio or transcripts are
redistributed here; the manifests store paths relative to `$VOICEBRIDGE_DATA`
and the dataset is downloaded separately under its own terms.

Base model: `CohereLabs/cohere-transcribe-03-2026`, Apache 2.0. Voice rendering:
OpenVoice V2 and MeloTTS, MIT.
