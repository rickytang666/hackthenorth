# VoiceBridge

Speech recovery for dysarthric speakers, spoken back in the speaker's own voice.

## The problem

Dysarthria is a motor speech disorder: stroke, cerebral palsy, Parkinson's or
ALS weaken the muscles that produce speech. The person knows what they want to
say. It comes out slurred, and listeners stop understanding them.

Ordinary speech recognition makes this worse. It does not fail visibly, it
produces fluent, confident, wrong English. On our held-out speaker the baseline
heard *"why yelled a warrior of a silly agent"* for *"why yell or worry over
silly items"*, and reported high confidence doing it. When you depend on that
transcript to be understood, a confident wrong answer is worse than none.

VoiceBridge does three things:

1. **Recovers the words.** A LoRA adapter over Cohere Transcribe cuts word error
   rate 38.6% on a speaker it never heard.
2. **Asks when the evidence is weak** instead of guessing, and shows the
   alternatives it was choosing between.
3. **Speaks the result in the person's own voice**, from a voice print taken
   from their own recordings.

## How it works

```text
  microphone or clip
         |  20 ms PCM frames over a WebSocket
         v
  ASR, Protocol 1           Cohere Transcribe 2.07B + LoRA, on Baseten H100
  Baseten H100              partial transcript every 400 ms
         |  text + per-token confidence
         v
  confidence gate           >= 0.879  accept
                            <  0.879  ask, with beam-search alternatives
         |  confirmed text
         v
  voice renderer            MeloTTS speaks it
  Protocol 2                OpenVoice V2 repaints it in the speaker's timbre
         |
         v  audio, about a second after the sentence ends
```

Three decisions carry the weight:

- **Only the encoder matters.** Dysarthria is acoustic, not linguistic, and the
  encoder holds 91.8% of the parameters. The adapter trains 3.4M weights,
  **0.166% of the model**, in 10.6 minutes on one H100.
- **Confidence is computed here.** Neither model ships a usable one, so we derive
  it from token log-probabilities and calibrated the 0.879 threshold on 244
  held-out decodes for 95% precision.
- **Voice conversion, not voice cloning.** OpenVoice transfers timbre only.
  Cloning the full delivery would reproduce the slurring we exist to remove.

### Stack

- **ASR** Cohere Transcribe 03-2026 (2.07B, Apache 2.0), LoRA via PEFT
- **Training and serving** Baseten, one H100, Truss with a WebSocket transport
- **Voice** OpenVoice V2 tone-color conversion over MeloTTS
- **Data** TORGO dysarthric speech corpus, 8 impaired speakers
- **Frontend** one static HTML file, no framework, no build step
- **Python** 3.13, uv, PyTorch

## Results

Speaker M02, held out from training and from every model-selection decision,
then decoded once. 388 clips.

| Metric | Baseline | Fine-tuned | |
|---|---:|---:|---|
| Word error rate | 0.5668 | **0.3481** | **38.6% better** |
| Isolated words | 1.0169 | **0.4628** | **54.5% better** |
| Safety-critical errors | 0.4000 | **0.2857** | **28.6% better** |
| Clips exactly right | 98 / 388 | **198 / 388** | **2.0x** |

All eight speakers improved, and safety-critical errors worsened in none. The
gain grows with severity: **70.9%** on the most impaired speaker. Full tables in
[docs/RESULTS.md](docs/RESULTS.md).

## Limitations

- 34.8% error on the held-out speaker is still high. The claim is the relative
  reduction, not that the problem is solved.
- Two of eight speakers are genuinely held out. Evidence, not proof.
- TORGO is read prompts. On spontaneous conversational speech from outside the
  corpus the adapter did not transfer: baseline 0.654, tuned 0.731. It is
  specialized, not universally better.
- LoRA costs latency, p95 154 ms to 216 ms. At 39x real time this does not
  matter, but it is not a win.

## Run it

Needs a Baseten API key and the TORGO dataset, which is not redistributed here.

```bash
uv sync && source env.sh
export BASETEN_API_KEY=...        # the page never sees this; app/serve.py holds it

uv run hf download abnerh/TORGO-database --repo-type dataset --local-dir "$VOICEBRIDGE_DATA/torgo-dataset"
uv run python -m data.prepare_torgo --source "$VOICEBRIDGE_DATA/torgo-dataset"
uv run python -m contract.manifest
uv run python demo/curate_clips.py
```

Then two processes, and open <http://127.0.0.1:5173>:

```bash
uv run python -m app.serve                                    # page + ASR proxy
cd serve/voice && uv run --no-sync python server.py           # voice renderer
```

`uv run python scripts/e2e_check.py` drives the same chain without a browser.

Built with the [TORGO database](https://huggingface.co/datasets/abnerh/TORGO-database),
downloaded separately under its own terms. Base model Apache 2.0; OpenVoice V2
and MeloTTS MIT.
