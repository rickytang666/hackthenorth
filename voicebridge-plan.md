# VoiceBridge

## Real-time speech recovery that still sounds like the person speaking

**Proposal date:** September 19, 2026
**Hackathon tracks:** Baseten — Best Use of Baseten; Rox — Best AI Agent

> A stroke survivor may know exactly what they want to say while weakened or poorly coordinated speech muscles make their words difficult for people and conventional speech recognition to understand. VoiceBridge recovers the intended words, asks instead of guessing when evidence is weak, and speaks the result in a voice that preserves the person's identity, accent, tone, pace, and style.

## The moment this project is trying to solve

### Jim: living with dysarthria after a stroke

<video controls preload="metadata" style="max-width: 100%;" src="https://www.stroke4carers.org/wp-content/uploads/DYSARTHRIA.mp4">
  Your Markdown viewer does not support embedded video. Use the link below.
</video>

**[Play Jim's 4:13 video](https://www.stroke4carers.org/wp-content/uploads/DYSARTHRIA.mp4)** · **[Source page and transcript](https://www.stroke4carers.org/?p=5409)**

Jim was filmed nine months after his stroke. He understands the conversation, but dysarthria makes producing and projecting speech difficult. He explains that particular words can be hard to understand over the telephone.

This is not a lack of intelligence or a lack of ideas. It is a damaged communication channel. VoiceBridge attempts to repair that channel without replacing the person behind it.

The recording is linked from its original publisher rather than redistributed. Public viewing is not permission to use Jim's likeness or voice for training, marketing, or a hackathon presentation. Request permission before publicly showing an excerpt. **The clip motivates the problem; it is not training data.**

## Why this is needed

- The American Speech-Language-Hearing Association estimates that **22%–58% of people with acute stroke present with dysarthria**. It can affect pronunciation, loudness, rate, and intonation, with substantial variation between people. [ASHA: Dysarthria in Adults](https://www.asha.org/Practice-Portal/Clinical-Topics/Dysarthria-in-Adults/)
- Heart & Stroke Canada notes that post-stroke communication problems can make it difficult to socialize or share thoughts and feelings, and can cause people to be treated as though they do not understand. [Heart & Stroke Canada](https://www.heartandstroke.ca/stroke/recovery-and-support/physical-changes/communication)
- Typing every sentence is slow and may be inaccessible to someone with impaired hand movement.
- Generic ASR is optimized for typical speech. Dysarthria changes articulation, timing, loudness, pitch, and breath support in speaker-specific ways.
- A generic synthetic voice solves intelligibility by erasing identity. For a daily communication system, sounding like oneself is part of agency rather than cosmetic polish.

The initial target is dysarthria: intended language is substantially preserved, but motor production is atypical. Aphasia and apraxia can co-occur, but they are different conditions and should not be silently grouped into the same claim.

## Product definition

VoiceBridge is an audio-first, low-latency application for a phone or laptop.

1. A streaming recognizer adapted to dysarthric speech produces partial text and an evidence-backed candidate set.
2. A constrained verifier may use approved conversational context to select among candidates, but cannot invent unrestricted text.
3. A calibrated policy either accepts the interpretation, shows two or three accessible choices, or asks for repetition.
4. A reference-conditioned speech synthesizer speaks the confirmed text using the person's tone color, accent, rhythm, pace, pauses, and intonation.

The minimum end-to-end interaction is:

```text
speech -> corrected text -> confirmation when needed -> the person's recovered voice
```

### Product requirements

- **Meaning comes first:** preserve negation, names, numbers, medication terms, and yes/no answers.
- **Identity is mandatory:** output should resemble the speaker rather than a generic assistant.
- **Real time means conversational:** stream partial results and minimize time to first useful audio, not merely offline real-time factor.
- **Abstention is a feature:** uncertain speech should trigger a small clarification, not a fluent hallucination.
- **Every optimization earns its place:** use Baseten capabilities only when they improve accuracy, latency, reliability, or measurable deployment efficiency.

### Research claim

> Under one speaker-disjoint TORGO protocol, supervised adaptation can reveal whether a compact transducer (Parakeet-TDT 0.6B v2) or a larger encoder–decoder with LoRA (Cohere Transcribe) provides the better dysarthria quality–latency tradeoff; measured inference optimization then reduces the winner's conversational latency without exceeding a fixed accuracy regression, while reference-conditioned synthesis preserves measurable speaker and prosody characteristics.

## Existing work: reference points, not the project thesis

These systems and papers are useful engineering references. VoiceBridge does not need to win by arguing that every alternative is inadequate.

| Reference | Lesson used by VoiceBridge |
|---|---|
| [Google Project Relate](https://sites.research.google/relate/) | Personalized data can materially improve recognition of non-standard speech. |
| [Voiceitt](https://voiceitt.com/) | Accessible enrollment and user-confirmed corrections matter in a real interaction. |
| [Whispp](https://www.whispp.com/how-it-works) | Low-latency reconstruction and preservation of voice identity are important product requirements. |
| [Apple Personal Voice and Live Speech](https://www.apple.com/accessibility/features/) | A personal synthetic voice can preserve agency, privacy, and familiarity. |
| [Project Relate research](https://arxiv.org/abs/1907.13511) | Selective adaptation can produce large gains with relatively little atypical-speech data. |

They are included for reference and design lessons, not as a competitive scorecard.

## Research foundation

### Dysarthria adaptation

[Shor et al., Interspeech 2019](https://www.isca-archive.org/interspeech_2019/shor19_interspeech.html) reported a 62% relative WER improvement for personalized models in its dysarthric cohort and found that adapting selected layers could outperform full-model fine-tuning. [Tobin and Tomanek](https://arxiv.org/abs/2110.04612) likewise found useful personalized ASR performance with only minutes of speaker-specific recordings.

The winning system in the 2025 Speech Accessibility Project Challenge fine-tuned Parakeet-TDT and reduced WER from the organizers' Whisper-large-v2 baseline of 17.82 to 8.11. [Interspeech 2025 paper](https://www.isca-archive.org/interspeech_2025/takahashi25_interspeech.html)

This supports the central decision: start from a strong speech foundation model and adapt a small set of weights to dysarthric speech rather than training a speech model from scratch.

### Streaming inference

A streaming speech model should not re-encode the entire conversation whenever a new audio chunk arrives. A shifted or chunked encoder can preserve left context while processing only new frames. [Shifted Chunk Encoder](https://arxiv.org/abs/2203.15206)

[FlashAttention](https://arxiv.org/abs/2205.14135) reduces high-bandwidth-memory traffic for exact attention. Combined with duration bucketing, preallocated tensors, CUDA Graphs, and asynchronous preprocessing, it can reduce avoidable serving overhead.

### Identity-preserving synthesis

[OpenVoice](https://arxiv.org/abs/2312.01479) demonstrates reference-conditioned voice cloning with separate control of tone color and styles including emotion, accent, rhythm, pauses, and intonation. [StyleTTS 2](https://arxiv.org/abs/2306.07691) demonstrates strong zero-shot speaker adaptation and explicit style modeling.

VoiceBridge should use a pretrained reference-conditioned synthesizer, not train a TTS foundation model during the hackathon. The technical contribution is the controlled handoff from recovered linguistic content to a preserved speaker/style representation.

## Proposed architecture

```text
microphone audio
      |
      v
streaming feature extractor
  ring buffer + cached chunk state
      |
      v
FP8/BF16 dysarthria ASR
  winner: TORGO-adapted Parakeet or Cohere
      |
      +------ high confidence ------> confirmed text
      |
      +------ low confidence -------> constrained verifier
                                        | select / clarify / abstain
                                        v
                                   confirmed text
                                        |
                                        v
reference-conditioned voice synthesis
  speaker embedding + source prosody controls
                                        |
                                        v
                             recovered personal voice
```

### 1. Streaming dysarthria recognizer

- Person A adapts `nvidia/parakeet-tdt-0.6b-v2` in NeMo and keeps `nvidia/parakeet-tdt-1.1b` frozen as a historical benchmark.
- Person B adapts `CohereLabs/cohere-transcribe-03-2026` with supervised PEFT/LoRA on the identical TORGO manifest.
- Compare both frozen and tuned models with the same speaker-disjoint evaluator, then deploy only the winner.
- Save the winning self-contained `.nemo` or merged Transformers checkpoint for deployment.
- Emit partial hypotheses, candidate scores, SNR, and calibrated confidence.

### 2. Constrained recovery policy

The verifier receives only:

- the acoustic candidate set;
- token-level or sequence-level confidence;
- an explicitly approved vocabulary and topic;
- immutable rules protecting negation, names, numbers, and medication terms.

It returns strict JSON containing either a selected candidate, a clarification choice, or abstention. It cannot produce arbitrary replacement prose.

### 3. Personal voice renderer

Enrollment uses short, consented reference recordings. Store separate controls for:

- **speaker identity/tone color** from a speaker embedding;
- **accent and pronunciation style** from reference style features;
- **pace and pauses** from source timing aligned to recovered tokens;
- **intonation and emotional tone** from a smoothed F0/energy contour.

The recognizer repairs linguistic content; the renderer restores identity. Do not force the synthesizer to copy dysarthric articulation that reduces intelligibility. Preserve higher-level pace and expressiveness while generating clear phonemes. Provide a user-adjustable “more like me ↔ more clear” control because perfect identity preservation and maximum intelligibility may conflict.

If pre-stroke recordings exist, the user may opt to use them. Otherwise the system derives tone color and style from current recordings. Voice enrollment and deletion must be explicit because a voice model is sensitive personal data.

### 4. Required text-to-personal-voice path

Corrected text is not the final product. Every confirmed phrase must be rendered back into an intelligible version of the user's own voice.

Use **[OpenVoice V2](https://github.com/myshell-ai/OpenVoice)** as the primary hackathon renderer. It supports zero-shot tone-color cloning plus controls for accent, rhythm, pauses, intonation, and emotion, and its code and released models are MIT-licensed. This avoids spending the 12-hour H100 budget training a TTS model. The serving sequence is:

```text
10–30 s consented enrollment audio
        -> VAD + clipping/SNR checks
        -> speaker/tone-color embedding (computed once and cached)

stable confirmed ASR clause + source timing/F0 summary
        -> clear base speech
        -> OpenVoice tone-color/style conversion
        -> PCM/Opus chunks over WebSocket
        -> recovered personal voice
```

Only synthesize a clause after its ASR prefix is stable or the user confirms it; otherwise the system may speak a word and immediately contradict itself. Segment at punctuation or a 300–500 ms pause, begin rendering the first stable clause while later speech is still being recognized, and cross-fade adjacent audio chunks. Cache the enrollment embedding and immutable model state—not generated health-related content.

Implement the renderer behind a small interface so the ASR experiment is independent of the TTS choice:

```python
voice_id = renderer.enroll(reference_wavs)  # once per consented session
audio_stream = renderer.synthesize_stream(
    text=confirmed_text,
    voice_id=voice_id,
    pace=source_pace,
    pitch_contour=smoothed_source_f0,
)
```

The hour-6 acceptance test is end to end: dysarthric audio in, corrected text displayed, then audible personalized speech out. Target **<1.5 seconds to first recovered audio**, real-time factor below 0.5, and no more than 250 ms between synthesized clauses. If OpenVoice chunking is not ready, use a generic pretrained voice as the reliability fallback rather than dropping audio output.

Baseten also publishes an [XTTS V2 streaming deployment](https://www.baseten.co/blog/streaming-real-time-text-to-speech-with-xtts-v2/) that can serve as a fast implementation reference or hackathon fallback. Its pretrained weights use the non-commercial Coqui Public Model License, so do not make it the unrestricted product path. Run the renderer on a separate lower-cost Baseten GPU or alongside the winning endpoint only if profiling shows sufficient headroom; its runtime is outside the 12 experimental H100-hours reserved for ASR training and comparison.

For a person whose current recordings contain severe dysarthric articulation, enrollment should select the cleanest available clips and extract identity from several references. The system can preserve timbre, accent, broad pace, and expressiveness, but it must not claim to reconstruct a lost pre-stroke voice without actual pre-stroke recordings.

## Data plan

### Selected dataset: TORGO

VoiceBridge will use the packaged **[`abnerh/TORGO-database`](https://huggingface.co/datasets/abnerh/TORGO-database)** release for the hackathon. It is a 1.56 GB English audio-text dataset containing approximately **5.5 hours of dysarthric recordings** and **8 hours of matched control recordings**. Those figures count the available microphone recordings; unique spoken content is lower because TORGO includes paired microphone views of some utterances. The package has 16,552 rows drawn from short words and restricted sentences. TORGO contains eight dysarthric speakers—three female and five male—with cerebral palsy or ALS. Its license permits academic, non-profit use with citation; confirm that the event and any downstream deployment remain within those terms.

Download the complete packaged dataset before the H100 clock starts:

```bash
hf download abnerh/TORGO-database \
  --repo-type dataset \
  --local-dir ./torgo-dataset
```

This repository embeds audio and exposes `audio`, `transcription`, `speech_status`, `gender`, and `duration`. The primary adaptation corpus is **only rows where `speech_status == "dysarthria"`**. Control recordings are not treated as positive training examples: reserve them for a small clean-speech replay set and a frozen regression test.

### What the TORGO labels mean

TORGO pairs read recordings with orthographic prompt text. This is useful supervision for VoiceBridge: a dysarthric acoustic realization maps to ordinary written English. However, the public documentation does not establish that every label was edited into a verbatim account of false starts, omissions, or substitutions. Therefore:

- use the packaged `transcription` as the ASR target;
- exclude empty labels, `xxx`, non-word tasks, and obvious recording failures;
- keep both microphone views only for training speakers, but evaluate only `headMic` so paired recordings are not double-counted in metrics;
- manually audit every dev/test recording against its transcript;
- describe the experiment as **dysarthric read-speech adaptation**, not open-conversation or clinical validation.

The main limitation is speaker diversity, not utterance count. Five and a half hours is sufficient for a controlled LoRA/partial-fine-tuning experiment, but eight speakers cannot establish generalization across all causes or severities of dysarthria.

### Dataset and split decision

1. Build the main train/dev/test sets entirely from `speech_status == "dysarthria"`.
2. Split by speaker before any sampling: six speakers for training, one for validation, and one untouched for testing. Freeze this assignment before baseline decoding.
3. Use every clean dysarthric word and restricted sentence from the six training speakers. The two microphone channels may act as acoustic augmentation during training, but they share one utterance-group ID and can never cross a speaker split.
4. Sample control-speaker audio into `clean_replay.jsonl` at no more than 10% of training batches, and keep a disjoint `clean_eval.jsonl` for the catastrophic-forgetting gate.
5. Report word-level and sentence-level subsets separately, because a model can look strong by memorizing TORGO's repeated isolated-word vocabulary.
6. Keep SAP as a future scale-up dataset only; it is not part of this hackathon's training or timing plan.

### Label policy

Every manifest row should retain the distinction explicitly:

```text
audio_path
speaker_id
diagnosis_and_severity
prompt_text             # what the participant was asked to say
verbatim_transcript     # what a human annotator judged was produced, if available
training_target         # chosen for this experiment
target_type             # intended_prompt | verbatim | consensus | uncertain
label_confidence
quality_flags
```

- For clean TORGO attempts, use `transcription` as the intended ASR label.
- If the attempt contains a restart, large omission, extra speech, or an obviously abandoned reading, exclude it from the 12-hour training set unless a verified verbatim transcript exists.
- Never treat a generic ASR model's pseudo-transcript as ground truth. It can flag suspicious rows for review, but dysarthric speech is exactly where that model is biased.
- Manually audit a speaker-balanced training sample and every validation/test utterance. For ambiguous test audio, require agreement from two listeners plus adjudication rather than silently copying the prompt. Training-label noise is tolerable in moderation; evaluation-label noise invalidates the headline result.

### Compiled corpus pipeline

1. Track dataset, speaker, speech status, microphone channel, duration, label type, license, and permitted use in one manifest.
2. Preserve prompt text, verbatim response, and normalized training target as separate fields.
3. Resample to 16 kHz mono and trim only non-speech margins.
4. Deduplicate using transcript hashes and acoustic fingerprints.
5. Split by speaker, never randomly by utterance.
6. Balance batches by speaker and severity.
7. Add a small typical-speech replay fraction to reduce catastrophic forgetting.
8. Freeze a critical slice containing names, numbers, negation, yes/no, short answers, and noise before training.
9. Cache resampled audio or log-Mel features before starting the H100 clock.

## Training plan

### Model selection: Parakeet-TDT 0.6B v2 as the primary candidate

Person A trains **[`nvidia/parakeet-tdt-0.6b-v2`](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v2)** as VoiceBridge's primary low-latency candidate. Keep **[`nvidia/parakeet-tdt-1.1b`](https://huggingface.co/nvidia/parakeet-tdt-1.1b)** frozen as a historical comparison benchmark. In parallel, Person B adapts Cohere Transcribe on exactly the same dysarthric TORGO train set. The winner is selected on the fixed held-out validation speaker using WER, critical errors, and serving latency rather than reputation or generic leaderboard rank.

| Model | Parameters | Reported training audio | Generic English WER | Role |
|---|---:|---:|---:|---|
| Parakeet-TDT 0.6B v2 | 0.6B | ~120,000 h English | 6.05 | **Primary fine-tuning base** |
| Parakeet-TDT 1.1B | 1.1B | ~64,000 h English | 7.02 | Frozen comparison; architecture used by the 2025 SAP winner |
| Parakeet-TDT 0.6B v3 | 0.6B | Large multilingual corpus | 6.32 English | Reference only; multilingual but not needed for the English core |

V2 combines roughly 10,000 human-transcribed hours with about 110,000 pseudo-labelled hours and a final high-quality human-labelled adaptation stage. It is smaller, faster, and currently stronger on generic English than 1.1B. The 1.1B checkpoint remains scientifically useful because the published SAP-winning system selected and fine-tuned that architecture, reaching 8.11 WER. [Parakeet v2 model card](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v2) · [SAP winner](https://www.isca-archive.org/interspeech_2025/takahashi25_interspeech.html)

### Experimental contract

The two owners must share the exact manifests, audio bytes, normalization code, prediction schema, evaluator, and hardware class. Each model uses its documented deterministic decoding path; do not force Parakeet-specific TDT settings onto Cohere.

```text
same speaker-disjoint evaluation manifest
same audio bytes
same text normalization
same WER and critical-error evaluator
same H100 class and latency measurement protocol

frozen Parakeet 0.6B v2  -> baseline_parakeet.jsonl
TORGO-tuned Parakeet v2  -> tuned_parakeet.jsonl
frozen Parakeet 1.1B     -> comparison_parakeet_1p1b.jsonl
frozen Cohere Transcribe -> baseline_cohere.jsonl
TORGO LoRA Cohere        -> tuned_cohere.jsonl
```

Every prediction row must contain `audio_filepath`, `speaker_id`, `text`, `prediction`, `model_id`, and `latency_ms`. Do not choose a deliberately weak baseline to inflate improvement. Report each model's own before/after gain as well as the final cross-model comparison.

### Build dysarthric-only TORGO manifests

Download the 1.56 GB repository once, then convert its embedded audio into stable local WAV paths. Split by speaker—not utterance. Keep both microphones for training-speaker augmentation and only `headMic` for validation/test scoring. The explicit speaker assignment below is frozen before any baseline is scored; rebalance it only once, based on duration and metadata rather than model performance.

```bash
hf download abnerh/TORGO-database \
  --repo-type dataset \
  --local-dir ./torgo-dataset
```

```python
import glob
import json
from pathlib import Path

import soundfile as sf
from datasets import Audio, load_dataset

SOURCE = "./torgo-dataset"
OUT = Path("/data/voicebridge")
WAV_OUT = OUT / "torgo_wav"
OUT.mkdir(parents=True, exist_ok=True)
WAV_OUT.mkdir(parents=True, exist_ok=True)

parquet_files = sorted(glob.glob(f"{SOURCE}/**/*.parquet", recursive=True))
assert parquet_files, "No Parquet shards found; verify the hf download completed"
dataset = load_dataset("parquet", data_files=parquet_files, split="train")

def add_ids(row):
    original_path = row["audio"]["path"]
    stem = Path(original_path).stem
    parts = stem.split("_")
    return {
        "speaker_id": parts[0],
        "mic": parts[2] if len(parts) >= 4 else "unknown",
        "utterance_id": stem,
        "utterance_group": f"{parts[0]}_{parts[1]}_{parts[3]}"
        if len(parts) >= 4 else stem,
    }

dataset = dataset.map(add_ids)
dataset = dataset.cast_column("audio", Audio(sampling_rate=16000))

# Eight TORGO dysarthric speakers. Freeze before looking at model results.
DEV_SPEAKERS = {"F04"}
TEST_SPEAKERS = {"M02"}

dysarthric_rows, control_rows = [], []
for row in dataset:
    text = row["transcription"].strip()
    if not text or text.lower() == "xxx":
        continue
    audio = row["audio"]
    wav_path = WAV_OUT / f'{row["utterance_id"]}.wav'
    sf.write(wav_path, audio["array"], 16000, subtype="PCM_16")
    record = {
        "audio_filepath": str(wav_path.resolve()),
        "text": text,
        "duration": float(row["duration"]),
        "speaker_id": row["speaker_id"],
        "speech_status": row["speech_status"],
        "mic": row["mic"],
        "utterance_id": row["utterance_id"],
        "utterance_group": row["utterance_group"],
    }
    if row["speech_status"] == "dysarthria":
        dysarthric_rows.append(record)
    else:
        control_rows.append(record)

def split_name(row):
    if row["speaker_id"] in DEV_SPEAKERS:
        return "dev"
    if row["speaker_id"] in TEST_SPEAKERS:
        return "test"
    return "train"

def write_jsonl(path, rows):
    with Path(path).open("w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")

train = [r for r in dysarthric_rows if split_name(r) == "train"]
dev = [
    r for r in dysarthric_rows
    if split_name(r) == "dev" and r["mic"].lower() == "headmic"
]
test = [
    r for r in dysarthric_rows
    if split_name(r) == "test" and r["mic"].lower() == "headmic"
]

assert {r["speaker_id"] for r in train}.isdisjoint(
    {r["speaker_id"] for r in dev + test}
)
assert {r["speaker_id"] for r in dev}.isdisjoint(
    {r["speaker_id"] for r in test}
)

write_jsonl(OUT / "torgo_dys_train.jsonl", train)
write_jsonl(OUT / "torgo_dys_dev.jsonl", dev)
write_jsonl(OUT / "torgo_dys_test.jsonl", test)

# Controls never replace dysarthric training data. Use at most 10% replay and
# keep a separate speaker-disjoint control slice for regression measurement.
control_rows = [r for r in control_rows if r["mic"].lower() == "headmic"]
control_speakers = sorted({r["speaker_id"] for r in control_rows})
assert len(control_speakers) >= 2
CONTROL_EVAL_SPEAKERS = set(control_speakers[-2:])
replay_pool = [
    r for r in control_rows if r["speaker_id"] not in CONTROL_EVAL_SPEAKERS
]
clean_eval = [
    r for r in control_rows if r["speaker_id"] in CONTROL_EVAL_SPEAKERS
]
replay_pool.sort(key=lambda r: (r["speaker_id"], r["utterance_id"]))
replay_count = min(len(replay_pool), max(1, len(train) // 10))
clean_replay = replay_pool[:replay_count]
assert {r["speaker_id"] for r in clean_replay}.isdisjoint(
    {r["speaker_id"] for r in clean_eval}
)
write_jsonl(OUT / "torgo_clean_replay.jsonl", clean_replay)
write_jsonl(OUT / "torgo_clean_eval.jsonl", clean_eval)

for name, rows in {"train": train, "dev": dev, "test": test}.items():
    hours = sum(r["duration"] for r in rows) / 3600
    speakers = sorted({r["speaker_id"] for r in rows})
    print(name, len(rows), round(hours, 2), speakers)
```

Before paying for H100 time, verify every WAV is readable 16 kHz mono audio with finite samples, hash the five JSONL manifests, and manually listen to all dev/test clips whose transcript or duration looks suspicious. Both people must reuse those exact hashes.

The first training run is **100% dysarthric TORGO rows**. Activate `torgo_clean_replay.jsonl` only if the tuned model fails the clean-speech regression gate, and even then cap replay at 10% of batches. The control corpus is protection against forgetting—not the main learning signal.

### Person A — Parakeet-TDT 0.6B v2 experiment

**Person A owns all Parakeet work:** NeMo baseline decoding, the TORGO fine-tuning job, checkpoint recovery, and Parakeet evaluation. Person A does not own Cohere. Both people consume the shared immutable `torgo_dys_train.jsonl`, `torgo_dys_dev.jsonl`, and `torgo_dys_test.jsonl` above.

#### A1. Decode the untouched Parakeet checkpoints

Use NeMo's `ASRModel.from_pretrained()` and `transcribe()` for both frozen checkpoints. Save every hypothesis, not merely the aggregate WER.

```python
import json
from pathlib import Path
from nemo.collections.asr.models import ASRModel

def read_jsonl(path):
    return [json.loads(line) for line in Path(path).open()]

def decode(model_name, manifest_path, output_path):
    rows = read_jsonl(manifest_path)
    model = ASRModel.from_pretrained(model_name=model_name)
    model.eval()
    hypotheses = model.transcribe(
        [r["audio_filepath"] for r in rows],
        batch_size=16,
        timestamps=True,
    )
    with Path(output_path).open("w") as f:
        for row, hypothesis in zip(rows, hypotheses):
            prediction = getattr(hypothesis, "text", str(hypothesis))
            f.write(json.dumps({**row, "prediction": prediction}) + "\n")

decode(
    "nvidia/parakeet-tdt-0.6b-v2",
    "/data/voicebridge/torgo_dys_dev.jsonl",
    "/results/baseline_parakeet.jsonl",
)
decode(
    "nvidia/parakeet-tdt-1.1b",
    "/data/voicebridge/torgo_dys_dev.jsonl",
    "/results/comparison_parakeet_1p1b.jsonl",
)
```

#### A2. Fine-tune Parakeet v2 with NeMo on Baseten

Use Baseten's [Qwen3 LoRA cookbook](https://github.com/basetenlabs/ml-cookbook/tree/main/examples/qwen3-8b-lora-dpo-trl/training) only for the `TrainingProject`, H100, cache, logging, and checkpoint pattern. Replace TRL entirely with NVIDIA NeMo and its TDT loss.

Recommended files:

```text
training/baseten/
  config.py                  # Baseten H100 Training Job
  run.sh                     # invokes NeMo fine-tuning
  evaluate.py                # identical before/after evaluator
  requirements.txt
serving/voicebridge_truss/
  config.yaml
  model/model.py             # loads winning .nemo checkpoint
```

Baseten configuration, using a NeMo image tag tested before the event:

```python
import os
from truss_train import definitions
from truss.base import truss_config

runtime = definitions.Runtime(
    start_commands=["/bin/bash ./run.sh"],
    environment_variables={
        "TRAIN_MANIFEST": "/data/voicebridge/torgo_dys_train.jsonl",
        "DEV_MANIFEST": "/data/voicebridge/torgo_dys_dev.jsonl",
    },
    cache_config=definitions.CacheConfig(enabled=True),
)

job = definitions.TrainingJob(
    image=definitions.Image(
        base_image=os.environ["PINNED_NEMO_IMAGE"]
    ),
    compute=definitions.Compute(
        accelerator=truss_config.AcceleratorSpec(
            accelerator=truss_config.Accelerator.H100,
            count=1,
        ),
        node_count=1,
    ),
    runtime=runtime,
)

training_project = definitions.TrainingProject(
    name="voicebridge-parakeet-v2-torgo", job=job
)
```

The core `run.sh` uses NeMo's official ASR fine-tuning entry point. Pin the NeMo revision and inspect the model config before the event because Hydra field names can change between releases.

```bash
#!/usr/bin/env bash
set -euo pipefail

python /opt/NeMo/examples/asr/speech_to_text_finetune.py \
  --config-path=/opt/NeMo/examples/asr/conf/asr_finetune \
  --config-name=speech_to_text_finetune \
  +init_from_pretrained_model=nvidia/parakeet-tdt-0.6b-v2 \
  model.train_ds.manifest_filepath="$TRAIN_MANIFEST" \
  model.validation_ds.manifest_filepath="$DEV_MANIFEST" \
  model.train_ds.batch_size=8 \
  model.validation_ds.batch_size=16 \
  model.train_ds.num_workers=8 \
  model.validation_ds.num_workers=8 \
  model.optim.name=adamw \
  model.optim.lr=1e-5 \
  model.optim.weight_decay=0.01 \
  trainer.accumulate_grad_batches=2 \
  trainer.max_steps=800 \
  trainer.val_check_interval=100 \
  trainer.devices=1 \
  trainer.precision=bf16-mixed \
  exp_manager.exp_dir="$BT_CHECKPOINT_DIR" \
  exp_manager.create_checkpoint_callback=true
```

These are starting values, not universal hyperparameters. First run 50–100 steps, record examples/second and peak memory, then set a limit corresponding to roughly three to five effective passes over the small TORGO training split; stop early when held-out-speaker WER stops improving. Cap audio at 30 seconds and use length bucketing to avoid rare TDT/RNNT batches causing large loss tensors and memory spikes.

Start with ordinary NeMo fine-tuning because it is the supported path. LoRA remains optional: promote it only after a local test proves that the adapter changes inference and survives save/reload. Do not change the English tokenizer for dysarthria adaptation.

Launch and monitor using the CLI version pinned with the project:

```bash
baseten train push --config config.py
baseten train logs --job-id "$JOB_ID" --tail
baseten train metrics --job-id "$JOB_ID"
```

Baseten synchronizes anything saved below `BT_CHECKPOINT_DIR`. Preserve the best `.nemo` checkpoint, optimizer state, run configuration, sampled-manifest hash, Git commit, and validation predictions. [Baseten Training Jobs](https://docs.baseten.co/training/overview)

#### A3. Decode the fine-tuned Parakeet and compare

Restore the winning `.nemo` checkpoint and call the same decoding function on the unchanged dev manifest:

```python
from nemo.collections.asr.models import ASRModel

model = ASRModel.restore_from("/checkpoints/best.nemo")
model.eval()
hypotheses = model.transcribe(dev_audio_paths, batch_size=16, timestamps=True)
```

Normalize references and predictions identically, calculate aggregate and per-speaker WER, and join rows by audio path before comparing:

```python
from collections import defaultdict
from jiwer import wer

def score(rows):
    by_speaker = defaultdict(lambda: {"ref": [], "hyp": []})
    for row in rows:
        bucket = by_speaker[row["speaker_id"]]
        bucket["ref"].append(row["text"])
        bucket["hyp"].append(row["prediction"])
    speaker_wer = {
        speaker: wer(values["ref"], values["hyp"])
        for speaker, values in by_speaker.items()
    }
    return {
        "micro_wer": wer(
            [r["text"] for r in rows],
            [r["prediction"] for r in rows],
        ),
        "held_out_speaker_wer": next(iter(speaker_wer.values()))
        if len(speaker_wer) == 1 else None,
        "per_speaker": speaker_wer,
    }
```

The final report must show:

```text
absolute WER reduction       = base WER - tuned WER
relative WER reduction       = (base WER - tuned WER) / base WER
critical-error reduction     = names/numbers/negation before vs after
clean-speech regression      = typical-speech WER before vs after
latency change               = identical serving configuration
dev result                   = model selection only
untouched test result        = final headline WER after selection is frozen
```

Keep data preparation outside the paid H100 window. **Before mounting TORGO audio on Baseten, confirm that academic non-profit use and third-party cloud processing are permitted for the team's event participation and account configuration.**

### Person B — Cohere Transcribe experiment

**Person B owns all Cohere work:** Hugging Face baseline decoding, PEFT/LoRA adaptation, checkpoint recovery, and Cohere evaluation. This is a separate experiment, not an extra task inside Person A's Parakeet run. Person B must use the exact same TORGO dysarthric train/dev/test manifests, normalization function, output columns, and scoring script as Person A.

The candidate is **[`CohereLabs/cohere-transcribe-03-2026`](https://huggingface.co/CohereLabs/cohere-transcribe-03-2026)**, a 2B Conformer encoder–Transformer decoder ASR model trained with supervised token cross-entropy. It is larger than Parakeet, but its strong open-ASR accuracy makes it a meaningful quality-oriented challenger. Its model files are gated, so Person B must accept the repository conditions and verify `HF_TOKEN` access before the hackathon clock starts.

#### B1. Decode frozen Cohere on the shared TORGO dev set

Use the official processor/model path and disable punctuation so normalization is comparable with Parakeet. Save one row per clip in the shared prediction schema.

```python
import json
import time
from pathlib import Path

import torch
from transformers import AutoProcessor, CohereAsrForConditionalGeneration
from transformers.audio_utils import load_audio

MODEL_ID = "CohereLabs/cohere-transcribe-03-2026"
processor = AutoProcessor.from_pretrained(MODEL_ID)
model = CohereAsrForConditionalGeneration.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.bfloat16,
    attn_implementation="sdpa",
).to("cuda").eval()

rows = [json.loads(line) for line in Path("/data/voicebridge/torgo_dys_dev.jsonl").open()]
with Path("/results/baseline_cohere.jsonl").open("w") as out:
    for row in rows:
        audio = load_audio(row["audio_filepath"], sampling_rate=16000)
        inputs = processor(
            audio,
            sampling_rate=16000,
            return_tensors="pt",
            language="en",
            punctuation=False,
        ).to(model.device, dtype=model.dtype)
        torch.cuda.synchronize()
        started = time.perf_counter()
        with torch.inference_mode():
            token_ids = model.generate(**inputs, max_new_tokens=256)
        torch.cuda.synchronize()
        decoded = processor.decode(token_ids, skip_special_tokens=True)
        prediction = decoded[0] if isinstance(decoded, list) else decoded
        out.write(json.dumps({
            **row,
            "prediction": prediction,
            "model_id": MODEL_ID,
            "latency_ms": 1000 * (time.perf_counter() - started),
        }) + "\n")
```

Warm the model before timing and record audio duration so the report includes real-time factor. Add a VAD/noise gate in the product path because the model card warns that silence can produce hallucinated text. Do not include model download or cold-start time in steady-state latency, but report cold start separately.

#### B2. Fine-tune Cohere with supervised LoRA on Baseten

This experiment uses standard audio-to-transcript cross-entropy—not DPO and not an LLM text objective. Start with rank-8 LoRA on the decoder attention and feed-forward projections; this keeps optimizer memory low enough for a single H100 while the acoustic encoder stays frozen. A [community Cohere Transcribe adapter](https://huggingface.co/surus-ai/cohere-transcribe-spanish-MegaASR) demonstrates that PEFT adapters can be attached to this model, but it is not an official recipe and must not be treated as validation of our English TORGO settings.

```python
from peft import LoraConfig, get_peft_model
from transformers import (
    AutoProcessor,
    CohereAsrForConditionalGeneration,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
)

MODEL_ID = "CohereLabs/cohere-transcribe-03-2026"
processor = AutoProcessor.from_pretrained(MODEL_ID)
model = CohereAsrForConditionalGeneration.from_pretrained(
    MODEL_ID,
    torch_dtype="auto",
    attn_implementation="sdpa",
)

# Confirm these suffixes against model.named_modules() for the pinned revision.
lora = LoraConfig(
    r=8,
    lora_alpha=16,
    lora_dropout=0.05,
    bias="none",
    task_type="SEQ_2_SEQ_LM",
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "fc1", "fc2"],
)
model = get_peft_model(model, lora)
# Module paths are revision-specific: freeze every encoder parameter, including
# any adapter inserted there, and assert that trainable LoRA weights remain.
for name, parameter in model.named_parameters():
    if ".encoder." in name:
        parameter.requires_grad = False
assert any(p.requires_grad for p in model.parameters())
model.print_trainable_parameters()

# torgo_train/torgo_dev load the SAME dysarthric JSONL files as Person A. The collator loads
# 16 kHz waveforms, calls processor(..., language="en", punctuation=False),
# tokenizes row["text"], pads the batch, and masks padded label IDs with -100.
args = Seq2SeqTrainingArguments(
    output_dir="/checkpoints/cohere_adapter",
    bf16=True,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    gradient_accumulation_steps=4,
    learning_rate=1e-4,
    max_steps=600,
    eval_strategy="steps",
    eval_steps=100,
    save_steps=100,
    save_total_limit=2,
    logging_steps=20,
    predict_with_generate=True,
    remove_unused_columns=False,
    report_to="tensorboard",
)
trainer = Seq2SeqTrainer(
    model=model,
    args=args,
    train_dataset=torgo_train,
    eval_dataset=torgo_dev,
    data_collator=speech_seq2seq_collator,
    processing_class=processor,
    compute_metrics=compute_wer,
)
trainer.train()
trainer.save_model("/checkpoints/cohere_adapter/best")
```

The collator is a required implementation, not a placeholder to ignore: unit-test one batch and one backward pass before launch. Print `model.named_modules()` and replace the example encoder-name predicate and LoRA suffixes with those verified for the pinned revision. Assert that no encoder parameter is trainable and that the remaining trainable-parameter count is non-zero and plausible. If the adapter fails to improve after the 200-step evaluation, unfreeze only the final encoder block or stop the run; do not spend the full budget blindly. Treat `attn_implementation="sdpa"` as an optimization flag to retain only if the smoke test passes.

Package this in its own Baseten `TrainingProject` named `voicebridge-cohere-torgo`, using a pinned CUDA/PyTorch image, one H100, cache-enabled model downloads, and `BT_CHECKPOINT_DIR` for adapter checkpoints. Reuse the Baseten job structure from the [Baseten LoRA cookbook](https://github.com/basetenlabs/ml-cookbook/tree/main/examples/qwen3-8b-lora-dpo-trl/training), but replace Qwen/TRL with the Transformers + PEFT script above:

```bash
baseten train push --config config_cohere.py
baseten train logs --job-id "$COHERE_JOB_ID" --tail
baseten train metrics --job-id "$COHERE_JOB_ID"
```

#### B3. Decode the adapted Cohere model and compare

Reload the base plus `/checkpoints/cohere_adapter/best`, run the unchanged B1 loop, and write `/results/tuned_cohere.jsonl`. For the final serving candidate, merge only after evaluation with `model.merge_and_unload()` and save a BF16 quality reference. Cohere is not assumed to have Parakeet-like streaming behavior: measure offline WER first, then measure interactive latency using the same rolling-window endpoint and stable-partial policy used for the product.

### Shared promotion gate — decide which model runs better

At hour 5, join all **development-speaker** files by `audio_filepath` and fill this table. TORGO provides only one validation speaker in the 6/1/1 split, so calling this “speaker-macro WER” would overstate the evidence. The primary selection metric is WER on that held-out speaker, with isolated-word and restricted-sentence subsets reported separately.

| Metric | Parakeet frozen | Parakeet TORGO-tuned | Cohere frozen | Cohere TORGO-LoRA |
|---|---:|---:|---:|---:|
| Held-out dev-speaker WER ↓ |  |  |  |  |
| Isolated-word WER ↓ |  |  |  |  |
| Restricted-sentence WER ↓ |  |  |  |  |
| Critical error rate ↓ |  |  |  |  |
| Typical-speech WER ↓ |  |  |  |  |
| p50 / p95 final latency ↓ |  |  |  |  |
| Real-time factor ↓ |  |  |  |  |

The candidate must introduce no new critical errors on the fixed safety slice and may regress typical-speech WER by at most 1.0 absolute point. Among candidates that pass, choose the lowest held-out dev-speaker WER. If the models are within 1.0 absolute WER point, choose the one with fewer critical errors; if still tied, choose lower p95 latency. Only the winner proceeds to FP8 and serving optimization. Preserve both baselines and adapters so the result remains reproducible even if the winner is Parakeet.

The test speaker is sealed during this decision. After the model and checkpoint are frozen, decode the winner and that model's corresponding frozen baseline on `torgo_dys_test.jsonl` exactly once. The headline improvement comes from this untouched test speaker, not the development table. A larger post-hackathon study should use leave-one-speaker-out cross-validation across all eight dysarthric speakers.

## Baseten inference engineering

This is where Baseten is central to the product rather than decorative infrastructure.

### Cache only state that avoids real recomputation

**Recommendation: build the cache interface early, but promote the cached path only after the winning BF16 checkpoint works.** Caching does not require retraining, so Person B can scaffold session state against a base checkpoint while the two training jobs run. The hour-6 milestone should retain a stateless or conservative fixed-chunk fallback. During hours 6–8, replay identical fixtures through cached and uncached paths, measure saved computation, and reject the cached path if hypotheses drift or session state leaks between users.

- Maintain a ring buffer of new audio features instead of rebuilding a 20–30 second window.
- For a streaming encoder, cache completed convolution state and attention K/V for left-context chunks.
- For Whisper-style decoding, preserve decoder self-attention KV state across stable partial hypotheses and compute encoder cross-attention K/V once per stable chunk.
- When a partial transcript changes, invalidate only the unstable tail.
- Cache the reference speaker embedding and synthesis style controls once per session.
- Reuse the verifier's fixed system/schema prefix with stable Baseten session affinity when the chosen LLM runtime supports KV-aware routing.

Baseten documents [KV-aware routing](https://docs.baseten.co/engines/bis-llm/advanced-features) for supported causal LLM deployments. That does not automatically cache a custom speech encoder; speech state reuse must be implemented inside the Truss service.

Only a streaming-native encoder may reuse encoder state across chunks without changing model semantics. For a non-streaming Whisper checkpoint, do not bolt on arbitrary encoder KV reuse: use overlapping audio windows, reuse decoder KV for the stable prefix, and reuse cross-attention projections only for encoder frames declared stable. A true chunked-Conformer or streaming transducer is the safer choice if encoder caching is a headline requirement.

Neither experiment should claim true encoder KV reuse without proving that the chosen architecture and implementation expose valid chunk state. Parakeet-TDT v2 is extremely fast offline but its full-attention encoder is not a cache-aware streaming encoder; Cohere Transcribe's documented production path is offline/vLLM rather than a demonstrated cache-aware streaming encoder. The first deployment should therefore use conservative overlapping windows and WebSocket delivery of stable partial hypotheses. If repeated encoding prevents the winner from meeting its first-partial latency gate, the principled fallback is `nvidia/nemotron-speech-streaming-en-0.6b`, whose encoder explicitly supports cache-aware chunks and configurable lookahead.

### Quantize behind an accuracy gate

**Recommendation: train and verify in BF16 first; quantize the frozen winner afterward.** This is post-training quantization (PTQ), not custom FP8 training. Quantizing before fine-tuning provides no stable inference artifact because training changes the weights, while FP8-aware or quantization-aware training adds numerical and implementation risk that the 12-hour build does not need.

1. Keep training and the master checkpoint in BF16.
2. Save the winning Parakeet `.nemo` checkpoint or merged Cohere checkpoint as the BF16 quality reference.
3. Build an FP8 engine for the dense speech-model projections on H100.
4. Keep layer normalization, softmax, output logits, and any sensitive first/last blocks in BF16 if calibration regresses.
5. Calibrate on clips balanced by speaker, severity, duration, and SNR.
6. Accept FP8 only if held-out-speaker WER rises by at most 1.0 absolute point and it introduces no additional critical errors on the fixed safety slice.

Baseten's [quantization guide](https://docs.baseten.co/engines/performance-concepts/quantization-guide) describes FP8 serving on H100. Its causal-model `FP8_KV` setting is not a speech-encoder-cache feature and should not be presented as one.

Do not assume FP8 will improve interactive latency. Baseten notes that smaller models may see limited benefit. A 0.6B Parakeet or 2B Cohere model at batch size one may be limited by feature extraction, autoregressive decoding, memory movement, or request overhead rather than dense matrix multiplication. Measure p50/p95 time to first token and final latency. If FP8 only saves VRAM without a meaningful latency gain, ship BF16 and prioritize request-path overhead.

Only consider FP8 training or quantization-aware fine-tuning later if PTQ clearly improves speed but fails the accuracy gate. It is not part of the hackathon critical path.

### Remove serving overhead

- Use PyTorch scaled-dot-product attention or FlashAttention where supported.
- Bucket requests by audio length and cap dynamic-batching wait time.
- Preallocate input/state tensors and use pinned memory plus asynchronous host-to-device copies.
- Test `torch.compile` and CUDA Graph capture only after input shapes stabilize.
- Stream over a Baseten [WebSocket endpoint](https://docs.baseten.co/development/model/websockets) so partial text and audio do not wait for a full utterance.
- Keep the LLM verifier on a hosted Model API; reserve the custom H100 service for speech recognition and voice rendering.

### Latency scorecard

| Variant | Held-out WER ↓ | Critical errors ↓ | First partial text ↓ | First recovered audio ↓ | p95 final latency ↓ | Peak VRAM ↓ |
|---|---:|---:|---:|---:|---:|---:|
| Base BF16 ASR |  |  |  |  |  |  |
| + TORGO fine-tuning |  |  |  |  |  |  |
| + streaming state reuse |  |  |  |  |  |  |
| + FP8 |  |  |  |  |  |  |
| Full ASR + personal voice |  |  |  |  |  |  |

Populate this table from saved JSON and traces. The Baseten story is the measured quality–latency improvement, not the number of platform features mentioned.

## Twelve-hour implementation plan

This is a **12-hour elapsed plan for two people**. It deliberately spends the first five hours on two independent, comparable experiments. The gate is non-negotiable: **by hour 6, a user must be able to speak, receive recovered text, and hear it in a selected personal voice through the winning Baseten-hosted BF16 path.** After that, both people work on the winner.

### Two-person ownership

- **Person A — Parakeet owner:** validates the shared manifests, runs frozen Parakeet v2/1.1B, launches NeMo fine-tuning, evaluates the tuned checkpoint, and prepares the Parakeet serving adapter.
- **Person B — Cohere owner:** independently verifies the same manifests, runs frozen Cohere, launches the supervised PEFT/LoRA job, evaluates the adapter, and prepares the Cohere serving adapter. While jobs run, Person B also owns the shared UI, verifier, OpenVoice enrollment, and text-to-personal-voice stream.
- **Shared after hour 5:** apply the promotion gate, deploy only the winner, then divide into serving optimization versus product verification. Neither person changes the dev set or evaluator after seeing results.

### Clock plan

| Hackathon time | Person A — Parakeet | Person B — Cohere / product | Joint exit condition |
|---:|---|---|---|
| **0:00–1:00** | Validate the TORGO dysarthric manifests; produce frozen v2 and 1.1B fixture predictions; run one NeMo backward-pass smoke test. | Independently validate the same manifest hashes; produce frozen Cohere fixture predictions; run one PEFT backward-pass smoke test; enroll one consented OpenVoice identity and synthesize a fixed sentence. | Identical data hashes, valid prediction schemas, trainable checkpoints, a shared evaluator, and audible cloned-voice output. |
| **1:00–4:00** | Run one Parakeet v2 NeMo job on H100-A, checkpointing every 200–250 steps. | Run one Cohere supervised LoRA job on H100-B; while it trains, connect the base endpoint to confirmation, stable-clause segmentation, and OpenVoice output. | `tuned_parakeet.jsonl` and `tuned_cohere.jsonl` candidates plus a base audio-in/audio-out shell. |
| **4:00–5:00** | Decode tuned Parakeet on the entire fixed dev set and populate its scorecard columns. | Decode tuned Cohere on the same dev set and populate its columns. | Promotion table filled; winner selected by the predeclared gate, not demo preference. |
| **5:00–6:00** | If Parakeet wins, deploy it; otherwise help wire the Cohere adapter and profile the common endpoint. | If Cohere wins, merge/deploy it; otherwise finish UI, verifier, OpenVoice integration, and streamed audio playback. | **Working product by hour 6:** speech → recovered text → confirmation → audible personal voice. |
| **6:00–8:00** | Build and benchmark FP8 for the frozen winner against BF16. | First decode the untouched test speaker once with the frozen baseline and selected BF16 checkpoint, then implement conservative rolling-window/session state, WebSocket partials, and latency traces. | Final held-out test result plus one accuracy-approved serving configuration; BF16 remains the fallback. |
| **8:00–9:00** | Test noise robustness and clean-speech regression without reopening model selection. | Test critical terms, cache agreement, voice identity, and end-to-end p50/p95 latency. | Frozen dysarthria deployment, populated scorecard, and documented failures. |
| **9:00–11:00** | Run stratified failure analysis across severity, utterance length, prompt type, and speaker; fix only reproducible serving defects. | Conduct blinded intelligibility/voice-identity checks, harden clarification UX, and capture the before/after evidence. | Final quality tables, representative successes and failures, and a stable dysarthria-only demo path. |
| **11:00–12:00** | Freeze model/checkpoint IDs and capture before/after evidence. | Freeze URLs, rehearse the story, and prepare recorded fallback inputs. | Reliable demo and evidence bundle. |

### Aggregate H100 budget

The plan caps experimental compute at **12 aggregate H100-hours**. The one-time test decode runs through the already-budgeted winner endpoint rather than starting another worker. Endpoint idle time is not part of this estimate; shut down unused workers promptly.

| GPU use | Spend |
|---|---:|
| Parakeet training on H100-A, hours 1–4 | 3 H100-hours |
| Cohere LoRA training on H100-B, hours 1–4 | 3 H100-hours |
| Full-dev evaluation on two H100s, hours 4–5 | 2 H100-hours |
| Winner BF16 endpoint plus FP8/serving benchmark worker, hours 5–7 | 4 H100-hours |
| **Total** | **12 H100-hours** |

Only two GPUs are required concurrently. A third GPU is unnecessary unless Baseten startup delays force a short replacement job, and using it would exceed the stated budget unless another worker is stopped.

### One-H100 backup

With one H100, preserve the two-model comparison and use the full dysarthric TORGO training split, but shorten the two sequential step budgets. The hour-6 path may use the better frozen model if neither adapter has cleared evaluation yet.

| Hackathon time | Person A | Person B | Single-GPU queue |
|---:|---|---|---|
| 0:00–1:00 | Parakeet smoke test and shared evaluator | Cohere smoke test plus OpenVoice enrollment/output fixture | Sequential frozen fixture decode |
| 1:00–3:00 | Monitor Parakeet; prepare its endpoint | Prepare Cohere dataset/collator | Parakeet NeMo adaptation |
| 3:00–5:00 | Evaluate Parakeet and support integration | Monitor Cohere LoRA | Cohere PEFT adaptation |
| 5:00–6:00 | Joint score and winner deployment | Joint score and product wiring | Winner decode/deploy; fall back to best frozen model if late |
| 6:00–8:00 | FP8 or compile benchmark | Streaming/UI instrumentation | One optimization at a time |
| 8:00–9:00 | Dysarthria quality verification | Safety/latency/voice verification | Final dysarthria test |
| 9:00–11:00 | Stratified error analysis and deployment fixes | Voice-identity checks, clarification UX, and evidence capture | Winner inference and reproducible test fixtures |
| 11:00–12:00 | Freeze evidence | Rehearse and capture fallback | Demo endpoint only |

If FP8 misses its accuracy gate, serve BF16. If cache invalidation is unstable, use conservative rolling windows rather than risking corrupted partial hypotheses. A reliable BF16 path always remains deployable.

## Evaluation

### Recognition and safety

- held-out-speaker WER/CER, reported separately for isolated words and restricted sentences;
- untouched test-speaker WER for the frozen winner versus its corresponding frozen baseline;
- critical error rate for negation, yes/no, names, medication terms, and numbers;
- expected calibration error and risk–coverage curve;
- clarification rate and successful recovery after clarification;
- unseen speakers, severity, quiet/noisy speech, fatigue, and long utterances.

### Real-time systems

- time to first partial transcript;
- time to first recovered audio;
- p50/p95 final latency and real-time factor;
- GPU milliseconds per utterance and peak VRAM;
- encoder/decoder state reuse and percentage of audio frames not recomputed;
- BF16 versus FP8 accuracy and latency.

### Voice identity and personality

- speaker-embedding cosine similarity between reference and recovered voice;
- F0 contour correlation and energy-envelope correlation;
- speaking-rate and pause-duration error;
- accent and identity preference from blinded listeners;
- intelligibility and naturalness ratings;
- user preference on the “more like me ↔ more clear” control.

Speaker-embedding similarity alone is insufficient. A voice can score as the same speaker while losing accent, emotion, rhythm, or perceived personality.

## Demo narrative

1. Play a permission-cleared excerpt from Jim to establish the human problem.
2. Speak a fixed dysarthric test utterance into the base ASR and show its error.
3. Reveal the Parakeet-versus-Cohere scorecard, switch to the winning TORGO-adapted endpoint, and show corrected partial text arriving incrementally.
4. Use an ambiguous example to show a two-choice clarification instead of a hallucination.
5. Confirm the sentence and play it in the enrolled personal voice, then compare it with a generic TTS voice.
6. Switch between BF16/stateless and FP8/cached paths while displaying first-partial latency, first-audio latency, macro WER, and critical errors.
7. Finish with the quantitative Parakeet-versus-Cohere result and one honest failure case showing when VoiceBridge asks for clarification.

The memorable line is:

> **VoiceBridge does not replace someone's voice. It repairs the path between what they meant and how the world hears them.**

## Go / no-go recommendation

**Go** if the team has a legally usable dysarthric corpus, a strong pretrained ASR checkpoint, and a reference-conditioned synthesizer that can be integrated without training from scratch.

The winning submission should make three claims and measure all three:

1. a controlled TORGO experiment identifies whether adapted Parakeet or adapted Cohere better recognizes unseen-speaker dysarthric read speech;
2. measured inference optimization makes the winning model conversationally fast without violating its accuracy gate;
3. the recovered output preserves the speaker's identity and expressive style better than generic TTS.
