"""Phase 0 gate: time one 7-word synthesis end to end.

The decision this feeds: under roughly 800 ms, whole-clause synthesis fits
inside the 1.5 s first-audio budget and streaming TTS gets deleted from the
plan, removing the piece most likely to stutter live.

    uv run python time_synthesis.py --reference <a wav of the target voice>
"""

import argparse
import sys
import time
from pathlib import Path

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE / "vendor" / "OpenVoice"))

import torch  # noqa: E402
from melo.api import TTS  # noqa: E402
from openvoice.api import ToneColorConverter  # noqa: E402

# Deliberately not importing openvoice.se_extractor: it imports faster-whisper
# at module level purely to segment long reference audio. Enrollment clips are
# short and hand-picked, so extract_se on the clips directly is equivalent and
# avoids the faster-whisper 0.9 / av 10 build failure entirely.

CKPT = HERE / "checkpoints" / "openvoice_v2"
SENTENCE = "I need my medication before dinner tonight"  # exactly seven words


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference", required=True, nargs="+",
                        help="one or more wavs of the voice to clone")
    parser.add_argument("--text", default=SENTENCE)
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--out", default="out/timed.wav")
    args = parser.parse_args()

    words = len(args.text.split())
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device {device}, {words} words: {args.text!r}\n")

    began = time.perf_counter()
    converter = ToneColorConverter(str(CKPT / "converter" / "config.json"), device=device)
    converter.load_ckpt(str(CKPT / "converter" / "checkpoint.pth"))
    tts = TTS(language="EN", device=device)
    speaker_id = tts.hps.data.spk2id["EN-US"]
    source_se = torch.load(CKPT / "base_speakers" / "ses" / "en-us.pth", map_location=device)
    print(f"load: {time.perf_counter()-began:.1f}s (cold start, not steady state)")

    began = time.perf_counter()
    target_se = converter.extract_se(list(args.reference))
    print(f"enroll: {time.perf_counter()-began:.1f}s from {len(args.reference)} clip(s) "
          f"(once per session, cached)\n")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    base = "out/_base.wav"
    timings = []
    for i in range(args.runs):
        began = time.perf_counter()
        tts.tts_to_file(args.text, speaker_id, base, speed=1.0)
        mid = time.perf_counter()
        converter.convert(audio_src_path=base, src_se=source_se,
                          tgt_se=target_se, output_path=args.out)
        done = time.perf_counter()
        timings.append((1000*(mid-began), 1000*(done-mid), 1000*(done-began)))
        print(f"  run {i+1}: base TTS {timings[-1][0]:7.0f} ms | "
              f"tone convert {timings[-1][1]:7.0f} ms | total {timings[-1][2]:7.0f} ms")

    steady = sorted(t[2] for t in timings)[len(timings)//2]
    base = sorted(t[0] for t in timings)[len(timings)//2]
    conv = sorted(t[1] for t in timings)[len(timings)//2]
    print(f"\nmedian total: {steady:.0f} ms for {words} words "
          f"(base TTS {base:.0f} + tone convert {conv:.0f})")
    if steady < 800:
        print("VERDICT: whole-clause synthesis fits; DELETE streaming TTS from the plan")
    elif device == "cpu":
        print(f"VERDICT: {steady:.0f} ms on CPU is over the 800 ms bar, but the bar was "
              "set for the serving GPU. Re-run this on the H100 before deciding; "
              "tone conversion dominates and is the part that must come down.")
    else:
        print(f"VERDICT: {steady:.0f} ms on {device} exceeds the 800 ms bar; "
              "streaming TTS stays on the table")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
