"""Add a video clip to the demo selector.

    uv run python demo/add_video.py --video ~/Downloads/dysarthria.mp4 \
        --label "Stroke survivor, in the wild" \
        --truth "what he actually says" --start 4.0 --end 12.5

Writes two files into app/clips/ and appends an entry to its manifest: the
video for the viewer, and a 16 kHz mono WAV holding exactly the same span,
which is what gets streamed to the ASR. Splitting them keeps the wire format
identical to every other clip, so a video source is not a second code path.

Both outputs are gitignored. Re-run this to rebuild them.
"""

import argparse
import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "app" / "clips"
MANIFEST = OUT / "manifest.json"
# serve/asr_cohere decodes a trailing window of this many seconds, so anything
# longer is silently truncated from the front.
WINDOW_SECONDS = 12.0


def run(*args: str) -> None:
    subprocess.run(args, check=True, capture_output=True)


def duration(path: Path) -> float:
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "default=nw=1:nk=1", str(path)],
        check=True, capture_output=True, text=True)
    return float(out.stdout.strip())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True, type=Path)
    ap.add_argument("--label", required=True, help="shown under the selector")
    ap.add_argument("--truth", default="", help="what is actually said; omit if unknown")
    ap.add_argument("--baseline", default="", help="what the frozen model heard, if measured")
    ap.add_argument("--start", type=float, default=0.0)
    ap.add_argument("--end", type=float, help="defaults to the end of the file")
    ap.add_argument("--id", default=None, help="defaults to the video's filename stem")
    args = ap.parse_args()

    src = args.video.expanduser().resolve()
    if not src.exists():
        raise SystemExit(f"no such video: {src}")
    uid = args.id or src.stem
    end = args.end if args.end is not None else duration(src)
    span = round(end - args.start, 2)
    if span <= 0:
        raise SystemExit(f"empty span: start {args.start} to end {end}")

    trim = ["-ss", str(args.start), "-t", str(span)]
    video_out, wav_out = OUT / f"{uid}.mp4", OUT / f"{uid}.wav"
    # Re-encoded rather than stream-copied: a keyframe-aligned copy drifts from
    # the WAV, and the two have to start on the same sample.
    run("ffmpeg", "-v", "error", "-y", *trim, "-i", str(src),
        "-c:v", "libx264", "-preset", "veryfast", "-c:a", "aac", str(video_out))
    run("ffmpeg", "-v", "error", "-y", *trim, "-i", str(src),
        "-ac", "1", "-ar", "16000", "-sample_fmt", "s16", str(wav_out))

    entry = {
        "utterance_id": uid,
        "wav": f"clips/{uid}.wav",
        "video": f"clips/{uid}.mp4",
        "label": args.label,
        "speaker_id": "live",
        "truth": args.truth,
        "baseline": args.baseline,
        "duration_s": span,
    }
    rows = json.loads(MANIFEST.read_text())
    rows = [r for r in rows if r["utterance_id"] != uid] + [entry]
    MANIFEST.write_text(json.dumps(rows, indent=2) + "\n")

    print(f"{uid}: {span}s -> {video_out.name} + {wav_out.name}")
    if span > WINDOW_SECONDS:
        print(f"  WARNING: longer than the {WINDOW_SECONDS}s decode window. The final "
              f"transcript will cover only the last {WINDOW_SECONDS}s. Trim with "
              f"--start/--end to a single utterance.")
    if not args.truth:
        print("  no --truth given, so the ground truth line stays hidden for this clip")


if __name__ == "__main__":
    main()
