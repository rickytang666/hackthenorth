"""Build the local-only VoiceBridge demo fallback bundle."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import wave
from pathlib import Path


DEFAULT_OUTPUT = Path(__file__).parent / "output"
SYNTHESIS_TEXT = "I need my medication before dinner tonight"


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def wav_details(path: Path) -> dict[str, int | float]:
    with wave.open(str(path), "rb") as audio:
        frames = audio.getnframes()
        sample_rate = audio.getframerate()
        return {
            "channels": audio.getnchannels(),
            "sample_rate_hz": sample_rate,
            "sample_width_bytes": audio.getsampwidth(),
            "duration_seconds": round(frames / sample_rate, 3),
        }


def require_file(path: Path, label: str) -> Path:
    if not path.is_file():
        raise SystemExit(f"missing {label}: {path}")
    if path.stat().st_size == 0:
        raise SystemExit(f"empty {label}: {path}")
    return path


def copy_media(source: Path, destination: Path) -> dict[str, object]:
    shutil.copy2(source, destination)
    details: dict[str, object] = {
        "file": destination.name,
        "bytes": destination.stat().st_size,
        "sha256": file_sha256(destination),
    }
    details.update(wav_details(destination))
    return details


def build(media_root: Path, output_dir: Path) -> dict[str, object]:
    cue_path = require_file(media_root / "demo/clips/cue_card.json", "cue card")
    cues = json.loads(cue_path.read_text())
    cue = cues[0]

    sources = {
        "m02_source": require_file(media_root / "demo/clips" / cue["file"], "M02 source WAV"),
        "personal_voice": require_file(media_root / "serve/voice/out/timed.wav", "personal-voice WAV"),
        "generic_tts": require_file(media_root / "serve/voice/out/_base.wav", "generic-TTS WAV"),
    }
    output_dir.mkdir(parents=True, exist_ok=True)

    artifacts = {
        "m02_source": copy_media(sources["m02_source"], output_dir / "01_m02_source.wav"),
        "personal_voice": copy_media(sources["personal_voice"], output_dir / "04_personal_voice.wav"),
        "generic_tts": copy_media(sources["generic_tts"], output_dir / "05_generic_tts.wav"),
    }

    transcripts = {
        "frozen_baseline": str(cue["frozen_heard"]),
        "tuned": str(cue["tuned"]),
    }
    transcript_files = {
        "frozen_baseline": output_dir / "02_frozen_baseline.txt",
        "tuned": output_dir / "03_tuned_transcript.txt",
    }
    for key, path in transcript_files.items():
        path.write_text(f"{transcripts[key]}\n")
        artifacts[key] = {
            "file": path.name,
            "bytes": path.stat().st_size,
            "sha256": file_sha256(path),
        }

    personal_duration = float(artifacts["personal_voice"]["duration_seconds"])
    generic_duration = float(artifacts["generic_tts"]["duration_seconds"])
    if abs(personal_duration - generic_duration) > 0.25:
        raise SystemExit("personal and generic synthesis files do not appear to contain the same sentence")

    manifest = {
        "bundle": "VoiceBridge booth fallback kit",
        "offline": True,
        "m02_fixture": {
            "source_file": cue["file"],
            "said": cue["said"],
            "frozen_heard": cue["frozen_heard"],
            "tuned": cue["tuned"],
        },
        "voice_comparison_text": SYNTHESIS_TEXT,
        "artifacts": artifacts,
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--media-root",
        type=Path,
        default=Path(__file__).resolve().parents[2],
        help="repo root containing the local, gitignored WAV sources",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    manifest = build(args.media_root.resolve(), args.output_dir.resolve())
    print(f"built {len(manifest['artifacts'])} verified artifacts in {args.output_dir}")


if __name__ == "__main__":
    main()
