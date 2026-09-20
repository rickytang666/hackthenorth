"""Decode a frozen or restored Parakeet checkpoint with the shared contract.

The default command decodes the frozen dev manifest. The Baseten baseline job
overrides it with Person A's derived 10-row smoke manifest; training must not be
launched until that file has been produced and scored.
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from contract import manifest
from contract.decode import decode_manifest
from train.parakeet import BASE_MODEL_ID


def hypothesis_text(hypothesis: Any) -> str:
    """Normalize NeMo's version-dependent string/Hypothesis return value."""
    if isinstance(hypothesis, str):
        return hypothesis
    text = getattr(hypothesis, "text", None)
    if not isinstance(text, str):
        raise TypeError(
            f"NeMo returned an unsupported hypothesis: {type(hypothesis)!r}"
        )
    return text


def resolve_nemo_checkpoint(path: str | Path) -> Path:
    """Resolve either a .nemo file or a directory containing exactly one."""
    candidate = Path(path)
    if candidate.is_file() and candidate.suffix == ".nemo":
        return candidate
    if candidate.is_dir():
        matches = sorted(candidate.rglob("*.nemo"))
        if len(matches) == 1:
            return matches[0]
        raise ValueError(
            f"expected exactly one .nemo under {candidate}, found {len(matches)}"
        )
    raise ValueError(f"checkpoint is not a .nemo file or directory: {candidate}")


def load_model(model_id: str, checkpoint: str | None):
    """Import NeMo lazily so contract-only tests do not require a CUDA stack."""
    from nemo.collections.asr.models import ASRModel

    if checkpoint:
        model = ASRModel.restore_from(str(resolve_nemo_checkpoint(checkpoint)))
    else:
        model = ASRModel.from_pretrained(model_name=model_id)
    model.eval()
    return model


def transcribe_batch(model, paths: Sequence[str], batch_size: int) -> list[str]:
    hypotheses = model.transcribe(
        audio=list(paths),
        batch_size=batch_size,
        return_hypotheses=True,
        timestamps=False,
        verbose=False,
    )
    return [hypothesis_text(hypothesis) for hypothesis in hypotheses]


def check_manifest_hash_scope(expected_path: str | Path, scope: str) -> None:
    """Verify every frozen manifest, or only train/dev in sealed-test jobs."""
    if scope == "all":
        manifest.check_hashes(expected_path)
        return
    if scope != "train-dev":
        raise ValueError(f"unsupported manifest hash scope: {scope}")

    expected = {}
    for line in Path(expected_path).read_text().splitlines():
        if line.strip() and not line.startswith("#"):
            digest, name = line.split()
            expected[name] = digest
    names = ("torgo_dys_train.jsonl", "torgo_dys_dev.jsonl")
    bad = {
        name: (expected.get(name), manifest.hash_manifest(name))
        for name in names
        if expected.get(name) != manifest.hash_manifest(name)
    }
    if bad:
        raise RuntimeError(f"manifest hash mismatch, do not decode: {bad}")


def run(args: argparse.Namespace) -> list[dict]:
    check_manifest_hash_scope(args.manifest_hashes, args.manifest_hash_scope)
    rows = manifest.load(args.manifest, resolve=True)
    if not rows:
        raise RuntimeError(f"manifest is empty: {args.manifest}")

    model = load_model(args.model_id, args.checkpoint)

    # Exclude model initialization and the first CUDA/kernel setup from the
    # steady-state latency stored in the prediction rows.
    transcribe_batch(model, [rows[0]["audio_filepath"]], batch_size=1)

    def transcribe(paths: list[str]) -> list[str]:
        return transcribe_batch(model, paths, batch_size=args.batch_size)

    output = decode_manifest(
        args.manifest,
        transcribe,
        model_id=args.model_id,
        out_path=args.output,
        batch_size=args.batch_size,
    )
    if len(output) != len(rows):
        raise RuntimeError(f"decoded {len(output)} rows but manifest has {len(rows)}")
    return output


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-id", default=BASE_MODEL_ID)
    parser.add_argument(
        "--checkpoint",
        default=None,
        help="Optional .nemo file or directory; omit for the untouched HF checkpoint.",
    )
    parser.add_argument("--manifest", default="torgo_dys_dev.jsonl")
    parser.add_argument("--output", default="results/baseline_parakeet.jsonl")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--manifest-hashes", default="contract/MANIFEST_HASHES")
    parser.add_argument(
        "--manifest-hash-scope",
        choices=("all", "train-dev"),
        default="all",
        help="Use train-dev only when clean/test manifests are deliberately sealed.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.batch_size < 1:
        raise SystemExit("--batch-size must be positive")
    rows = run(args)
    print(f"wrote {len(rows)} predictions to {args.output}")


if __name__ == "__main__":
    main()
