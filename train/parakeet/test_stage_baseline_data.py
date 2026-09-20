import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from train.parakeet import BASELINE_MANIFEST, TRAINING_MANIFEST, VALIDATION_MANIFEST
from train.parakeet.splits import phase1_split
from train.parakeet.stage_baseline_data import stage


class StageBaselineDataTest(unittest.TestCase):
    def test_phase1_split_is_disjoint_and_stratified(self):
        splits = phase1_split()
        self.assertEqual(len(splits.training), 4109)
        self.assertEqual(len(splits.baseline), 10)
        self.assertEqual(len(splits.validation), 244)
        self.assertEqual(
            sum(len(row["text"].split()) == 1 for row in splits.baseline), 5
        )
        self.assertEqual(
            {row["speaker_id"] for row in splits.baseline}, {"F01", "M01", "M04"}
        )
        self.assertEqual({row["speaker_id"] for row in splits.validation}, {"F04"})
        self.assertFalse(
            {row["utterance_group"] for row in splits.baseline}
            & {row["utterance_group"] for row in splits.training}
        )

    def test_stages_only_baseline_audio_and_all_manifests(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "safe" / "data"
            receipt = stage(output)
            staged_root = output / "voicebridge"

            self.assertEqual(receipt["baseline_rows"], 10)
            self.assertEqual(receipt["training_rows"], 4109)
            self.assertEqual(receipt["validation_rows"], 244)
            self.assertEqual(len(list((staged_root / "torgo_wav").glob("*.wav"))), 10)
            self.assertEqual(len(list((staged_root / "manifests").glob("*.jsonl"))), 8)
            self.assertTrue((staged_root / "manifests" / TRAINING_MANIFEST).is_file())
            self.assertTrue((staged_root / "manifests" / BASELINE_MANIFEST).is_file())
            self.assertTrue((staged_root / "manifests" / VALIDATION_MANIFEST).is_file())
            self.assertTrue((staged_root / "BASELINE_BUNDLE.json").is_file())

            staged = json.loads((staged_root / "BASELINE_BUNDLE.json").read_text())
            self.assertEqual(staged["audio_bytes"], receipt["audio_bytes"])

    def test_refuses_broad_staging_path(self):
        with mock.patch("shutil.rmtree") as remove:
            with self.assertRaises(ValueError):
                stage(Path("/tmp"))
            remove.assert_not_called()


if __name__ == "__main__":
    unittest.main()
