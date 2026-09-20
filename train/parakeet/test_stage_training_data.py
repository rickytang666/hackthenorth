import tempfile
import unittest
from pathlib import Path
from unittest import mock

from train.parakeet.stage_training_data import stage


class StageTrainingDataTest(unittest.TestCase):
    def test_stages_only_dysarthric_phase1_audio(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "safe" / "data"
            receipt = stage(output)
            staged_root = output / "voicebridge"

            self.assertEqual(receipt["training_rows"], 4109)
            self.assertEqual(receipt["validation_rows"], 244)
            self.assertEqual(receipt["baseline_rows"], 10)
            self.assertEqual(receipt["audio_files"], 4363)
            self.assertEqual(receipt["over_30_second_training_rows"], 2)
            self.assertLess(receipt["audio_bytes"], 500_000_000)
            self.assertEqual(len(list((staged_root / "torgo_wav").glob("*.wav"))), 4363)
            self.assertFalse(
                any(
                    path.name.startswith(("FC", "MC", "M02"))
                    for path in (staged_root / "torgo_wav").glob("*.wav")
                )
            )
            self.assertFalse(
                (staged_root / "manifests" / "torgo_dys_test.jsonl").exists()
            )
            self.assertFalse(
                (staged_root / "manifests" / "torgo_clean_eval.jsonl").exists()
            )

    def test_refuses_broad_staging_path(self):
        with mock.patch("shutil.rmtree") as remove:
            with self.assertRaises(ValueError):
                stage(Path("/tmp"))
            remove.assert_not_called()


if __name__ == "__main__":
    unittest.main()
