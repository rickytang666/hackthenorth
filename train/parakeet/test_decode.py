import tempfile
import unittest
from pathlib import Path

from train.parakeet.decode import (
    check_manifest_hash_scope,
    hypothesis_text,
    resolve_nemo_checkpoint,
)


class _Hypothesis:
    text = "hello world"


class DecodeHelpersTest(unittest.TestCase):
    def test_hypothesis_text_accepts_string_and_nemo_shape(self):
        self.assertEqual(hypothesis_text("hello"), "hello")
        self.assertEqual(hypothesis_text(_Hypothesis()), "hello world")

    def test_hypothesis_text_rejects_unknown_shape(self):
        with self.assertRaises(TypeError):
            hypothesis_text(object())

    def test_resolve_nemo_checkpoint_file_or_directory(self):
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "model.nemo"
            checkpoint.touch()
            self.assertEqual(resolve_nemo_checkpoint(checkpoint), checkpoint)
            self.assertEqual(resolve_nemo_checkpoint(directory), checkpoint)

    def test_resolve_nemo_checkpoint_rejects_ambiguous_directory(self):
        with tempfile.TemporaryDirectory() as directory:
            (Path(directory) / "a.nemo").touch()
            (Path(directory) / "b.nemo").touch()
            with self.assertRaises(ValueError):
                resolve_nemo_checkpoint(directory)

    def test_manifest_hash_scope_rejects_unknown_scope(self):
        with self.assertRaises(ValueError):
            check_manifest_hash_scope("contract/MANIFEST_HASHES", "unknown")


if __name__ == "__main__":
    unittest.main()
