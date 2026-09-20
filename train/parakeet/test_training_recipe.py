import unittest
from pathlib import Path


class AdapterTrainingRecipeTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.script = (
            Path(__file__).with_name("run_training.sh").read_text(encoding="utf-8")
        )

    def test_uses_parameter_efficient_encoder_adapter(self):
        self.assertIn("train_asr_adapter.py", self.script)
        self.assertIn("model.adapter.adapter_module_name=encoder", self.script)
        self.assertIn("model.adapter.adapter_type=linear", self.script)
        self.assertIn("model.adapter.linear.in_features=1024", self.script)
        self.assertIn("trainer.strategy=auto", self.script)

    def test_uses_requested_learning_rate_and_early_stopping(self):
        self.assertIn('ADAPTER_LR="${ADAPTER_LR:-3e-6}"', self.script)
        self.assertIn(
            'EARLY_STOPPING_PATIENCE="${EARLY_STOPPING_PATIENCE:-5}"', self.script
        )
        self.assertIn("create_early_stopping_callback=true", self.script)
        self.assertIn("early_stopping_callback_params.monitor=val_wer", self.script)
        self.assertIn("checkpoint_callback_params.save_top_k=1", self.script)
        self.assertIn("checkpoint_callback_params.save_last=false", self.script)
        self.assertIn("checkpoint_callback_params.save_best_model=false", self.script)
        self.assertIn("checkpoint_callback_params.save_nemo_on_train_end=false", self.script)

    def test_uses_manifest_text_as_transcript_field(self):
        self.assertIn("+model.train_ds.text_field=text", self.script)
        self.assertIn("+model.validation_ds.text_field=text", self.script)

    def test_caps_audio_duration_for_oom_safe_microbatches(self):
        self.assertIn(
            'TRAIN_BATCH_DURATION="${TRAIN_BATCH_DURATION:-60.0}"', self.script
        )
        self.assertIn(
            '"+model.train_ds.batch_duration=${TRAIN_BATCH_DURATION}"', self.script
        )
        self.assertIn('GRAD_ACCUMULATION="${GRAD_ACCUMULATION:-4}"', self.script)
        self.assertIn("+model.train_ds.use_bucketing=false", self.script)


class AdapterContinuationRecipeTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        root = Path(__file__).parent
        cls.script = (root / "run_continuation.sh").read_text(encoding="utf-8")
        cls.python = (root / "continue_asr_adapter.py").read_text(encoding="utf-8")

    def test_loads_existing_adapter_without_unfreezing_base(self):
        self.assertIn("model.load_adapters", self.python)
        self.assertNotIn("model.add_adapter", self.python)
        self.assertIn("model.freeze()", self.python)
        self.assertIn("model.unfreeze_enabled_adapters()", self.python)

    def test_runs_explicit_thousand_step_gate(self):
        self.assertIn('ADDITIONAL_STEPS="${ADDITIONAL_STEPS:-1000}"', self.script)
        self.assertIn('START_TOTAL_STEPS="${START_TOTAL_STEPS:-2000}"', self.script)
        self.assertIn('CONTINUATION_LR="${CONTINUATION_LR:-3e-7}"', self.script)
        self.assertIn('"trainer.max_steps=${ADDITIONAL_STEPS}"', self.script)

    def test_preserves_best_validation_checkpoint(self):
        self.assertIn("checkpoint_callback_params.save_top_k=1", self.script)
        self.assertIn("early_stopping_callback_params.monitor=val_wer", self.script)
        self.assertIn("model.save_adapters(sys.argv[2])", self.script)


if __name__ == "__main__":
    unittest.main()
