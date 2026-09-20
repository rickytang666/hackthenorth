"""Continue NeMo ASR adapter training from a compact ``save_adapters`` file.

The upstream NeMo adapter trainer can create a new adapter, but it does not
offer a CLI path for loading an existing adapter before ``trainer.fit``.  This
module intentionally stays close to that trainer and changes only the adapter
initialisation step: ``load_adapters`` replaces ``add_adapter``.
"""

import os

import lightning.pytorch as pl
from nemo.collections.asr.models import ASRModel
from nemo.core import adapter_mixins
from nemo.core.config import hydra_runner
from nemo.utils import logging
from nemo.utils.exp_manager import exp_manager
from nemo.utils.trainer_utils import resolve_trainer_cfg
from omegaconf import OmegaConf, open_dict


def update_model_config_to_support_adapter(model_cfg, current_cfg):
    """Swap the encoder class for NeMo's registered adapter-aware variant."""
    with open_dict(model_cfg):
        model_cfg.log_prediction = current_cfg.model.get("log_prediction", False)
        adapter_metadata = adapter_mixins.get_registered_adapter(
            model_cfg.encoder._target_
        )
        if adapter_metadata is not None:
            model_cfg.encoder._target_ = adapter_metadata.adapter_class_path


def update_model_cfg(original_cfg, new_cfg):
    """Apply dataset overrides using the same policy as NeMo's trainer."""
    with open_dict(original_cfg), open_dict(new_cfg):
        for key in ("num_workers", "pin_memory"):
            if key in new_cfg:
                original_cfg[key] = new_cfg[key]

        for key in list(new_cfg.keys()):
            if key not in original_cfg:
                new_cfg.pop(key)

        return OmegaConf.merge(original_cfg, new_cfg)


@hydra_runner(config_path="../conf/asr_adapters", config_name="asr_adaptation.yaml")
def main(cfg):
    logging.info("Continuation config:\n%s", OmegaConf.to_yaml(cfg))
    if cfg.model.nemo_model is None:
        raise ValueError("model.nemo_model is required for adapter continuation")

    trainer = pl.Trainer(**resolve_trainer_cfg(cfg.trainer))
    exp_log_dir = exp_manager(trainer, cfg.get("exp_manager", None))

    model_cfg = ASRModel.restore_from(cfg.model.nemo_model, return_config=True)
    update_model_config_to_support_adapter(model_cfg, cfg)
    model = ASRModel.restore_from(
        cfg.model.nemo_model, override_config_path=model_cfg, trainer=trainer
    )

    cfg.model.train_ds = update_model_cfg(model.cfg.train_ds, cfg.model.train_ds)
    model.setup_training_data(cfg.model.train_ds)
    cfg.model.validation_ds = update_model_cfg(
        model.cfg.validation_ds, cfg.model.validation_ds
    )
    model.setup_multiple_validation_data(cfg.model.validation_ds)
    model.setup_optimization(cfg.model.optim)

    if "spec_augment" in cfg.model:
        model.spec_augmentation = model.from_config_dict(cfg.model.spec_augment)
    else:
        model.spec_augmentation = None

    with open_dict(cfg.model.adapter):
        adapter_name = cfg.model.adapter.pop("adapter_name")
        adapter_module_name = cfg.model.adapter.pop("adapter_module_name", None)
        restore_state_path = cfg.model.adapter.pop("restore_state_path")

    if adapter_module_name is not None and ":" not in adapter_name:
        adapter_name = f"{adapter_module_name}:{adapter_name}"
    if not os.path.isfile(restore_state_path):
        raise FileNotFoundError(f"adapter state not found: {restore_state_path}")

    # The .pt contains both the adapter configuration and its trained weights.
    # The base model must not already contain an adapter with the same name.
    model.load_adapters(restore_state_path, map_location="cpu", strict=True)
    if not model.is_adapter_available():
        raise RuntimeError("load_adapters completed without installing an adapter")

    model.set_enabled_adapters(enabled=False)
    model.set_enabled_adapters(adapter_name, enabled=True)
    model.freeze()
    model = model.train()
    model.unfreeze_enabled_adapters()

    trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    frozen = sum(parameter.numel() for parameter in model.parameters() if not parameter.requires_grad)
    logging.info("Continuation parameter counts: trainable=%d frozen=%d", trainable, frozen)
    if trainable == 0 or frozen == 0:
        raise RuntimeError("expected a trainable adapter and a frozen base model")

    trainer.fit(model)

    # This is the terminal state, useful for diagnosis. The launcher separately
    # exports the selected best-validation .nemo back to a compact adapter file.
    state_path = os.path.join(exp_log_dir or os.getcwd(), "checkpoints")
    os.makedirs(state_path, exist_ok=True)
    model.save_adapters(os.path.join(state_path, "terminal_adapter.pt"))


if __name__ == "__main__":
    main()
