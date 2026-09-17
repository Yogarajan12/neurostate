"""Configuration loading with per-dataset defaults.

Configs are YAML files under configs/. Defaults below apply to every dataset;
each YAML overrides only what genuinely differs, so the differences between
the three published recipes are visible in a few lines rather than buried in
three near-identical files.
"""

from pathlib import Path

import yaml

__all__ = ["DEFAULTS", "load_config"]

DEFAULTS = {
    'n_samples': 3000,
    'embed_dim': 128,
    'n_layers': 4,
    'dropout': 0.1,
    'n_intra': 4,
    'n_inter': 2,
    'n_cross': 2,
    'contrast_scales': [1, 4, 16],
    'cp_hidden': 64,
    'n_context_epochs': 3,
    'n_epochs': 50,
    'warmup_epochs': 3,
    'formation_epochs': 12,
    'patience': 12,
    'split_seed': 42,
    'acbl_weight': 0.3,
    'cls_weight': 1.0,
    'pseudo_weight': 0.3,
    'prior_weight': 0.5,
    'sigma': 5.0,
    'pseudo_temperature': 2.0,
    'pseudo_temp_min': 0.5,
    'pseudo_temp_anneal_epochs': 15,
    'attn_prior_weight': 0.5,
    'variance_weight': 1.0,
    'lr_other': 1.0e-4,
    'lr_boundary': 3.0e-4,
    'wd_other': 1.0e-4,
    'wd_boundary': 1.0e-5,
    'use_plateau_scheduler': True,
    'single_param_group': False,
    'class_weighted_loss': False,
    'loader': 'sleep_edf',
    'binary': False,
}


def load_config(path, overrides=None):
    """Load a YAML config on top of DEFAULTS, then apply CLI overrides."""
    cfg = dict(DEFAULTS)
    cfg.update(yaml.safe_load(Path(path).read_text()))
    if overrides:
        cfg.update(overrides)
    return cfg
