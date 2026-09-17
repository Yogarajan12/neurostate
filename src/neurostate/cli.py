"""Console entry points for the installed package.

`scripts/train.py` remains the documented way to run training from a clone.
This module exists so the `neurostate-train` console script resolves after a
plain `pip install`, where the top-level `scripts/` directory is not on the
path.
"""

import argparse
import json
from pathlib import Path

import torch

from .config import load_config
from .train import aggregate, run_seeds

__all__ = ["train_main"]


def train_main():
    parser = argparse.ArgumentParser(
        description='Train NeuroState on one dataset across one or more '
                    'seeds.')
    parser.add_argument('--config', required=True)
    parser.add_argument('--seeds', type=int, nargs='+',
                        default=[42, 43, 44, 45, 46])
    parser.add_argument('--out-dir', default='results/runs')
    parser.add_argument('--device',
                        default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--epochs', type=int, default=None,
                        help='override n_epochs, useful for smoke tests')
    parser.add_argument('--verbose', action='store_true')
    args = parser.parse_args()

    overrides = {'n_epochs': args.epochs} if args.epochs else None
    cfg = load_config(args.config, overrides)
    results = run_seeds(cfg, seeds=tuple(args.seeds), device=args.device,
                        out_dir=args.out_dir, tag=cfg['name'],
                        verbose=args.verbose)
    agg = aggregate(results)
    print(f"\n{cfg['name']} over {len(results)} seeds "
          f"(split_seed {cfg['split_seed']} fixed)")
    for k in sorted(agg):
        print(f"  {k:22s} {agg[k]['mean']:.4f} +/- {agg[k]['std']:.4f}")
    out = Path(args.out_dir) / f"{cfg['name']}_aggregate.json"
    out.write_text(json.dumps(agg, indent=2))
    print(f"\nSaved aggregate to {out}")
