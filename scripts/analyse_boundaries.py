"""Run the boundary analyses that constrain the interpretability claims.

    python scripts/analyse_boundaries.py --config configs/chb_mit.yaml \
        --checkpoint results/runs/checkpoints/chb_mit_seed42.pt \
        --analyses conditional centre_surround

Each analysis is described in src/neurostate/analysis.py, including what the
published result was and how to read it.
"""

import argparse
import json
from pathlib import Path

import torch

from neurostate.analysis import (
    boundary_by_contiguity,
    boundary_by_window_type,
    centre_vs_surround,
    contiguity_mask,
    null_boundary_std,
)
from neurostate.config import load_config
from neurostate.train import build_dataloaders, build_model

CHOICES = ['null', 'conditional', 'contiguity', 'centre_surround']


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--analyses', nargs='+', choices=CHOICES,
                        default=CHOICES)
    parser.add_argument('--out', default=None)
    parser.add_argument('--device',
                        default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()

    cfg = load_config(args.config)
    _, _, test_ld = build_dataloaders(cfg)
    model = build_model(cfg, device=args.device)
    model.load_state_dict(torch.load(args.checkpoint, map_location='cpu'))

    out = {}
    if 'null' in args.analyses:
        out['null'] = null_boundary_std(model, test_ld, device=args.device)
        print(f"null mean {out['null']['null_mean']:.4f}, "
              f"95th percentile {out['null']['null_p95']:.4f}")
    if 'conditional' in args.analyses:
        out['conditional'] = boundary_by_window_type(model, test_ld,
                                                     device=args.device)
        c = out['conditional']
        nan = float('nan')
        print(f"transition n={c['n_transition']} "
              f"window std="
              f"{c.get('mean_window_std_transition', nan):.4f} "
              f"| flat n={c['n_flat']} "
              f"window std={c.get('mean_window_std_flat', nan):.4f} "
              f"| p={c.get('p_value', nan):.4g}")
    if 'contiguity' in args.analyses:
        if cfg['loader'] != 'sleep_edf':
            print("contiguity needs a times array; only Sleep-EDF stores one")
        else:
            mask = contiguity_mask(test_ld.dataset, cfg['h5_path'])
            print(f"contiguous windows: {int(mask.sum())} of {len(mask)} "
                  f"({mask.mean():.1%})")
            out['contiguity'] = boundary_by_contiguity(model, test_ld, mask,
                                                       device=args.device)
            for name, stats in out['contiguity'].items():
                print(f"  {name}: n={stats['n']} "
                      f"pooled_std={stats['pooled_std']:.4f}")
    if 'centre_surround' in args.analyses:
        out['centre_surround'] = centre_vs_surround(model, test_ld,
                                                    device=args.device)
        for key, stats in out['centre_surround'].items():
            if key.startswith('class_'):
                print(f"  {key}: n={stats['n']} "
                      f"positive={stats['frac_positive']:.1%} "
                      f"delta={stats['mean_delta']:.4f}")

    dest = Path(args.out or f"results/{cfg['name']}_analysis.json")
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(out, indent=2))
    print(f"\nSaved to {dest}")


if __name__ == '__main__':
    main()
