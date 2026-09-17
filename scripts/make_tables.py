"""Regenerate the camera-ready tables from the stored result files.

Every number in Table 2 of the paper is computed here from
results/*_results.json rather than transcribed. Running this is the fastest
way to confirm that the reported values follow from the runs that produced
them.

    python scripts/make_tables.py
    python scripts/make_tables.py --format latex
"""

import argparse
import json
from pathlib import Path

import numpy as np

RESULTS_DIR = Path(__file__).resolve().parents[1] / 'results'

# Table 2 of the camera-ready, for cross-checking the regenerated values.
PUBLISHED = {
    'sleep_edf': {'accuracy': (0.698, 0.013), 'f1_macro': (0.612, 0.019),
                  'kappa': (0.596, 0.018), 'boundary_std': (0.026, 0.003)},
    'chb_mit': {'accuracy': (0.901, 0.043), 'auroc': (0.924, 0.014),
                'kappa': (0.356, 0.132), 'sensitivity': (0.796, 0.047),
                'boundary_std': (0.014, 0.006)},
}


def aggregate(results, key):
    vals = np.array([r[key] for r in results if key in r], dtype=float)
    if len(vals) == 0:
        return None
    std = float(vals.std(ddof=1)) if len(vals) > 1 else 0.0
    return float(vals.mean()), std


def load(tag):
    path = RESULTS_DIR / f'{tag}_results.json'
    if not path.exists():
        return None
    return json.loads(path.read_text())


def fmt(stat, dp=3, latex=False):
    if stat is None:
        return '--'
    mean, std = stat
    if latex:
        return f"{mean:.{dp}f} $\\pm$ {std:.{dp}f}"
    return f"{mean:.{dp}f} +/- {std:.{dp}f}"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--format', choices=['text', 'latex'], default='text')
    parser.add_argument('--check', action='store_true',
                        help='compare against the published table and exit '
                             'non-zero on any mismatch')
    args = parser.parse_args()
    latex = args.format == 'latex'

    rows = [
        ('Sleep-EDF', 'sleep_edf',
         ['accuracy', 'f1_macro', 'kappa', 'boundary_std']),
        ('CHB-MIT', 'chb_mit',
         ['accuracy', 'auroc', 'kappa', 'sensitivity', 'boundary_std']),
    ]

    print("Table 2: cross-task validation, five training seeds, "
          "fixed split\n")
    mismatches = []
    for label, tag, keys in rows:
        results = load(tag)
        if results is None:
            print(f"{label}: no results file, skipping")
            continue
        cells = []
        for key in keys:
            stat = aggregate(results, key)
            cells.append(fmt(stat, latex=latex))
            if stat is not None and key in PUBLISHED.get(tag, {}):
                pm, ps = PUBLISHED[tag][key]
                # The paper prints three decimals, so the published value is
                # matched against the rounded statistic rather than by an
                # absolute tolerance.
                if (round(stat[0], 3) != pm or round(stat[1], 3) != ps):
                    mismatches.append(
                        f"{label} {key}: computed "
                        f"{stat[0]:.3f}+/-{stat[1]:.3f}, "
                        f"published {pm:.3f}+/-{ps:.3f}")
        n = len(results)
        if latex:
            print(f"{label} & " + ' & '.join(cells) + " \\\\")
        else:
            print(f"{label} ({n} seeds): " + '  '.join(
                f"{k}={c}" for k, c in zip(keys, cells)))

    tuab = RESULTS_DIR / 'tuab_eval_results.json'
    if tuab.exists():
        t = json.loads(tuab.read_text())
        print(f"\nTUAB (single seed): acc={t.get('acc', float('nan')):.3f} "
              f"auroc={t.get('auroc', float('nan')):.3f} "
              f"kappa={t.get('kappa', float('nan')):.3f}")
        print("  TUAB is a specificity control, not a performance result; "
              "see docs/reproducibility.md")

    if mismatches:
        print("\nMismatches against the published table:")
        for m in mismatches:
            print(f"  {m}")
        if args.check:
            raise SystemExit(1)
    elif args.check:
        print("\nAll regenerated values match the published table.")


if __name__ == '__main__':
    main()
