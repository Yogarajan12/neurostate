"""Train NeuroState on one dataset across one or more seeds.

    python scripts/train.py --config configs/sleep_edf.yaml --seeds 42 43 44

This is a thin wrapper so the command works from a clone without installing.
The implementation lives in `neurostate.cli`, which also backs the
`neurostate-train` console script, so the two entry points cannot drift apart.
"""

from neurostate.cli import train_main

if __name__ == '__main__':
    train_main()
