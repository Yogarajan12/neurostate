# Results

The files behind every number reported in the paper. `scripts/make_tables.py`
regenerates the published tables from them, and
`tests/test_tables.py` checks the regenerated values against the camera-ready.

Contents:

| File | Contents |
|---|---|
| `sleep_edf_results.json` | Per-seed Sleep-EDF test metrics, five seeds |
| `chb_mit_results.json` | Per-seed CHB-MIT test metrics, five seeds |
| `tuab_eval_results.json` | TUAB test metrics, single seed |
| `chb_mit_conditional.json` | Per-seed conditional boundary analysis |
| `chb_mit_centre_surround.json` | Per-seed centre-versus-surround control |
| `null_distribution.json` | Shuffled-input null for the 0.020 threshold |
| `sleep_edf_contiguity.json` | Boundary statistics by window contiguity |
| `onset_eval_results.json` | CHB-MIT onset evaluation and PELT baseline |
| `sleep_edf_trace_seed42.json` to `seed46.json` | Per-epoch training traces, five seeds |

All nine values in Table 2 of the camera-ready regenerate exactly from these
files, as do the null distribution, the contiguity comparison, the CHB-MIT
conditional analysis, the centre-versus-surround control and the onset
latencies. The single exception is the TUAB boundary statistic: the paper
reports the pooled value (0.003) while `tuab_eval_results.json` records it
per class (0.0023 abnormal, 0.0027 normal), so that one cell is excluded
from the provenance check.

The trace files label the middle phase `form`, whereas `PhaseSchedule` emits
`formation`. `scripts/make_figures.py` accepts either.

Each per-seed file is a JSON list with one object per seed, carrying at
minimum `seed`, `accuracy`, `kappa`, `boundary_std`, `best_epoch` and, for
binary tasks, `auroc` and `sensitivity`.

Fresh runs write to `results/runs/`, which is gitignored. The files above are
the published ones and should not be overwritten by a re-run.
