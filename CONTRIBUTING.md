# Contributing

Notes for anyone extending this repository or reproducing its results,
covering decisions that are not obvious from the code alone.

Issues and pull requests are welcome, particularly on the pending work listed
at the end of [`docs/reproducibility.md`](docs/reproducibility.md).

## What this repository is

Reference implementation for "NeuroState: Gradient-Isolated Boundary Learning
for Interpretable EEG Transformers" (IEEE MLSP 2026), by Yogarajan Sivakumar
and Hong Man.

The camera-ready paper is the source of truth. Where code and paper disagree,
the paper governs claims and the code governs behaviour, and the difference is
documented rather than silently resolved. Several such differences are
recorded in [`docs/method.md`](docs/method.md) and
[`docs/reproducibility.md`](docs/reproducibility.md).

## Four things not to change

**Checkpoint compatibility.** Module and parameter names in
`src/neurostate/model.py` match the notebooks that produced the published
results, so checkpoints from those runs load without remapping keys. Renaming
a module breaks that silently.
`tests/test_model.py::test_state_dict_keys_are_stable` guards the important
ones.

**Reported numbers.** Every value in the README and `docs/` comes from the
camera-ready or from a file in `results/`. A number should not be adjusted to
match a fresh run; where a fresh run disagrees, the disagreement is worth
investigating and documenting instead.

**Negative results.** The CHB-MIT conditional non-result, the window-geometry
confound and the Sleep-EDF contiguity result are load-bearing. They are why
the interpretability claims are scoped as preliminary.

**Scope.** Collapse and gradient isolation are established for the EEG
settings measured in the paper. What was shown to carry across the four
boundary parameterisations is the collapse, not the remedy, since two of them
produce no comparable boundary statistic. Language claiming general
architecture-independence would overstate the evidence, including in the
synthetic demo.

## Code conventions

Line length is 79, enforced by ruff. Imports are consolidated at the top of
each module. Comments explain why something is as it is, particularly where
behaviour looks like a bug but is deliberate, and describe what the code does
rather than addressing the reader. Prose and comments use British spelling;
American spelling stays in existing code identifiers such as `normalize` for
compatibility. Section headers are plain comments, with no banner rules,
cell markers or printed separators, and no emojis.

## Layout

```
src/neurostate/     model, losses, isolation, data, train, evaluate, analysis
configs/            one YAML per dataset; variants/ reproduce table rows
scripts/            train, analyse_boundaries, make_tables, make_figures
results/            result files behind every published number
notebooks/          the originals that produced the published results
demo/               synthetic CPU demo of the useful-when-flat property
docs/               method, collapse_experiments, data, results, reproducibility
tests/              mechanism, model, table-provenance
```

`neurostate.cli` backs both `scripts/train.py` and the `neurostate-train`
console script, so new CLI logic belongs there rather than in `scripts/`,
which is not an installed package.

## Testing

```bash
pytest -q                              # CPU, seconds
python scripts/make_tables.py --check  # regenerate tables, compare to paper
ruff check src scripts tests demo
```

`tests/test_mechanism.py` asserts the science rather than the plumbing: that a
flat boundary output degenerates the regime mask into distance decay, and that
gradient isolation blocks exactly one direction of gradient flow. These are
the claims the paper rests on, so a change that breaks them is a change to
investigate rather than a test to update.

## Things that look wrong but are not

- **TUAB uses `variance_weight: 0.0`.** The published TUAB run used three ACBL
  terms, not four. Setting it to 1.0 would not reproduce the numbers.
- **Sleep-EDF uses two optimiser parameter groups** with a higher boundary
  learning rate; CHB-MIT and TUAB use one. The paper's setup line reads as a
  single recipe but the runs differ.
- **The regime mask is rebuilt in every transformer block**, not computed
  once.
- **`round_robin_split` ignores any seed.** It is deterministic by design.
- **The consistency term sits outside the lambda-weighted ACBL objective.** It
  belongs to the boundary module.
- **The demo reproduces the cause of collapse, not collapse itself.** Tuning
  the synthetic setup until collapse appears would be fitting a demonstration
  to a desired conclusion.
- **The TUAB boundary statistic is excluded from the provenance check.** The
  paper reports a pooled value and the result file stores it per class. Every
  other reported number regenerates exactly.
- **Trace files label the middle phase `form`, not `formation`.** The figure
  script accepts both.

## Known issues in the shipped notebooks

- `boundary_spectral_analysis.ipynb` cannot resolve delta, theta, alpha or
  sigma at its 0.3 second segment length; those bands print as exact zeros and
  are unmeasured rather than null.
- Three interpretability notebooks load checkpoints with `strict=False` after
  renaming keys, so a failed load would pass silently. Prefer `strict=True`
  when rerunning them.
- `tcav.ipynb` prints concept-classifier accuracies outside the 0.84 to 0.89
  range the paper reports. The TCAV scores themselves match, and the paper's
  range stands.

## Data

All three datasets require registration or a data use agreement and none are
redistributed here; see [`docs/data.md`](docs/data.md). The TUH agreement
governs TUAB-derived artefacts, including model weights, so check its terms
before sharing any.
