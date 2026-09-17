# Notebooks

The originals that produced the published results, preserved with their
outputs because the printed values are much of the provenance. Deepnote
metadata and progress-bar noise have been stripped; paths and code are
otherwise unchanged.

| Notebook | Purpose | Produces |
|---|---|---|
| `preprocessing_sleep_edf.ipynb` | Sleep-EDF download, filtering, epoching, PELT baseline | `sleep_edf_processed.h5` |
| `preprocessing_chbmit.ipynb` | CHB-MIT summary parsing, montage standardisation | `chbmit_combined_seizure_detection.h5` |
| `preprocessing_chbmit_legacy.ipynb` | The earlier CHB-MIT pipeline, with 50 per cent overlapping seizure windows | chb01, chb03, chb05 portions of the above |
| `preprocessing_tuab.ipynb` | TUAB channel alias mapping, chunked HDF5 | `tuab_*_processed.h5` |
| `multiseed_runs.ipynb` | The five-seed sweeps behind Table 2, and the CHB-MIT conditional analysis | `sleep_edf_results.json`, `chb_mit_results.json`, `chb_mit_conditional.json` |
| `collapse_sensitivity.ipynb` | Shuffled-input null, schedule sweep, contiguity analysis | `null_distribution.json`, `sleep_edf_contiguity.json` |
| `transition_density.ipynb` | Epoch adjacency and seam-step diagnostics on both datasets | the 33.5 per cent and 4.10x figures in `docs/reproducibility.md` |
| `onset_evaluation.ipynb` | Seizure onset latency and the PELT comparison | `onset_eval_results.json` |
| `tcav.ipynb` | Concept activation vectors on both datasets | the TCAV scores in `docs/results.md` |
| `centre_surround.ipynb` | The window-geometry control on both classes | `chb_mit_centre_surround.json` |
| `boundary_spectral_analysis.ipynb` | Boundary-conditioned spectral contrasts | the beta and gamma ratios in `docs/results.md` |
| `interpretability_figures.ipynb` | Boundary overlays, attention heatmaps, t-SNE | figures used in the paper and poster |
| `tuab_crosstask_validation.ipynb` | The TUAB specificity control run | `tuab_eval_results.json` |

`preprocessing_chbmit_legacy.ipynb` is included because it explains a real
property of the shipped data rather than as history: three subjects in the
combined file, including one test subject, were windowed with 50 per cent
overlap by this pipeline. See `docs/reproducibility.md`.

The model, losses and training loop in these notebooks are superseded by
`src/neurostate/`, which was ported from `multiseed_runs.ipynb` directly.
Where a notebook and the package differ, the package is the maintained
version. Class and parameter names were deliberately kept identical, so
checkpoints saved by these notebooks load into the package unchanged.

Exploratory notebooks from the eleven collapse experiments are not included;
what each tried and how it failed is written up in
`docs/collapse_experiments.md`.

## Two caveats in the interpretability notebooks

`boundary_spectral_analysis.ipynb` prints 0.0000 with p = 1.0000 for delta,
theta, alpha and sigma. Those bands were not measured: the 0.3 second token
segment gives Welch bins 3.33 Hz apart, each of those bands spans fewer than
two bins, and integrating over one bin returns zero. The beta and gamma
contrasts are unaffected.

`boundary_spectral_analysis.ipynb`, `tcav.ipynb` and
`interpretability_figures.ipynb` load checkpoints with
`load_state_dict(..., strict=False)` after renaming keys, which silently
tolerates a failed load. Set `strict=True` before trusting a rerun.

Both are documented in `docs/results.md` and constrain only the preliminary
interpretability claims, not Table 2 or the collapse result.
