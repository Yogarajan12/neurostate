# Results and their provenance

Every number in the paper, with the file it comes from. The point of this
page is that no reported value needs to be taken on trust: each one is either
in `results/` or regenerable by a script in `scripts/`.

Run `python scripts/make_tables.py --check` to regenerate Table 2 from the
stored result files and compare against the published values.

## Table 2: cross-task validation

Five training seeds (42 to 46), fixed split, mean plus or minus sample
standard deviation.

| Dataset | Acc | AUROC | kappa | Sens. | Bnd Std |
|---|---|---|---|---|---|
| Sleep-EDF | 0.698 +/- 0.013 | --- | 0.596 +/- 0.018 | --- | 0.026 +/- 0.003 |
| CHB-MIT | 0.901 +/- 0.043 | 0.924 +/- 0.014 | 0.356 +/- 0.132 | 0.796 +/- 0.047 | 0.014 +/- 0.006 |
| TUAB | 0.916 | 0.913 | 0.545 | 0.569 | 0.003 |

Source: `results/sleep_edf_results.json`, `results/chb_mit_results.json`,
`results/tuab_eval_results.json`.

Sleep-EDF also reports macro F1 0.612 +/- 0.019.

**Reading the CHB-MIT kappa.** 0.356 alongside AUROC 0.924 and sensitivity
0.796 reflects severe class imbalance: 3.2 per cent seizure prevalence
compresses chance-corrected agreement even under strong ranking performance.
The two metrics are not in conflict.

**Reading the TUAB row.** Single seed. The 0.003 boundary figure is the
pooled statistic reported in the paper. `tuab_eval_results.json` records the
statistic per class instead (0.0023 abnormal, 0.0027 normal), so that single
cell cannot be regenerated from the shipped file and is excluded from the
provenance check; the classification metrics in the row all reproduce
exactly. Either way the value sits far below the 0.020 threshold, which is
the point of the control. TUAB is a specificity control; see
`reproducibility.md` before quoting its classification numbers.

**Per-seed CHB-MIT variation is wide.** Accuracy ranges 0.859 to 0.965 and
kappa 0.246 to 0.576 across the five seeds, which is why the standard
deviations are large. AUROC is far more stable (0.901 to 0.937), which is the
metric to read on this task.

Supporting per-seed detail, all verified against `results/`: mean window std
0.0136 +/- 0.0054 on transition windows against 0.0137 +/- 0.0054 on flat
ones, and a mean peak of 0.3251 +/- 0.0232 that is identical between the two
populations to four decimals. Best epochs cluster at 15 to 20 on both
datasets, so no run was selected during the formation phase.

## Boundary threshold: the empirical null

The 0.020 degeneracy threshold is justified against a shuffled-input null
rather than asserted. `null_boundary_std` recomputes the pooled boundary
standard deviation after shuffling sample order within each window, which
destroys temporal structure while leaving the marginal distribution intact.

| Quantity | Value |
|---|---|
| Null mean | 0.0154 |
| Null 95th percentile | 0.0155 |
| Observed Sleep-EDF, five seeds | 0.026 +/- 0.003 |
| Per-seed ratio to the null | 1.47x to 1.94x |

Source: `results/null_distribution.json`. Reproduce with
`scripts/analyse_boundaries.py --analyses null`.

Three qualifications the paper attaches to this, each worth carrying:

The **0.020 cutoff is a descriptive label, not a test**. Two variants sit near
it, bimodal scheduling just below at 0.014 and cumulative-product masking just
above at 0.018, and the statistic alone does not classify either. Both are
treated as degenerate on the strength of the failure seen during training,
constant outputs and all-ones mask saturation respectively, together with
accuracies of 0.626 and 0.678 against 0.718 for gradient isolation.

The **null is measured for one model on one dataset** and is not transferred
to other variants or to other datasets. This is why CHB-MIT is assessed by the
conditional test below rather than by comparing its boundary magnitude
against this figure.

The margin justifies the threshold on Sleep-EDF. It does not by itself show
the head tracks physiology; read it alongside the contiguity result below.

## Schedule sensitivity

Single seed, three settings of (warmup, formation):

| Setting | Acc | kappa | Bnd Std |
|---|---|---|---|
| (3, 12), as reported | 0.706 | 0.610 | 0.0225 |
| (1, 12) | 0.711 | 0.609 | 0.0192 |
| (3, 16) | 0.690 | 0.587 | 0.0326 |

Accuracy ranges 0.690 to 0.711 and boundary std 0.019 to 0.033, all above the
null. The collapse outcome is not knife-edge in the schedule, though the
boundary magnitude is schedule-sensitive.

## The three constraining analyses

These are reported in the paper and are why the interpretability claims are
scoped as preliminary. All three are reproducible with
`scripts/analyse_boundaries.py`.

### CHB-MIT conditional analysis: null

If the head marked transitions, windows containing one would show higher
per-window boundary variation than windows without.

| Quantity | Transition windows | Flat windows |
|---|---|---|
| n (per seed) | 59 | 1,437 |
| Pooled std | 0.0139 +/- 0.0059 | 0.0141 +/- 0.0061 |
| Mean window std | 0.0136 +/- 0.0054 | 0.0137 +/- 0.0054 |
| Mean peak | 0.3251 +/- 0.0232 | 0.3251 +/- 0.0232 |

Mann-Whitney p >= 0.21 across all five seeds (max 0.9997), Cohen's
d = -0.25 +/- 0.20. Source: `results/chb_mit_conditional.json`.

The pooled statistic is also diluted on this task, since only 59 of 1,496 test
windows contain a transition and the rest are windows where a flat output is
correct.

### Sleep-EDF contiguity: no difference

| Window type | n | Pooled std | Mean window std |
|---|---|---|---|
| Contiguous in recording time | 160 | 0.0221 | 0.0214 |
| Containing a splice | 865 | 0.0226 | 0.0220 |

Source: `results/sleep_edf_contiguity.json`. Only 15.6 per cent of test
windows are contiguous. The head tracks junction position more than signal
content; this is the strongest single constraint on the interpretability
claims.

### CHB-MIT centre versus surround: window geometry

| Class | n | Positive margin | Mean delta |
|---|---|---|---|
| Seizure windows | 53 | 98.9 +/- 1.7 per cent | 0.0050 |
| Non-seizure windows | 1,443 | 99.9 +/- 0.2 per cent | larger |

Cohen's d = -0.82 +/- 0.27, the margin being larger for non-seizure windows.
The centre epoch is bracketed by two junctions while each flanking epoch
borders only one, so the ordering follows from window geometry rather than
seizure content.

An earlier version of this analysis ran on seizure windows only and reported
"positive in 53/53 windows", which read as evidence of a seizure effect.
Running both classes is what showed otherwise. The camera-ready reports the
control.

## Onset localisation

| Quantity | Value |
|---|---|
| Model onset latency, median | 30.0 s |
| Model onset latency, mean | 21.5 s |
| Within 5 s | 15 of 53 |
| PELT baseline, median | 10.5 s |

Source: `results/onset_eval_results.json`.

The 0.46 s figure that appeared in earlier drafts is a **supervised alignment
offset** on n = 7, measured after placing Gaussian targets at the exact onset
during formation. It is not a detection latency and is not comparable with
event-level detection latencies reported elsewhere in the literature. The
camera-ready states this plainly rather than implying superior onset
detection.

The PELT median rests on only three windows in which PELT produced any
changepoint at all (average 0.3 changepoints per window), so it is a weak
comparison in both directions.

## TCAV and spectral analyses

The camera-ready reports concept classifiers at 0.84 to 0.89 cross-validated
accuracy across the three Sleep-EDF concepts. The run preserved in
`notebooks/tcav.ipynb` prints 0.909 (delta), 0.821 (spindle) and 0.868
(alpha), a slightly wider spread on both ends, so the saved run and the
reported range are not the same computation. The paper's range stands; no
claim turns on the difference.

The TCAV scores themselves do reproduce exactly: alpha drives Wake (0.807),
the spindle band drives N1 (0.747), and high-frequency content drives seizure
detection (0.810), against the paper's 0.81, 0.75 and 0.81.

**A caveat the notebook makes visible.** Not every concept-class pair follows
clinical expectation. Delta activity dominates N3 sleep physiologically, yet
the delta concept scores 0.483 for N3, marginally below the 0.5 line that
marks a positive influence, and 0.050 for N1. The CHB-MIT concept classifiers
are also weaker than the Sleep-EDF ones (0.736 for high frequency, 0.714 for
high amplitude), so the seizure-side scores rest on a shakier concept
separation. The pairs the paper reports are real and reproduce, but they are
the strongest of a mixed set. This is part of why the interpretability
results are marked preliminary rather than confirmatory.

Boundary-conditioned spectral contrasts, comparing the top and bottom 20 per
cent of tokens by boundary value: on Sleep-EDF, high-boundary regions show
2.1x higher beta and 3.3x higher gamma power (p < 10^-22), and CHB-MIT
high-boundary regions roughly 25 per cent lower amplitude variability
(p < 10^-38), both surviving Bonferroni correction.
`notebooks/boundary_spectral_analysis.ipynb` reproduces the Sleep-EDF ratios
exactly (2.105 and 3.267). It also shows that on CHB-MIT the beta and gamma
ratios themselves run below one (0.729 and 0.762), so the CHB-MIT contrast is
carried by amplitude variability rather than by band power.

**Two limits of this analysis, visible in the notebook output.** Each token's
spectral features are computed from a 0.3 second segment (30 samples at
100 Hz) with `nperseg=30`, giving Welch bins spaced 3.33 Hz apart. Delta,
theta, alpha and sigma each span fewer than two bins at that resolution, and
`np.trapz` over a single bin returns exactly zero. That is why those four
bands print as 0.0000 with p = 1.0000 throughout: they were never measured,
rather than measured and found absent. Only beta and gamma are wide enough to
be resolved, so the reported contrasts are real but the low-frequency
comparison is vacuous. Re-running on longer segments, or with a resolution
matched to the delta band, is the fix.

Second, this notebook and the TCAV and figure notebooks all load checkpoints
with `load_state_dict(..., strict=False)` after renaming keys. Under
`strict=False` a key mismatch is silent, so a failed rename would leave the
analysis running on partly random weights without raising. The outputs look
like a trained model, but the load is not verified by the code as written.
Anyone rerunning these should set `strict=True` or assert on the returned
missing and unexpected key lists.

These are reported as **consistent with** neurophysiological meaning, not as
confirmation of it. Given the contiguity result above, a plausible alternative
account is that high-boundary tokens sit near epoch junctions, where splices
introduce spectral discontinuities. Distinguishing the two accounts needs the
contiguous re-preprocessing described in `reproducibility.md`.

## Comparison with published methods

| Method | Acc | kappa | Context | Interpretability |
|---|---|---|---|---|
| AttnSleep | 0.829 | 0.77 | 3 ep. | post-hoc attention |
| SleepTransformer | 0.812 | --- | 21 ep. | attention + UQ |
| L-SeqSleepNet | 0.886 | 0.845 | 200 ep. | none |
| NeuroState | 0.698 | 0.596 | 3 ep. | ACBL boundaries |

| Method (CHB-MIT) | AUROC | Context |
|---|---|---|
| BIOT vanilla | 0.865 | 1 ep. |
| BIOT pre-trained | 0.876 | 1 ep. |
| NeuroState | 0.924 | 3 ep. |

Protocol differences are real and disclosed: AttnSleep is on Sleep-EDF-78,
L-SeqSleepNet on a 39-recording subset with SHHS pretraining, BIOT on its
appendix 6/3/3 patient-wise split. These are cross-cited published numbers,
not matched-protocol re-runs. See `reproducibility.md`.

None of the compared architectures provides structural boundary detection.
The accuracy gap on Sleep-EDF is a deliberate trade for an auxiliary boundary
output none of them produces, subject to the limits the analyses above place
on what that output has been shown to represent.
