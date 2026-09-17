# The eleven interventions

A record of what was tried against auxiliary head collapse and how each
attempt failed. Negative results are usually left out of papers and repos;
they are the bulk of the work here and the reason the positive result is
credible, so they are written up in full.

All runs are on Sleep-EDF, single seed (42), which is what Table 1 of the
paper reports. Boundary std below **0.020** is degenerate; that threshold is
justified empirically against a shuffled-input null of 0.0154 (95th percentile
0.0155), documented in [`results.md`](results.md).

| # | Category | Intervention | Acc | Bnd Std | Outcome |
|---|---|---|---|---|---|
| 1 | Masking | Sigmoid mask (baseline) | 0.703 | 0.010 | collapsed |
| 2 | Masking | Gumbel-Softmax discretisation | 0.642 | 0.003 | collapsed |
| 3 | Masking | Hard threshold | 0.677 | 0.004 | collapsed |
| 4 | ACBL loss | Boundary losses alone | 0.667 | 0.011 | collapsed |
| 5 | ACBL loss | + variance penalty | 0.690 | 0.007 | collapsed |
| 6 | ACBL loss | **+ gradient isolation** | **0.718** | **0.028** | **non-degenerate** |
| 7 | Scheduling | Bimodal loss, extended formation | 0.626 | 0.014 | collapsed |
| 8 | Scheduling | Temperature annealing + focal loss | 0.666 | 0.006 | collapsed |
| 9 | Architectural | Cumulative product masking | 0.678 | 0.018 | collapsed (to all-ones) |
| 10 | Alt. parameterisation | Differentiable HMM | 0.546 | n/a | degenerate |
| 11 | Alt. parameterisation | Query-based segmentation | 0.707 | n/a | degenerate |

Rows 4, 5 and 6 are **cumulative**, not alternatives: row 5 is row 4 plus the
variance penalty, and row 6 is row 5 plus gradient isolation.

Two rows sit close enough to the cutoff that the statistic alone does not
settle them: bimodal scheduling at 0.014 and cumulative-product masking at
0.018. Both are called degenerate on the failure observed during training,
constant outputs and all-ones mask saturation, together with their accuracies
of 0.626 and 0.678 against 0.718 for gradient isolation. The 0.020 line is a
descriptive label rather than a test.

The boundary statistic in this table is pooled over all test windows, matching
Table 2. The **ablation table in the paper uses batch-averaged boundary
standard deviations instead**, so its entries are comparable within that table
but not against the values here.

---

## Masking variants (1 to 3)

**Hypothesis.** The sigmoid mask is too soft. If boundaries were forced to be
discrete, the head could not hedge with a mid-range constant and would have to
commit to positions.

**What was tried.** Gumbel-Softmax discretisation with a temperature schedule,
and a hard threshold on `b(t)`.

**What happened.** Both collapsed harder than the sigmoid baseline (0.003 and
0.004 against 0.010), and accuracy fell. Discreteness changes what the mask
looks like but not the gradient dynamics that produce it: a discrete mask
stuck at a constant value is still a constant mask, and the classifier is
still content with the distance-decay prior it induces.

**What this ruled out.** That collapse is an artefact of soft masking.

---

## ACBL loss variants (4 to 6)

**Hypothesis.** The head collapses because nothing supervises it. Adding
auxiliary objectives should give it something to learn.

**Row 4, boundary losses alone (0.011).** Pseudo-boundary contrast, Gaussian
targets at epoch junctions and the attention KL prior, all active, gradients
fully coupled. Still degenerate. The auxiliary losses pull toward structure
while the classification gradient pulls toward flatness, and flatness wins.

**Row 5, plus the variance penalty (0.007).** Penalising constant outputs
directly seemed like the obvious fix, and it made things slightly *worse*. The
term rewards spread but provides no gradient distinguishing one boundary
position from another, so it can be satisfied by noise rather than structure
while the classification gradient continues flattening anything meaningful.

This is the most instructive negative result in the table: **the direct fix
for a symptom does not address its cause.**

**Row 6, plus gradient isolation (0.028).** Same four losses, same weights,
same schedule. The only change is that the boundary output is detached on the
path into attention during formation. Non-degenerate, and accuracy improves to
the best in the table.

**What this established.** The problem is not insufficient supervision. It is
that supervision and classification compete for the same parameters at the
same time, and classification wins because it has a shortcut available.

Reproduce these with `configs/variants/no_isolation.yaml`,
`no_variance_penalty.yaml` and `variance_only.yaml`.

---

## Aggressive scheduling (7 and 8)

**Hypothesis.** The head has enough time but the wrong pressure. Pushing the
output distribution toward the extremes, or extending the window in which the
auxiliary losses dominate, should force structure.

**Row 7, bimodal loss with extended formation (0.014).** A loss encouraging
`b(t)` toward 0 or 1, with a longer formation phase.

**Row 8, temperature annealing with focal loss (0.006).** An annealed
temperature on the pseudo-targets plus focal weighting on the boundary BCE.

**What happened.** Neither recovered structure. Both found *new degenerate
solutions*: the head satisfied the reshaped objective while remaining
uninformative, for instance by producing a bimodal but input-independent
pattern.

**What this ruled out.** That collapse is an optimisation-schedule problem
solvable by tuning. Harder pressure produces more creative degeneracy, not
structure.

---

## Architectural change (9)

**Hypothesis.** The failure lives in the additive cumulative-sum mask. A
different masking operation might not admit a useful-when-flat solution.

**What was tried.** Cumulative-product masking in place of the cumulative sum.

**What happened.** The head collapsed to **all-ones** rather than all-zeros
(std 0.018). The direction of collapse changed; the collapse did not.

**Why this matters.** It shows the masking operation sets the *direction* of
the degenerate solution while the underlying pathology is invariant to it.
The problem is not which degenerate point the head finds, but that a
degenerate point is always available and always sufficient for the classifier.

---

## Alternative parameterisations (10 and 11)

**Hypothesis.** If the pathology is a property of this particular boundary
representation, a fundamentally different representation should avoid it.

**Row 10, differentiable HMM.** Boundaries as state-transition probabilities
in a learned hidden Markov model rather than a per-token sigmoid.

**Row 11, query-based segmentation.** A Mask2Former-style formulation where
learned queries attend to token features and produce segment assignments.

**What happened.** Both degenerated analogously despite representing
boundaries in entirely different ways. The HMM run also lost substantial
accuracy (0.546).

**An honest caveat.** Boundary std is marked n/a for these two rows because
their boundary statistic is not comparable with the sigmoid-based runs: they
do not produce a per-token probability in `[0,1]` with the same meaning. The
judgment that they degenerated rests on accuracy together with inspection of
the learned outputs, not on the same threshold applied to the other nine rows.
That is weaker evidence, and the paper scopes the claim accordingly rather
than asserting architecture-independent collapse.

**What this suggests.** The pathology lies in the joint optimisation itself
rather than in any particular boundary representation. That a single mechanism
resolves collapse across four very different parameterisations is the main
argument for gradient isolation being principled rather than a lucky trick.

---

## Reproducing these

Interventions expressible as configuration changes have configs in
`configs/variants/`:

```bash
python scripts/train.py --config configs/variants/no_isolation.yaml --seeds 42
```

Rows 2, 3, 9, 10 and 11 need a different forward pass or module and are not
reproducible by configuration alone. The model code in `src/neurostate/model.py`
is the place to start for those; each is a change to
`ContrastiveBoundaryModule` or to `RegimeStructuredAttention._build_regime_mask`.

## What was not run

**Table 1 is single-seed.** Running seven variants across three seeds is
roughly 60 GPU-hours, which did not fit before the camera-ready deadline. The
paper states this explicitly. `run_collapse_table` in the collapse notebook
remains in place so the sweep can be run later; the headline tables (Table 2)
did get five seeds, which was the higher priority since that is where the
reported performance numbers live.

**Split-variance experiments.** Varying the subject split rather than the
initialisation was dropped for time. It applies only to Sleep-EDF, since the
CHB-MIT round-robin split is deterministic. This means the reported variance
captures initialisation and batch order, not split sensitivity.
