# Reproducibility and known limitations

What to check before comparing these numbers against published work, and what
a reproducer will find on inspecting the data pipeline. Everything here
describes how the code and the preprocessed files behave. Where that behaviour
constrains a claim, the constraint is stated.

## Protocol

**Model selection uses validation classification loss only.** Test data is
never consulted during training or checkpoint selection. Early stopping is
enabled only once the full phase begins, so a run cannot terminate during
formation and be evaluated with the boundary head still detached.

**Classification metrics and boundary statistics come from the same
checkpoint.** `evaluate()` computes both from whatever state is loaded. This
matters: an earlier version of the CHB-MIT numbers reported classification
from the validation-selected epoch-15 checkpoint but boundary std from the
final epoch, which is not internally consistent. The camera-ready reports both
from the restored best checkpoint.

**Hyperparameters were fixed in advance.** Phase lengths (warmup 3, formation
12) and ACBL weights are identical across all three datasets. They were not
tuned per task and never selected on test data. The reuse is the evidence.

**Variance is over training seeds, not splits.** For Sleep-EDF the split seed
is held at 42 while training seeds vary, so the reported standard deviations
reflect initialisation and batch order on the same held-out set. Split
sensitivity was not measured. For CHB-MIT the round-robin split is
deterministic and does not depend on a seed at all.

## Determinism

**The published runs did not enforce deterministic kernels.** The paper states
this plainly: two runs of the reported Sleep-EDF setting gave boundary
standard deviations of 0.023 and 0.027, within the across-seed spread. Run-to-
run variation of that size is the resolution at which the boundary statistics
should be read.

`set_all_seeds()` in this package goes further than the published runs did. It
seeds Python, NumPy and PyTorch, sets `cudnn.deterministic = True` and
disables `cudnn.benchmark`, which costs some speed and buys repeatability on
identical hardware and library versions. A reproduction attempt should
therefore expect to land within the 0.023 to 0.027 band rather than to hit a
published figure exactly, and should not read a difference of that magnitude
as a failure to reproduce.

## Split behaviour worth knowing

These points describe the preprocessed files and the split code. None
contradicts the paper, and each is something a reproducer will encounter
directly.

### Sleep-EDF: identifiers are recordings, not people

`subject_ids` in the HDF5 file are derived from PSG filenames, for example
`SC4591G0` and `SC4592G0`. In Sleep-EDF Expanded these two files are the two
nights of the **same** subject. The split in
`create_sleep_edf_dataloaders` is over these identifiers, so the two nights of
one person can land on opposite sides of a train/test boundary.

The 147 identifiers therefore correspond to at most 78 people. Published
Sleep-EDF results using subject-level cross-validation are split at the person
level. When comparing, note that this protocol is not identical.

### Sleep-EDF: most windows are not contiguous in time

Preprocessing kept only annotations lasting exactly one 30-second epoch.
Hypnograms encode variable-duration runs of a single stage, so this filter
retains the short transitional runs and discards long stable ones. Two
consequences:

- The retained subset is dominated by transitional epochs, the hardest to
  classify. This is part of why accuracy sits below published sleep stagers,
  alongside the deliberately short 3-epoch context window.
- Only **33.5%** of within-subject adjacent epoch pairs are one epoch apart in
  recording time (median gap 180 s), so only **15.6%** of test windows are
  fully contiguous. A three-epoch window usually contains at least one splice.

A signal-level check confirms this: the mean sample step across an epoch join
is 4.10× the within-epoch step overall, 1.55× on truly contiguous pairs and
5.42× on spliced ones.

**This is the strongest constraint on the interpretability claims.** Boundary
std is 0.0221 on contiguous windows against 0.0226 on spliced ones. If the
head responded to physiological change rather than window position, those
would differ. Reproduce with
`scripts/analyse_boundaries.py --analyses contiguity`.

### CHB-MIT: the combined file mixes two preprocessing pipelines

Subjects chb01, chb03 and chb05 were processed by an earlier pipeline that
cut **50% overlapping** windows around seizures and sampled normal epochs at a
3:1 ratio. The remaining nine subjects were processed by the later pipeline,
which cuts **non-overlapping** 30-second epochs and adds up to three whole
normal files per subject for balance.

chb05 is a test subject and contributes 34 of the 54 test seizure epochs. For
those, "three consecutive epochs" in the stored array are overlapping crops of
the same underlying recording rather than three successive 30-second windows.
This affects what a window means on part of the test set.

The expected split is train chb15, chb12, chb01, chb03, chb14, chb17; val
chb08, chb10, chb22; test chb05, chb20, chb19, giving 1,496 test windows of
which 53 are seizure. A different subject list means the file does not match
the published run.

### TUAB: split leakage and class ordering

Two things compound here, which is why TUAB is framed as a specificity control
rather than a performance result:

**The split is over epoch indices, not recordings.** A window needs three
consecutive surviving indices, so of 14,879 test epochs only **604** form
windows (about 4%). Those windows come from recordings also represented in
training, so the TUAB AUROC of 0.913 is measured under within-recording
leakage on a small window count.

**File discovery lists normal before abnormal within each split.** Loading a
prefix of chunks (`max_train_chunks: 7`) therefore draws a class-skewed sample
rather than a representative one.

Neither affects the specificity claim the paper makes with TUAB, which is that
the boundary output correctly stays flat (0.003) where no within-window
transition exists by construction. Both would matter if the TUAB
classification number were quoted as a benchmark result. It should not be.

## What was not measured

- **Table 1 is single-seed.** Seven variants across three seeds is roughly 60
  GPU-hours and did not fit before the deadline. The five-seed statistics went
  to the headline tables instead.
- **Split sensitivity on Sleep-EDF.** Dropped for time.
- **A matched-protocol baseline re-run.** The comparison table cross-cites
  published numbers. Differences in subset and protocol are disclosed in the
  table footnotes: AttnSleep on Sleep-EDF-78, L-SeqSleepNet on a 39-recording
  subset with SHHS pretraining, BIOT on its appendix 6/3/3 patient split.
- **Boundary supervision independent of the labels used to validate it.**
  `L_Gauss` anchors the head to epoch junctions, and the contiguity result
  shows junction position is much of what the head tracks. Breaking that
  circularity is the main methodological next step.

## Next steps that would strengthen the work

In rough order of value:

1. **Re-preprocess Sleep-EDF keeping contiguous runs**, then re-run the
   contiguity analysis. This directly addresses the strongest limitation.
2. **Re-split TUAB at the recording level** and re-run, removing leakage.
3. **Rebuild CHB-MIT through a single pipeline** so all subjects share a
   windowing convention.
4. **Split Sleep-EDF at the person level** by mapping recordings to subjects,
   making the protocol directly comparable to published work.
5. **Run the collapse table across seeds** now that deadline pressure has
   passed.
6. **Matched-protocol baselines** on the same splits.
