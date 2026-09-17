# Model card: NeuroState

## Model details

**Name.** NeuroState, a multi-resolution EEG transformer with an auxiliary
boundary module trained by Attention-Contrastive Boundary Learning (ACBL) with
gradient isolation.

**Developed by.** Yogarajan Sivakumar and Hong Man, Department of Electrical
and Computer Engineering, Stevens Institute of Technology. Originated as an
MSc capstone project.

**Version.** 1.0.0, matching the IEEE MLSP 2026 camera-ready.

**Type.** Supervised sequence classifier over multi-channel EEG, with an
auxiliary per-token changepoint output.

**Size.** 1.00 M parameters on a 3-channel montage, 1.07 M on 17 or 19
channels. The difference is the spatial convolution, which scales with input
channel count.

**Input.** Three consecutive 30-second EEG epochs at 100 Hz, z-scored per
channel per epoch. 588 tokens per window.

**Output.** A class label for the centre epoch, and a boundary probability
sequence `b(t) ∈ [0,1]^588`.

**Licence.** MIT for the code. Datasets carry their own terms and are not
redistributed.

## Intended use

**Intended.** Research into auxiliary-head optimisation dynamics in
transformers; research into structurally interpretable sequence models for
biosignals; a reproduction target for the published results; a starting point
for applying gradient isolation to other auxiliary-head architectures.

**Out of scope.** This is **not a medical device** and must not be used for
clinical decision-making, diagnosis, triage, patient monitoring or any
deployment where an output affects patient care. It has no regulatory
clearance, has not been validated prospectively, and has not been evaluated on
any clinical population beyond the three research datasets below.

**Not recommended without substantial further work.** Transfer to other
biosignals, deployment on recording setups differing from the training
montages, or use of the boundary output as a clinical timing signal. The
boundary output in particular has not been shown to localise physiological
transitions; see "Limitations".

## Training data

| Dataset | Task | Size | Access |
|---|---|---|---|
| Sleep-EDF Expanded | 5-class sleep staging | 7,360 epochs, 147 recordings, 3 ch | PhysioNet, ODC-By |
| CHB-MIT | Binary seizure detection | 9,639 epochs, 12 subjects, 17 ch | PhysioNet, ODC-By |
| TUAB | Binary abnormality detection | 74,395 epochs used, 19 ch | TUH data use agreement |

All three are research corpora. CHB-MIT is paediatric; Sleep-EDF is a European
adult cohort; TUAB is a US clinical population. Demographic composition was
not analysed and no subgroup performance analysis was performed. Full
preprocessing detail is in [`docs/data.md`](docs/data.md).

## Evaluation

Five training seeds with a fixed split; TUAB is single-seed. Model selection
used validation classification loss only, and test data was never consulted
during training or selection.

| Dataset | Acc | AUROC | κ | Bnd Std |
|---|---|---|---|---|
| Sleep-EDF | 0.698 ± 0.013 | — | 0.596 ± 0.018 | 0.026 ± 0.003 |
| CHB-MIT | 0.901 ± 0.043 | 0.924 ± 0.014 | 0.356 ± 0.132 | 0.014 ± 0.006 |
| TUAB | 0.916 | 0.913 | 0.545 | 0.003 |

Sleep-EDF accuracy is below published sleep stagers (AttnSleep 0.829,
L-SeqSleepNet 0.886). This is partly a deliberate trade for the short 3-epoch
context window and partly a consequence of an annotation filter that leaves
the subset dominated by short transitional runs.

CHB-MIT κ of 0.356 alongside AUROC 0.924 reflects 3.2 per cent seizure
prevalence, which compresses chance-corrected agreement under strong ranking
performance.

Full numbers and their provenance: [`docs/results.md`](docs/results.md).

## Limitations

**The boundary output has not been shown to mark physiological transitions.**
Three analyses constrain this, all reported in the paper:

- On CHB-MIT, windows containing a label transition are statistically
  indistinguishable from those without (Mann-Whitney p ≥ 0.21, d = −0.25).
- On Sleep-EDF, boundary variation is the same on windows contiguous in
  recording time (0.0221) as on windows containing a splice (0.0226),
  indicating the head tracks window position more than signal content.
- The apparent elevation of boundary activation in seizure epochs follows from
  window geometry: it is present in 99.9 per cent of non-seizure windows too,
  and more strongly.

**Sub-epoch onset localisation is not established.** General onset latency is
median 30.0 s across all 53 test seizures, against a PELT baseline at 10.5 s.
The 0.46 s figure in earlier drafts is a supervised alignment offset on n = 7,
not a detection latency.

**Interpretability results are preliminary.** TCAV and spectral analyses are
consistent with neurophysiological meaning but do not establish it. Given the
contiguity result, splice-induced spectral discontinuity is a live alternative
explanation.

**Protocol caveats.** Sleep-EDF identifiers are recordings rather than people,
so two nights of one subject can fall on opposite sides of a split. The TUAB
split is over epochs rather than recordings, giving within-recording leakage
on a small window count; TUAB should be read as a specificity control only.
The CHB-MIT file mixes two preprocessing pipelines, with overlapping windows
for three subjects including one test subject. All detailed in
[`docs/reproducibility.md`](docs/reproducibility.md).

**Scope of the central finding.** Collapse and gradient isolation are
established within the EEG settings measured here. Architecture-independence
is suggested by the result holding across four boundary parameterisations, but
has not been measured.

**Not measured.** Table 1 is single-seed. Split sensitivity, matched-protocol
baseline re-runs, and subgroup performance were not run.

## Ethical considerations

**Clinical risk.** A model that appears to localise seizure onset but does not
is worse than one making no such claim, because a spurious timing signal could
be trusted. The negative results above are reported prominently for that
reason, in the paper, the README and this card.

**Data.** All three datasets are de-identified and publicly documented.
CHB-MIT contains paediatric recordings; TUAB requires a signed agreement.
Users must obtain data through the proper channels and respect the terms.

**Redistribution of weights.** Sleep-EDF and CHB-MIT checkpoints may be shared
under ODC-By with attribution. **Check the TUH data use agreement before
sharing TUAB-derived weights.**

**Environmental cost.** Reproducing the published results is roughly 25 to 30
GPU-hours. The unrun collapse sweep would add about 60 more.

## Citation

```bibtex
@inproceedings{sivakumar2026neurostate,
  title     = {NeuroState: Gradient-Isolated Boundary Learning for
               Interpretable {EEG} Transformers},
  author    = {Sivakumar, Yogarajan and Man, Hong},
  booktitle = {2026 IEEE International Workshop on Machine Learning for
               Signal Processing (MLSP)},
  year      = {2026},
  address   = {Atlanta, GA, USA},
}
```
