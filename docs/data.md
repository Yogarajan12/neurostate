# Datasets

None of the three datasets is redistributed here. Each requires registration
or a data use agreement. This page covers how to obtain each one, how it was
preprocessed, and what the resulting HDF5 file contains.

All three share a common format so one model and one training loop serve all
of them:

| Key | Shape | Notes |
|---|---|---|
| `epochs` | (n_epochs, n_channels, 3000) | 30 s at 100 Hz, z-scored per channel per epoch |
| `labels` | (n_epochs,) | task-specific class index |
| `subject_ids` | (n_epochs,) | byte strings; absent in TUAB chunks |
| `times` | (n_epochs,) | epoch onset in seconds; **Sleep-EDF only** |

Common preprocessing: 0.5 to 35 Hz bandpass, resampling to 100 Hz,
non-overlapping 30-second epochs, per-channel z-scoring, amplitude-based
artefact rejection. A 60 Hz notch is applied to the two US datasets.

---

## Sleep-EDF Expanded

**Task.** 5-class sleep staging (Wake, N1, N2, N3, REM).
**Access.** Public on PhysioNet under ODC-By 1.0. Fetched through
`mne.datasets.sleep_physionet.age.fetch_data`, both nights per subject.
**Licence.** Open Data Commons Attribution. Cite Kemp et al. 2000.

**Preprocessing.** Channels `EEG Fpz-Cz`, `EEG Pz-Oz` and `EOG horizontal`
(three channels). Bandpass 0.5 to 35 Hz with a firwin design, no notch since
the European recordings are already line-filtered. Stage labels follow AASM
mapping with N3 and N4 merged; unknown and movement epochs are dropped.
Epochs with peak-to-peak amplitude above 500 µV are flagged in
`bad_epochs_mask`. Z-scored per channel per epoch.

**Result.** 7,360 epochs, 147 recording identifiers, 3 channels.

**Two things a reproducer should know.** The annotation filter keeps only
annotations lasting exactly one 30-second epoch, which retains short
transitional runs and discards long stable ones. And the stored identifiers
are recordings rather than people, so the two nights of one subject carry
different identifiers. Both are explained in
[`reproducibility.md`](reproducibility.md); they affect how these numbers
compare with published sleep stagers.

The file also optionally stores `pelt_boundaries`, changepoints from PELT
(rbf cost, min size 100, penalty 10) computed on unnormalised epochs, used as
a classical baseline.

---

## CHB-MIT Scalp EEG

**Task.** Binary seizure detection, evaluated cross-subject.
**Access.** Public on PhysioNet at `physionet.org/content/chbmit/1.0.0/`.
**Licence.** ODC-By 1.0. Cite Shoeb and Guttag 2010.

**Subjects used.** Twelve: chb01, chb03, chb05, chb08, chb10, chb12, chb14,
chb15, chb17, chb19, chb20, chb22.

**Preprocessing.** Seizure annotations are parsed from the per-subject summary
files, which use several inconsistent formats (`Seizure Start Time`,
`Seizure 1 Start Time`, `Seizure2 Start Time`); the parser in
`CHB_MIT_Expanded_Preprocessing` handles all known variants. Channel names are
normalised and restricted to the 18 standard bipolar montage channels, then
truncated to the 17 shared across all twelve subjects. A 60 Hz notch,
0.5 to 35 Hz bandpass, resampling 256 to 100 Hz, z-scoring per channel per
epoch, and rejection of epochs exceeding 10 standard deviations.

An epoch is labelled seizure if it overlaps any annotated seizure interval.
Up to three whole non-seizure files per subject are added for class balance.

**Result.** 9,639 epochs, 12 subjects, 17 channels, 312 seizure and 9,327
normal (3.2 per cent prevalence).

**Splitting.** `round_robin_split` sorts subjects by seizure count descending
and deals them in a train, train, val, test pattern. The result is
deterministic and does not depend on a seed, so it is fixed by construction
across multi-seed runs:

```
train  chb15, chb12, chb01, chb03, chb14, chb17   (195 seizure epochs)
val    chb08, chb10, chb22                        (63)
test   chb05, chb20, chb19                        (54)
```

giving 4,671 / 3,448 / 1,496 windows, with 53 seizure windows in test. **If a
rebuilt file gives a different subject list or window count, it does not match
the published run.**

**Mixed pipelines.** Subjects chb01, chb03 and chb05 were processed by an
earlier pipeline that cut 50 per cent overlapping seizure windows. See
[`reproducibility.md`](reproducibility.md).

---

## TUAB (TUH Abnormal EEG Corpus)

**Task.** Binary normal versus abnormal EEG, used as a specificity control.
**Access.** Requires a signed data use agreement with the Temple University
Hospital EEG Corpus at `isip.piconepress.com`. Fetched by rsync with an SSH
key once access is granted.
**Licence.** Restricted by the TUH agreement. **Do not redistribute the data
or, without checking the agreement terms, model weights derived from it.**

**Preprocessing.** Channel names are mapped through an alias table covering
the `-REF`, `-LE`, `EEG ` prefix and T7/T8 versus T3/T4 conventions, giving
the standard 19-channel 10-20 montage. Files shorter than 60 seconds or with
fewer than 10 mappable channels are dropped. A 60 Hz notch, 0.5 to 35 Hz
bandpass, resampling to 100 Hz, zero-padding to 19 channels where a channel
is missing, z-scoring per channel per epoch, and amplitude-based rejection.

Labels apply to the whole recording, so every epoch from a file carries the
same label. **This is what makes TUAB a specificity control: no window
contains a within-window transition by construction, so a flat boundary output
is the correct response.**

**Result.** Saved as chunked HDF5 (`tuab_train_chunk00.h5` onward, 200 files
per chunk) because the corpus does not fit in memory. Eval is 12,241 epochs
from 276 files; train is 123,461 epochs from 2,717 files across 14 chunks. The
published run loaded eval plus the first seven train chunks, 74,395 epochs.

**Caveats.** The random split is over epoch indices rather than recordings,
and file discovery lists normal before abnormal. Both are documented in
[`reproducibility.md`](reproducibility.md) and are why the TUAB
classification number should not be quoted as a benchmark result.

---

## Preprocessing scripts

The preprocessing pipelines live in the notebooks under `notebooks/`, which
are the originals used to produce the published files. They depend on MNE and,
for the PELT baseline, `ruptures`:

```bash
pip install -e ".[preprocessing]"
```

Expected layout once built:

```
data/
  raw/
    physionet-sleep-data/          SC*-PSG.edf, SC*-Hypnogram.edf
    chb-mit/chbNN/                 *.edf, chbNN-summary.txt
    tuh_eeg_abnormal/{train,eval}/{normal,abnormal}/01_tcp_ar/*.edf
  processed/
    sleep_edf_processed.h5
    chbmit_combined_seizure_detection.h5
    tuab_eval_processed.h5, tuab_train_chunk*.h5
```

`data/` is gitignored. Paths are set per dataset in `configs/`.

## Citing the datasets

- **Sleep-EDF.** B. Kemp et al., "Analysis of a sleep-dependent neuronal
  feedback loop: the slow-wave microcontinuity of the EEG," *IEEE Trans.
  Biomed. Eng.*, 47(9), 2000.
- **CHB-MIT.** A. H. Shoeb and J. V. Guttag, "Application of machine learning
  to epileptic seizure detection," *ICML*, 2010.
- **TUAB.** I. Obeid and J. Picone, "The Temple University Hospital EEG data
  corpus," *Front. Neurosci.*, 10:196, 2016.
