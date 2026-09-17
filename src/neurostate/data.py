"""Datasets and dataloaders for Sleep-EDF, CHB-MIT and TUAB.

All three datasets are stored as HDF5 with an `epochs` array of shape
(n_epochs, n_channels, n_samples), a `labels` array, and, where available, a
`subject_ids` array. See docs/data.md for how each file is built and
docs/reproducibility.md for the split behaviour that matters when comparing
against published numbers.
"""

from pathlib import Path

import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

__all__ = [
    "MultiEpochDataset",
    "CHBMITOnsetDataset",
    "TUABWindowDataset",
    "create_sleep_edf_dataloaders",
    "create_chbmit_dataloaders",
    "create_tuab_dataloaders",
    "round_robin_split",
    "get_class_weighted_sampler",
]


class MultiEpochDataset(Dataset):
    """Three consecutive epochs, labelled by the centre epoch.

    A window is kept only when all three epochs are present in the same index
    subset and share a subject identifier, so windows never straddle a split
    boundary or a subject change.

    Adjacency here means adjacency in the stored array. Whether two adjacent
    rows are also adjacent in recording time depends on the preprocessing;
    docs/reproducibility.md quantifies this for Sleep-EDF.
    """

    def __init__(self, h5_path, target_channels=3, target_samples=3000,
                 indices=None, context_size=1, verbose=True):
        self.h5_path = Path(h5_path)
        self.target_channels = target_channels
        self.target_samples = target_samples
        self.context_size = context_size
        self._h5_file = None
        with h5py.File(self.h5_path, 'r') as f:
            self.total_samples = f['epochs'].shape[0]
            self.labels = f['labels'][:]
            self.subject_ids = f['subject_ids'][:]
            if isinstance(self.subject_ids[0], (bytes, np.bytes_)):
                self.subject_ids = np.array([
                    s.decode() if isinstance(s, bytes) else s
                    for s in self.subject_ids])
        base_indices = (indices if indices is not None
                        else np.arange(self.total_samples))
        base_set = set(base_indices)
        self.valid_indices = []
        for idx in base_indices:
            valid = True
            for offset in range(-context_size, context_size + 1):
                neighbour = idx + offset
                if neighbour < 0 or neighbour >= self.total_samples:
                    valid = False
                    break
                if neighbour not in base_set:
                    valid = False
                    break
                if self.subject_ids[neighbour] != self.subject_ids[idx]:
                    valid = False
                    break
            if valid:
                self.valid_indices.append(idx)
        self.valid_indices = np.array(self.valid_indices)
        self._compute_class_weights()
        if verbose:
            dist = dict(zip(*np.unique(
                self.labels[self.valid_indices], return_counts=True)))
            print(f"  MultiEpoch: {len(self.valid_indices)} windows, "
                  f"classes={dist}")

    def _get_h5(self):
        if self._h5_file is None:
            self._h5_file = h5py.File(self.h5_path, 'r')
        return self._h5_file

    def _compute_class_weights(self):
        sub = self.labels[self.valid_indices]
        cls, cnt = np.unique(sub, return_counts=True)
        w = 1.0 / cnt
        w /= w.sum()
        self.class_weights = dict(zip(cls, w))
        self.sample_weights = np.array(
            [self.class_weights[label] for label in sub])

    def __len__(self):
        return len(self.valid_indices)

    def __getitem__(self, idx):
        centre_idx = self.valid_indices[idx]
        f = self._get_h5()
        epochs = []
        epoch_labels = []
        for offset in range(-self.context_size, self.context_size + 1):
            ep = f['epochs'][centre_idx + offset]
            epochs.append(self._standardise_shape(ep))
            epoch_labels.append(self.labels[centre_idx + offset])
        return {
            'epoch': torch.tensor(np.stack(epochs, axis=0),
                                  dtype=torch.float32),
            'label': torch.tensor(self.labels[centre_idx], dtype=torch.long),
            'epoch_labels': torch.tensor(epoch_labels, dtype=torch.long),
        }

    def _standardise_shape(self, epoch):
        nc, nt = epoch.shape
        if nc < self.target_channels:
            epoch = np.vstack(
                [epoch, np.zeros((self.target_channels - nc, nt))])
        elif nc > self.target_channels:
            epoch = epoch[:self.target_channels]
            nc = self.target_channels
        if nt < self.target_samples:
            epoch = np.hstack(
                [epoch, np.zeros((nc, self.target_samples - nt))])
        elif nt > self.target_samples:
            s = (nt - self.target_samples) // 2
            epoch = epoch[:, s:s + self.target_samples]
        return epoch

    def get_sampler(self):
        return WeightedRandomSampler(self.sample_weights, len(self), True)


class CHBMITOnsetDataset(Dataset):
    """Three consecutive CHB-MIT epochs from one subject, labelled by centre.

    Windows are built per subject and require strictly consecutive indices in
    the stored array.
    """

    def __init__(self, epochs, labels, subject_ids, subject_list,
                 n_context=3, verbose=True):
        mask = np.isin(subject_ids, subject_list)
        self.epochs = epochs[mask]
        self.labels = labels[mask]
        self.subject_ids = subject_ids[mask]
        self.n_ctx = n_context
        self.windows = []
        for subj in subject_list:
            subj_idx = np.where(self.subject_ids == subj)[0]
            if len(subj_idx) < n_context:
                continue
            for start in range(len(subj_idx) - n_context + 1):
                group = subj_idx[start:start + n_context]
                if np.all(np.diff(group) == 1):
                    centre = n_context // 2
                    self.windows.append({
                        'indices': group,
                        'label': int(self.labels[group[centre]]),
                        'epoch_labels': [int(self.labels[g]) for g in group],
                        'subject': subj,
                    })
        if verbose:
            win_labels = [w['label'] for w in self.windows]
            print(f"  {len(self.windows)} windows "
                  f"({sum(win_labels)} seizure, "
                  f"{len(win_labels) - sum(win_labels)} normal)")

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, idx):
        w = self.windows[idx]
        x = np.stack([self.epochs[i] for i in w['indices']], axis=0)
        return {
            'epoch': torch.tensor(x, dtype=torch.float32),
            'label': torch.tensor(w['label'], dtype=torch.long),
            'epoch_labels': torch.tensor(w['epoch_labels'],
                                         dtype=torch.long),
        }


class TUABWindowDataset(Dataset):
    """Three consecutive TUAB epochs, labelled by centre.

    TUAB labels apply to whole recordings, so no window contains a
    within-window label transition by construction. This is what makes TUAB a
    specificity control: a flat boundary output is the correct response.
    """

    def __init__(self, epochs, labels, indices, n_context=3, verbose=True):
        self.epochs = epochs
        self.labels = labels
        self.n_ctx = n_context
        self.windows = []
        sorted_idx = np.sort(indices)
        for i in range(len(sorted_idx) - n_context + 1):
            group = sorted_idx[i:i + n_context]
            if np.all(np.diff(group) == 1):
                centre = n_context // 2
                self.windows.append({
                    'indices': group,
                    'label': int(labels[group[centre]]),
                    'epoch_labels': [int(labels[g]) for g in group],
                })
        if verbose:
            wl = [w['label'] for w in self.windows]
            print(f"  {len(self.windows)} windows "
                  f"({sum(wl)} abnormal, {len(wl) - sum(wl)} normal)")

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, idx):
        w = self.windows[idx]
        x = np.stack([self.epochs[i] for i in w['indices']], axis=0)
        return {
            'epoch': torch.tensor(x, dtype=torch.float32),
            'label': torch.tensor(w['label'], dtype=torch.long),
            'epoch_labels': torch.tensor(w['epoch_labels'],
                                         dtype=torch.long),
        }


def get_class_weighted_sampler(dataset):
    labels = [w['label'] for w in dataset.windows]
    counts = np.bincount(labels)
    weights = 1.0 / counts
    return WeightedRandomSampler([weights[label] for label in labels],
                                 len(labels))


def create_sleep_edf_dataloaders(h5_path, batch_size=16, split_seed=42,
                                 n_channels=3, n_samples=3000,
                                 verbose=True):
    """Split Sleep-EDF by the identifier in `subject_ids`, then build windows.

    The split is 70/15/15 over unique identifiers, shuffled with `split_seed`.
    Holding `split_seed` fixed across training seeds means run-to-run variance
    reflects initialisation and batch order on the same held-out set.

    See docs/reproducibility.md on what the stored identifiers represent.
    """
    h5_path = Path(h5_path)
    rng = np.random.RandomState(split_seed)
    with h5py.File(h5_path, 'r') as f:
        labels = f['labels'][:]
        subject_ids = f['subject_ids'][:]
        if isinstance(subject_ids[0], (bytes, np.bytes_)):
            subject_ids = np.array([
                s.decode() if isinstance(s, bytes) else s
                for s in subject_ids])
    unique_subjects = np.unique(subject_ids)
    rng.shuffle(unique_subjects)
    n_subjects = len(unique_subjects)
    n_train = int(0.7 * n_subjects)
    n_val = int(0.15 * n_subjects)
    splits = {
        'train': set(unique_subjects[:n_train]),
        'val': set(unique_subjects[n_train:n_train + n_val]),
        'test': set(unique_subjects[n_train + n_val:]),
    }
    idx = {name: np.where(np.isin(subject_ids, list(s)))[0]
           for name, s in splits.items()}
    if verbose:
        print(f"Split: {len(splits['train'])} train, {len(splits['val'])} "
              f"val, {len(splits['test'])} test identifiers")
        for name in ('train', 'val', 'test'):
            dist = dict(zip(*np.unique(labels[idx[name]],
                                       return_counts=True)))
            print(f"  {name}: {len(idx[name])} epochs, {dist}")
    datasets = {
        name: MultiEpochDataset(h5_path, target_channels=n_channels,
                                target_samples=n_samples,
                                indices=idx[name], context_size=1,
                                verbose=verbose)
        for name in ('train', 'val', 'test')}
    train_ld = DataLoader(datasets['train'], batch_size,
                          sampler=datasets['train'].get_sampler(),
                          num_workers=0, drop_last=True)
    val_ld = DataLoader(datasets['val'], batch_size, shuffle=False,
                        num_workers=0)
    test_ld = DataLoader(datasets['test'], batch_size, shuffle=False,
                         num_workers=0)
    return train_ld, val_ld, test_ld


def load_chbmit_data(h5_path, verbose=True):
    with h5py.File(h5_path, 'r') as f:
        epochs = f['epochs'][:]
        labels = f['labels'][:]
        subject_ids = np.array([s.decode() for s in f['subject_ids'][:]])
    subj_sz = {s: int(np.sum(labels[subject_ids == s] == 1))
               for s in np.unique(subject_ids)}
    if verbose:
        print(f"Loaded: {epochs.shape[0]} epochs, {epochs.shape[1]} channels")
        print(f"  Seizure: {np.sum(labels == 1)}, "
              f"Normal: {np.sum(labels == 0)}")
    return epochs, labels, subject_ids, subj_sz


def round_robin_split(subj_sz, verbose=True):
    """Deal subjects into train/val/test by descending seizure count.

    The pattern is train, train, val, test repeated, which balances seizure
    counts across splits. The result is deterministic and does not depend on
    any seed, so it is fixed by construction across multi-seed runs.

    On the published file this yields:
        train chb15, chb12, chb01, chb03, chb14, chb17
        val   chb08, chb10, chb22
        test  chb05, chb20, chb19
    giving 1,496 test windows of which 53 are seizure. A different subject
    list means the HDF5 file does not match the published run.
    """
    sorted_s = sorted(subj_sz.keys(), key=lambda s: subj_sz[s], reverse=True)
    splits = {'train': [], 'val': [], 'test': []}
    pattern = ['train', 'train', 'val', 'test'] * 4
    for i, s in enumerate(sorted_s):
        splits[pattern[i % len(pattern)]].append(s)
    if verbose:
        for name, subjs in splits.items():
            total = sum(subj_sz[s] for s in subjs)
            print(f"  {name}: {subjs} ({total} seizure epochs)")
    return splits


_CHB_CACHE = {}


def create_chbmit_dataloaders(h5_path, batch_size=32, n_context=3,
                              verbose=True):
    key = str(h5_path)
    if key not in _CHB_CACHE:
        _CHB_CACHE[key] = load_chbmit_data(Path(h5_path), verbose=verbose)
    epochs, labels, subject_ids, subj_sz = _CHB_CACHE[key]
    splits = round_robin_split(subj_sz, verbose=verbose)
    datasets = {
        name: CHBMITOnsetDataset(epochs, labels, subject_ids, splits[name],
                                 n_context, verbose=verbose)
        for name in ('train', 'val', 'test')}
    train_ld = DataLoader(datasets['train'], batch_size=batch_size,
                          sampler=get_class_weighted_sampler(
                              datasets['train']),
                          num_workers=0)
    val_ld = DataLoader(datasets['val'], batch_size=batch_size,
                        shuffle=False, num_workers=0)
    test_ld = DataLoader(datasets['test'], batch_size=batch_size,
                         shuffle=False, num_workers=0)
    return train_ld, val_ld, test_ld


def load_tuab_all(processed_dir, max_train_chunks=None, verbose=True):
    """Load the TUAB eval file and a prefix of the train chunks.

    `max_train_chunks` exists because the full corpus does not fit in memory.
    Chunks are consumed in filename order, which follows the order files were
    discovered. See docs/reproducibility.md on what that ordering implies.
    """
    processed_dir = Path(processed_dir)
    all_epochs, all_labels = [], []
    eval_path = processed_dir / 'tuab_eval_processed.h5'
    if eval_path.exists():
        with h5py.File(eval_path, 'r') as f:
            all_epochs.append(f['epochs'][:])
            all_labels.append(f['labels'][:])
        if verbose:
            print(f"  Eval: {len(all_labels[-1])} epochs")
    chunks = sorted(processed_dir.glob('tuab_train_chunk*.h5'))
    if max_train_chunks is not None:
        chunks = chunks[:max_train_chunks]
    for cp in chunks:
        with h5py.File(cp, 'r') as f:
            all_epochs.append(f['epochs'][:])
            all_labels.append(f['labels'][:])
    epochs = np.concatenate(all_epochs, axis=0)
    labels = np.concatenate(all_labels, axis=0)
    if verbose:
        print(f"Combined: {len(labels)} epochs "
              f"({np.sum(labels == 0)} normal, {np.sum(labels == 1)} "
              f"abnormal) from {len(chunks)} train chunks")
    return epochs, labels


def create_tuab_dataloaders(processed_dir, batch_size=32, split_seed=42,
                            n_channels=19, max_train_chunks=7, n_context=3,
                            verbose=True):
    """Random 60/20/20 split over epochs, then build windows.

    The split is over epoch indices rather than recordings. Because a window
    needs three consecutive surviving indices, only a small fraction of
    validation and test epochs form windows, and those that do come from
    recordings also represented in training. Both effects are quantified in
    docs/reproducibility.md and are why the paper treats TUAB as a
    specificity control rather than a performance result.
    """
    epochs, labels = load_tuab_all(processed_dir, max_train_chunks, verbose)
    if epochs.shape[1] < n_channels:
        pad = np.zeros((len(epochs), n_channels - epochs.shape[1],
                        epochs.shape[2]), dtype=epochs.dtype)
        epochs = np.concatenate([epochs, pad], axis=1)
    elif epochs.shape[1] > n_channels:
        epochs = epochs[:, :n_channels, :]
    rng = np.random.RandomState(split_seed)
    idx = rng.permutation(len(labels))
    n_train = int(0.60 * len(labels))
    n_val = int(0.20 * len(labels))
    parts = {'train': idx[:n_train],
             'val': idx[n_train:n_train + n_val],
             'test': idx[n_train + n_val:]}
    datasets = {name: TUABWindowDataset(epochs, labels, part, n_context,
                                        verbose=verbose)
                for name, part in parts.items()}
    train_ld = DataLoader(datasets['train'], batch_size=batch_size,
                          sampler=get_class_weighted_sampler(
                              datasets['train']),
                          num_workers=0)
    val_ld = DataLoader(datasets['val'], batch_size=batch_size,
                        shuffle=False, num_workers=0)
    test_ld = DataLoader(datasets['test'], batch_size=batch_size,
                         shuffle=False, num_workers=0)
    return train_ld, val_ld, test_ld
