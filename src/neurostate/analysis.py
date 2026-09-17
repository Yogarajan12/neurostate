"""Analyses that test what the boundary output actually represents.

These are the checks that constrain the interpretability claims. Each one is
reported in the camera-ready, including where the result is negative. Running
them is the fastest way to see what the boundary head does and does not do.
"""

from collections import defaultdict

import numpy as np
import torch
from scipy.stats import mannwhitneyu

__all__ = ["null_boundary_std", "boundary_by_window_type",
           "boundary_by_contiguity", "centre_vs_surround",
           "contiguity_mask"]


def null_boundary_std(model, test_ld, device='cuda', n_shuffles=20):
    """Empirical null for the collapse threshold.

    Recomputes the pooled boundary standard deviation after shuffling sample
    order within each window, which destroys temporal structure while leaving
    the marginal signal distribution intact. The resulting value is what a
    structureless input produces.

    On Sleep-EDF the null sits at 0.0154 (95th percentile 0.0155) against an
    observed 0.0260 across five seeds, a ratio of 1.68. That margin justifies
    the 0.020 threshold, but it does not by itself show the head tracks
    physiology; read it alongside `boundary_by_contiguity`.
    """
    model.eval()
    model.to(device)
    stds = []
    with torch.no_grad():
        for _ in range(n_shuffles):
            batch_stds = []
            for batch in test_ld:
                x = batch['epoch'].to(device)
                perm = torch.randperm(x.shape[-1], device=x.device)
                out = model(x[..., perm], return_boundaries=True)
                batch_stds.append(out['boundaries'].cpu().numpy())
            stds.append(float(np.concatenate(batch_stds).std()))
    stds = np.array(stds)
    return {
        'null_mean': float(stds.mean()),
        'null_std': float(stds.std(ddof=1)),
        'null_p95': float(np.percentile(stds, 95)),
        'values': [float(v) for v in stds],
    }


def boundary_by_window_type(model, loader, device='cuda'):
    """Split boundary statistics by whether a window contains a label
    transition.

    The pooled statistic is diluted on tasks where most windows contain no
    within-window transition. On CHB-MIT only 59 of 1,496 test windows do, so
    the pooled figure is dominated by windows where a flat output is correct.

    If the head marked transitions, windows containing one would show a higher
    per-window standard deviation. Across five seeds they do not: the two
    populations are indistinguishable (Mann-Whitney p >= 0.21, Cohen's
    d = -0.25 +/- 0.20). This is a reported non-result, not a bug.
    """
    model.eval()
    model.to(device)
    trans_tokens, flat_tokens = [], []
    trans_win, flat_win = [], []
    trans_peak, flat_peak = [], []
    with torch.no_grad():
        for batch in loader:
            x = batch['epoch'].to(device)
            el = batch['epoch_labels'].numpy()
            out = model(x, return_boundaries=True)
            b = out['boundaries'].cpu().numpy()
            has_trans = el.max(axis=1) != el.min(axis=1)
            for i in range(b.shape[0]):
                if has_trans[i]:
                    trans_tokens.append(b[i])
                    trans_win.append(float(b[i].std()))
                    trans_peak.append(float(b[i].max()))
                else:
                    flat_tokens.append(b[i])
                    flat_win.append(float(b[i].std()))
                    flat_peak.append(float(b[i].max()))
    res = {'n_transition': len(trans_win), 'n_flat': len(flat_win)}
    if trans_tokens:
        res['pooled_std_transition'] = float(
            np.concatenate(trans_tokens).std())
        res['mean_window_std_transition'] = float(np.mean(trans_win))
        res['mean_peak_transition'] = float(np.mean(trans_peak))
    if flat_tokens:
        res['pooled_std_flat'] = float(np.concatenate(flat_tokens).std())
        res['mean_window_std_flat'] = float(np.mean(flat_win))
        res['mean_peak_flat'] = float(np.mean(flat_peak))
    if len(trans_win) >= 3 and len(flat_win) >= 3:
        u, p = mannwhitneyu(trans_win, flat_win, alternative='greater')
        res['mannwhitney_u'] = float(u)
        res['p_value'] = float(p)
        pooled_sd = np.sqrt((np.var(trans_win, ddof=1) +
                             np.var(flat_win, ddof=1)) / 2)
        if pooled_sd > 0:
            res['cohens_d'] = float(
                (np.mean(trans_win) - np.mean(flat_win)) / pooled_sd)
    return res


def contiguity_mask(dataset, h5_path, epoch_seconds=30.0):
    """True where both neighbours of the centre epoch are one epoch away in
    recording time.

    Requires a `times` array in the HDF5 file. Only Sleep-EDF stores one.
    """
    import h5py
    with h5py.File(h5_path, 'r') as f:
        times = f['times'][:]
    vi = dataset.valid_indices
    prev_gap = times[vi] - times[vi - 1]
    next_gap = times[vi + 1] - times[vi]
    return ((np.abs(prev_gap - epoch_seconds) < 1.0) &
            (np.abs(next_gap - epoch_seconds) < 1.0))


def boundary_by_contiguity(model, loader, mask, device='cuda'):
    """Compare boundary statistics on windows that are contiguous in
    recording time against windows containing a splice.

    On Sleep-EDF only 15.6 per cent of test windows are truly contiguous, and
    boundary standard deviation is 0.0221 on those against 0.0226 on spliced
    ones. If the head were responding to physiological change rather than to
    window position, these would differ. They do not, which is the strongest
    single constraint on the interpretability claims.

    The loader must be unshuffled and built from the same split as the mask.
    """
    model.eval()
    model.to(device)
    groups = {True: [], False: []}
    pos = 0
    with torch.no_grad():
        for batch in loader:
            out = model(batch['epoch'].to(device), return_boundaries=True)
            b = out['boundaries'].cpu().numpy()
            for i in range(b.shape[0]):
                groups[bool(mask[pos])].append(b[i])
                pos += 1
    if pos != len(mask):
        raise RuntimeError(
            f"mask length {len(mask)} does not match {pos} windows seen; "
            "the loader must be unshuffled and built from the same split")
    out_stats = {}
    for is_cont, name in ((True, 'contiguous'), (False, 'spliced')):
        g = groups[is_cont]
        if not g:
            continue
        out_stats[name] = {
            'n': len(g),
            'pooled_std': float(np.concatenate(g).std()),
            'mean_window_std': float(np.mean([float(w.std()) for w in g])),
        }
    return out_stats


def centre_vs_surround(model, loader, device='cuda'):
    """Compare mean boundary activation in the centre epoch against the two
    flanking epochs, split by class.

    On CHB-MIT the centre margin is positive in 98.9 +/- 1.7 per cent of
    seizure windows, but also in 99.9 +/- 0.2 per cent of non-seizure windows,
    and is larger in the latter (d = -0.82 +/- 0.27). The centre epoch is
    bracketed by two junctions while each flanking epoch borders only one, so
    the ordering follows from window geometry rather than from seizure
    content. An earlier version of this analysis was run on seizure windows
    only and read as evidence of a seizure effect; running both classes is
    what showed otherwise.
    """
    model.eval()
    model.to(device)
    by_class = defaultdict(list)
    with torch.no_grad():
        for batch in loader:
            x = batch['epoch'].to(device)
            labels = batch['label'].numpy()
            out = model(x, return_boundaries=True)
            b = out['boundaries'].cpu().numpy()
            tpe = model.tokens_per_epoch
            for i in range(b.shape[0]):
                centre = b[i][tpe:2 * tpe]
                surround = np.concatenate([b[i][:tpe], b[i][2 * tpe:]])
                by_class[int(labels[i])].append(
                    float(centre.mean() - surround.mean()))
    res = {}
    for cls, deltas in by_class.items():
        if not deltas:
            continue
        res[f'class_{cls}'] = {
            'n': len(deltas),
            'mean_delta': float(np.mean(deltas)),
            'frac_positive': float(np.mean([d > 0 for d in deltas])),
        }
    # Cohen's d is defined only for the binary contrast this analysis was
    # written for; on a multi-class task the per-class margins are still
    # reported but the effect size is omitted.
    if len(by_class) == 2 and all(len(v) > 1 for v in by_class.values()):
        pos, neg = by_class[1], by_class[0]
        pooled_sd = np.sqrt((np.var(pos, ddof=1) + np.var(neg, ddof=1)) / 2)
        if pooled_sd > 0:
            res['cohens_d'] = float(
                (np.mean(pos) - np.mean(neg)) / pooled_sd)
    return res
