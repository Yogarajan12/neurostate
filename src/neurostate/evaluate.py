"""Test-set evaluation.

Classification metrics and boundary statistics are computed from whatever
model state is currently loaded. Taking both from the same state removes the
checkpoint mismatch that would otherwise arise from reporting classification
at the validation-selected epoch and boundary statistics at the final epoch.
"""

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    recall_score,
    roc_auc_score,
)

__all__ = ["evaluate"]


def evaluate(model, test_ld, device='cuda', binary=False):
    """Return classification metrics and pooled boundary statistics.

    `boundary_std` is the standard deviation over all boundary values from all
    test windows pooled together. The paper labels a value below 0.020 a
    degenerate boundary distribution, as a descriptive cutoff calibrated
    against a shuffled-input null rather than as a statistical test.
    docs/results.md records that null, the two variants that sit close enough
    to the line for the statistic not to settle them, and the conditions under
    which the pooled statistic is uninformative.
    """
    model.eval()
    model.to(device)
    all_preds, all_labels, all_probs, all_bounds = [], [], [], []
    with torch.no_grad():
        for batch in test_ld:
            x = batch['epoch'].to(device)
            y = batch['label']
            out = model(x, return_boundaries=True)
            logits = out['logits']
            all_preds.extend(logits.argmax(1).cpu().numpy())
            all_labels.extend(y.numpy())
            if binary:
                probs = torch.softmax(logits, dim=1)[:, 1]
                all_probs.extend(probs.cpu().numpy())
            all_bounds.append(out['boundaries'].cpu().numpy())
    preds = np.array(all_preds)
    labels = np.array(all_labels)
    bounds = np.concatenate(all_bounds)
    res = {
        'accuracy': float(accuracy_score(labels, preds)),
        'balanced_accuracy': float(balanced_accuracy_score(labels, preds)),
        'f1_macro': float(f1_score(labels, preds, average='macro',
                                   zero_division=0)),
        'kappa': float(cohen_kappa_score(labels, preds)),
        'boundary_mean': float(bounds.mean()),
        'boundary_std': float(bounds.std()),
        'boundary_max': float(bounds.max()),
        'boundary_frac_05': float((bounds > 0.5).mean()),
    }
    if binary:
        res['auroc'] = float(roc_auc_score(labels, np.array(all_probs)))
        res['sensitivity'] = float(recall_score(labels, preds, pos_label=1,
                                                zero_division=0))
        res['specificity'] = float(recall_score(labels, preds, pos_label=0,
                                                zero_division=0))
        tn, fp, fn, tp = confusion_matrix(labels, preds).ravel()
        res['confusion'] = [int(tn), int(fp), int(fn), int(tp)]
    return res
