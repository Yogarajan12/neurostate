"""Three-phase ACBL training loop and multi-seed runner."""

import json
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from .data import (
    create_chbmit_dataloaders,
    create_sleep_edf_dataloaders,
    create_tuab_dataloaders,
)
from .evaluate import evaluate
from .isolation import PhaseSchedule
from .losses import ACBLLoss
from .model import MultiResContrastiveNeuroState, forward_with_intermediates

__all__ = ["set_all_seeds", "build_model", "build_dataloaders", "train_one",
           "run_seeds", "aggregate"]


def set_all_seeds(seed):
    """Seed every source of randomness and disable non-deterministic kernels.

    cudnn.benchmark is disabled so that convolution algorithm selection does
    not vary between runs; this costs some speed and buys reproducibility.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def build_model(cfg, device='cuda', verbose=False):
    return MultiResContrastiveNeuroState(
        n_channels=cfg['n_channels'], n_samples=cfg['n_samples'],
        n_classes=cfg['n_classes'], embed_dim=cfg['embed_dim'],
        n_layers=cfg['n_layers'], dropout=cfg['dropout'],
        n_intra=cfg['n_intra'], n_inter=cfg['n_inter'],
        n_cross=cfg['n_cross'],
        contrast_scales=tuple(cfg['contrast_scales']),
        cp_hidden=cfg['cp_hidden'],
        n_context_epochs=cfg['n_context_epochs'],
        verbose=verbose).to(device)


def build_dataloaders(cfg, verbose=True):
    loader = cfg.get('loader', 'sleep_edf')
    if loader == 'chbmit':
        return create_chbmit_dataloaders(
            cfg['h5_path'], batch_size=cfg['batch_size'],
            n_context=cfg['n_context_epochs'], verbose=verbose)
    if loader == 'tuab':
        return create_tuab_dataloaders(
            cfg['processed_dir'], batch_size=cfg['batch_size'],
            split_seed=cfg['split_seed'], n_channels=cfg['n_channels'],
            max_train_chunks=cfg.get('max_train_chunks', 7),
            n_context=cfg['n_context_epochs'], verbose=verbose)
    return create_sleep_edf_dataloaders(
        cfg['h5_path'], batch_size=cfg['batch_size'],
        split_seed=cfg['split_seed'], n_channels=cfg['n_channels'],
        n_samples=cfg['n_samples'], verbose=verbose)


def _build_optimiser(model, cfg):
    """Sleep-EDF gives the boundary head its own parameter group with a
    higher learning rate and lower weight decay; CHB-MIT and TUAB use a
    single group. This mirrors the per-dataset recipes of the published runs.
    """
    if cfg.get('single_param_group', False):
        return torch.optim.AdamW(model.parameters(), lr=cfg['lr_other'],
                                 weight_decay=cfg['wd_other'])
    boundary_params, other_params = [], []
    for pname, p in model.named_parameters():
        (boundary_params if 'changepoint' in pname
         else other_params).append(p)
    return torch.optim.AdamW([
        {'params': other_params, 'lr': cfg['lr_other'],
         'weight_decay': cfg['wd_other']},
        {'params': boundary_params, 'lr': cfg['lr_boundary'],
         'weight_decay': cfg['wd_boundary']},
    ])


def _build_criterion(cfg, train_ld, device):
    if not cfg.get('class_weighted_loss', False):
        return nn.CrossEntropyLoss()
    train_labels = [w['label'] for w in train_ld.dataset.windows]
    n0 = sum(1 for label in train_labels if label == 0)
    n1 = sum(1 for label in train_labels if label == 1)
    cw = torch.tensor([len(train_labels) / (2 * n0),
                       len(train_labels) / (2 * n1)],
                      dtype=torch.float32).to(device)
    return nn.CrossEntropyLoss(weight=cw)


def train_one(cfg, seed, device='cuda', verbose=False, ckpt_dir=None):
    """One full three-phase run at a given training seed.

    The data split is controlled by the config and held fixed, so variance
    across seeds reflects initialisation and batch order rather than a
    different held-out set.

    Model selection uses validation classification loss only. Test data is
    never consulted during training or selection. Classification metrics and
    boundary statistics are both computed from the restored best checkpoint,
    so the two cannot come from different points in training.
    """
    set_all_seeds(seed)
    train_ld, val_ld, test_ld = build_dataloaders(cfg, verbose=verbose)
    model = build_model(cfg, device=device, verbose=verbose)
    acbl_loss = ACBLLoss(
        tokens_per_epoch=model.tokens_per_epoch,
        n_epochs=model.n_context,
        pseudo_weight=cfg['pseudo_weight'],
        prior_weight=cfg['prior_weight'],
        sigma=cfg['sigma'],
        pseudo_temperature=cfg['pseudo_temperature'],
        pseudo_temp_min=cfg['pseudo_temp_min'],
        pseudo_temp_anneal_epochs=cfg['pseudo_temp_anneal_epochs'],
        attn_prior_weight=cfg['attn_prior_weight'],
        variance_weight=cfg['variance_weight']).to(device)
    optimiser = _build_optimiser(model, cfg)
    scheduler = None
    if cfg.get('use_plateau_scheduler', True):
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimiser, mode='min', factor=0.5, patience=5)
    criterion = _build_criterion(cfg, train_ld, device)
    schedule = PhaseSchedule(cfg['warmup_epochs'], cfg['formation_epochs'])

    best_val = float('inf')
    best_state, best_epoch, wait = None, -1, 0
    trace = []

    for epoch in range(cfg['n_epochs']):
        phase = schedule.phase(epoch)
        use_acbl = schedule.use_auxiliary_loss(epoch)
        detach_bnd = schedule.detach_boundaries(epoch)
        if phase == 'full' and trace and trace[-1]['phase'] == 'formation':
            wait = 0
            best_val = float('inf')

        model.train()
        tr_cls, tr_n = 0.0, 0
        for batch in train_ld:
            x = batch['epoch'].to(device)
            y = batch['label'].to(device)
            elabels = batch['epoch_labels'].to(device)
            optimiser.zero_grad()
            if use_acbl:
                out = forward_with_intermediates(
                    model, x, detach_boundaries=detach_bnd)
                cls_loss = criterion(out['logits'], y)
                acbl_results = acbl_loss(
                    out['boundary_probs'], out['encoder_h'],
                    out['regime_attention'], elabels, epoch)
                loss = (cfg['cls_weight'] * cls_loss +
                        cfg['acbl_weight'] * acbl_results['acbl_total'] +
                        out['boundary_loss'])
            else:
                out = model(x, return_boundaries=True)
                cls_loss = criterion(out['logits'], y)
                loss = cls_loss + out['boundary_loss']
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimiser.step()
            tr_cls += cls_loss.item() * len(y)
            tr_n += len(y)

        model.eval()
        v_cls, v_n, v_correct = 0.0, 0, 0
        v_bounds = []
        with torch.no_grad():
            for batch in val_ld:
                x = batch['epoch'].to(device)
                y = batch['label'].to(device)
                out = model(x, return_boundaries=True)
                v_cls += criterion(out['logits'], y).item() * len(y)
                v_correct += (out['logits'].argmax(1) == y).sum().item()
                v_n += len(y)
                v_bounds.append(out['boundaries'].cpu().numpy())
        val_cls_loss = v_cls / v_n
        bounds_arr = np.concatenate(v_bounds)
        if scheduler is not None:
            scheduler.step(val_cls_loss)
        trace.append({
            'epoch': epoch, 'phase': phase,
            'train_cls': tr_cls / tr_n, 'val_cls': val_cls_loss,
            'val_acc': v_correct / v_n,
            'bnd_mean': float(bounds_arr.mean()),
            'bnd_std': float(bounds_arr.std()),
        })
        if val_cls_loss < best_val:
            best_val = val_cls_loss
            best_state = {k: v.cpu().clone()
                          for k, v in model.state_dict().items()}
            best_epoch = epoch
            wait = 0
        else:
            wait += 1
        if verbose:
            print(f"  seed {seed} ep {epoch:3d} [{phase:9s}] "
                  f"val_cls={val_cls_loss:.4f} "
                  f"val_acc={v_correct / v_n:.4f} "
                  f"bnd_std={bounds_arr.std():.4f}")
        if wait >= cfg['patience'] and epoch >= schedule.min_epochs:
            break

    if best_state:
        model.load_state_dict(best_state)
        model.to(device)
    if ckpt_dir is not None:
        ckpt_dir = Path(ckpt_dir)
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(),
                   ckpt_dir / f"{cfg['name']}_seed{seed}.pt")
    results = evaluate(model, test_ld, device=device, binary=cfg['binary'])
    results.update({
        'best_epoch': int(best_epoch), 'seed': int(seed),
        'split_seed': int(cfg['split_seed']),
        'stopped_epoch': int(trace[-1]['epoch']),
    })
    return results, trace


def run_seeds(cfg, seeds=(42, 43, 44, 45, 46), device='cuda',
              out_dir='results/runs', tag=None, verbose=False):
    """Run every seed, saving after each so an interrupted session never
    loses completed work. Re-running the same call resumes.
    """
    tag = tag or cfg['name']
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    results_path = out_dir / f'{tag}_results.json'
    completed = {}
    if results_path.exists():
        completed = {int(r['seed']): r
                     for r in json.loads(results_path.read_text())}
        print(f"Resuming: seeds {sorted(completed)} already done")
    for seed in seeds:
        if seed in completed:
            continue
        print(f"Running {tag} seed {seed}")
        t0 = time.time()
        try:
            res, trace = train_one(cfg, seed, device=device, verbose=verbose,
                                   ckpt_dir=out_dir / 'checkpoints')
        except Exception as exc:
            print(f"Seed {seed} failed: {exc}")
            continue
        res['runtime_sec'] = round(time.time() - t0, 1)
        completed[seed] = res
        (out_dir / f'{tag}_trace_seed{seed}.json').write_text(
            json.dumps(trace, indent=2))
        results_path.write_text(
            json.dumps([completed[s] for s in sorted(completed)], indent=2))
        summary = (f"  acc={res['accuracy']:.4f} kappa={res['kappa']:.4f} "
                   f"bnd_std={res['boundary_std']:.4f} "
                   f"best_ep={res['best_epoch']} ({res['runtime_sec']:.0f}s)")
        if cfg['binary']:
            summary += f" auroc={res['auroc']:.4f}"
        print(summary)
    return [completed[s] for s in sorted(completed)]


def aggregate(results, keys=None):
    """Mean and sample standard deviation across seeds for each metric."""
    if not results:
        return {}
    if keys is None:
        keys = [k for k, v in results[0].items()
                if isinstance(v, (int, float)) and k not in
                ('seed', 'split_seed', 'best_epoch', 'stopped_epoch',
                 'runtime_sec')]
    agg = {}
    for k in keys:
        vals = np.array([r[k] for r in results if k in r], dtype=float)
        agg[k] = {
            'mean': float(vals.mean()),
            'std': float(vals.std(ddof=1)) if len(vals) > 1 else 0.0,
            'n': int(len(vals)),
            'values': [float(v) for v in vals],
        }
    return agg
