"""Why auxiliary boundary heads collapse: the useful-when-flat property.

Runs on CPU in about a minute and needs no dataset access, so the mechanism
behind the paper can be inspected without PhysioNet downloads or a TUH data
agreement.

The setup is a miniature of the real one. Signals are piecewise stationary:
each window is a sequence of segments with different frequencies, labelled by
which frequency dominates the centre. A small transformer classifies the
window while an auxiliary head predicts per-token changepoint probabilities,
and that output conditions attention through the same exp(-|C_i - C_j|)
regime mask used in NeuroState.

Three conditions are compared under identical seeds:

    frozen-flat   the boundary output is replaced by a constant, so the mask
                  is a fixed distance-decay filter carrying no information
    joint         the boundary head is trained jointly, no isolation
    isolated      the boundary head is detached from the classification path
                  during a formation phase

The result to read is the accuracy column. Classification with a deliberately
uninformative boundary output is about as good as with a learned one, because
a constant b(t) makes the cumulative sum linear and the mask depends only on
token distance, which is already a useful attention prior. The classifier
therefore gains nothing from an informative auxiliary head, and joint
training has no gradient incentive to produce one. That is the shortcut
pathology of Section 2 of the paper and the architectural cause of collapse.

Scope, stated plainly. This demo reproduces the *cause* of collapse, not
collapse itself. At this scale, with a strong variance penalty and a short
schedule, the joint and isolated runs both keep a non-degenerate boundary
head; the collapse reported in the paper emerges in the full setting, across
eleven interventions on real EEG. Read the accuracy comparison as the
mechanism, and Table 1 of the paper as the evidence. The paper scopes the
collapse and isolation findings to the EEG settings measured there, and this
synthetic reproduction is an illustration rather than additional evidence of
generality.

    python demo/collapse_demo.py
    python demo/collapse_demo.py --epochs 30 --plot demo/collapse.png
"""

import argparse

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from neurostate.isolation import PhaseSchedule, isolate

N_TOKENS = 64
N_CLASSES = 3
FREQUENCIES = [3.0, 7.0, 13.0]


def make_window(rng, n_samples=256):
    """One piecewise-stationary signal with two changepoints."""
    cuts = sorted(rng.choice(np.arange(20, n_samples - 20), size=2,
                             replace=False))
    segments, token_labels = [], []
    bounds = [0, cuts[0], cuts[1], n_samples]
    classes = rng.choice(N_CLASSES, size=3, replace=True)
    for i in range(3):
        length = bounds[i + 1] - bounds[i]
        t = np.arange(length) / 50.0
        seg = np.sin(2 * np.pi * FREQUENCIES[classes[i]] * t)
        seg = seg + 0.3 * rng.standard_normal(length)
        segments.append(seg)
        token_labels.extend([classes[i]] * length)
    signal = np.concatenate(segments)
    token_labels = np.array(token_labels)
    centre = token_labels[n_samples // 2]
    true_cuts = np.array(cuts) / n_samples
    return signal.astype(np.float32), int(centre), true_cuts


def make_dataset(n, seed=0):
    rng = np.random.default_rng(seed)
    xs, ys, cuts = [], [], []
    for _ in range(n):
        x, y, c = make_window(rng)
        xs.append(x)
        ys.append(y)
        cuts.append(c)
    return (torch.tensor(np.stack(xs)).unsqueeze(1),
            torch.tensor(ys), np.stack(cuts))


class TinyBoundaryModel(nn.Module):
    """Encoder, auxiliary boundary head, and regime-masked attention."""

    def __init__(self, dim=32):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv1d(1, 16, 9, stride=2, padding=4), nn.GELU(),
            nn.Conv1d(16, dim, 9, stride=2, padding=4), nn.GELU())
        self.boundary = nn.Sequential(
            nn.Linear(dim, 16), nn.GELU(), nn.Linear(16, 1))
        self.qkv = nn.Linear(dim, 3 * dim)
        self.out_proj = nn.Linear(dim, dim)
        self.head = nn.Linear(dim, N_CLASSES)
        self.scale = dim ** -0.5

    def forward(self, x, detach_boundaries=False, freeze_flat=False):
        h = self.encoder(x).permute(0, 2, 1)
        b = torch.sigmoid(self.boundary(h).squeeze(-1))
        b_attn = torch.full_like(b, 0.5) if freeze_flat else isolate(
            b, active=detach_boundaries)
        cum = torch.cumsum(b_attn, dim=1)
        mask = torch.exp(-torch.abs(cum.unsqueeze(2) - cum.unsqueeze(1)))
        q, k, v = self.qkv(h).chunk(3, dim=-1)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn + torch.log(mask + 1e-6)
        out = F.softmax(attn, dim=-1) @ v
        out = self.out_proj(out)
        logits = self.head(out.mean(dim=1))
        return logits, b, h


def pseudo_targets(h, temperature=2.0):
    """Self-supervised boundary targets from representation contrast.

    Mirrors L_pseudo: cosine distance between adjacent encoder tokens,
    min-max normalised per window and passed through a sigmoid. Targets are
    detached so the encoder cannot satisfy the auxiliary head by inflating
    adjacent dissimilarity.

    Using self-supervised targets rather than ground-truth changepoints is
    what makes this demo faithful. Given ground-truth targets the auxiliary
    head learns under either condition and no collapse appears, because the
    supervision is strong enough to overcome any gradient pressure. ACBL's
    formation phase has no such luxury: sub-epoch changepoint annotations do
    not exist, which is precisely why the head is vulnerable to collapse.
    """
    h = F.normalize(h.detach(), dim=-1)
    cos_dist = 1.0 - (h[:, :-1] * h[:, 1:]).sum(dim=-1)
    d_min = cos_dist.min(dim=-1, keepdim=True).values
    d_max = cos_dist.max(dim=-1, keepdim=True).values
    norm = (cos_dist - d_min) / (d_max - d_min).clamp(min=1e-6)
    return F.pad(torch.sigmoid((norm - 0.5) * temperature), (0, 1),
                 mode='replicate')


def run(condition, epochs=25, seed=0, verbose=True):
    """condition is one of 'frozen-flat', 'joint', 'isolated'."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    x_tr, y_tr, _ = make_dataset(480, seed=seed)
    x_te, y_te, _ = make_dataset(160, seed=seed + 100)
    model = TinyBoundaryModel()
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3)
    schedule = PhaseSchedule(warmup_epochs=3, formation_epochs=8)
    history = []

    for epoch in range(epochs):
        detach = (condition == 'isolated'
                  and schedule.detach_boundaries(epoch))
        freeze = condition == 'frozen-flat'
        use_aux = schedule.use_auxiliary_loss(epoch) and not freeze
        model.train()
        perm = torch.randperm(len(x_tr))
        for i in range(0, len(x_tr), 32):
            idx = perm[i:i + 32]
            opt.zero_grad()
            logits, b, h = model(x_tr[idx], detach_boundaries=detach,
                                 freeze_flat=freeze)
            loss = F.cross_entropy(logits, y_tr[idx])
            if use_aux:
                aux = F.binary_cross_entropy(b.clamp(1e-6, 1 - 1e-6),
                                             pseudo_targets(h))
                loss = loss + 0.3 * (aux - b.var(dim=-1).mean())
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            logits, b, _ = model(x_te, freeze_flat=freeze)
            acc = (logits.argmax(1) == y_te).float().mean().item()
            std = b.std().item()
        history.append({'epoch': epoch, 'phase': schedule.phase(epoch),
                        'accuracy': acc, 'boundary_std': std})
        if verbose and epoch % 5 == 0:
            print(f"  epoch {epoch:2d} [{schedule.phase(epoch):9s}] "
                  f"acc={acc:.3f} boundary_std={std:.4f}")
    return model, history


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--epochs', type=int, default=25)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--plot', default=None,
                        help='optional path for a boundary-profile figure')
    args = parser.parse_args()

    runs = {}
    for condition in ('frozen-flat', 'joint', 'isolated'):
        print(f"{condition}:")
        model, history = run(condition, args.epochs, args.seed)
        runs[condition] = (model, history)
        print()

    print("Final test-set results")
    for condition, (_, history) in runs.items():
        h = history[-1]
        print(f"  {condition:12s} accuracy {h['accuracy']:.3f}  "
              f"boundary std {h['boundary_std']:.4f}")

    flat = runs['frozen-flat'][1][-1]['accuracy']
    joint = runs['joint'][1][-1]['accuracy']
    print(f"\nA deliberately uninformative boundary output reaches "
          f"{flat:.3f} against {joint:.3f} for a learned one, a gap of "
          f"{abs(joint - flat):.3f}.")
    print("The mask is useful even when flat, so the classification "
          "objective has almost no incentive to make the auxiliary head "
          "informative. That missing incentive is what drives collapse in "
          "the full setting; see docs/collapse_experiments.md.")

    if args.plot:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        x_te, _, cuts_te = make_dataset(4, seed=args.seed + 100)
        with torch.no_grad():
            _, b, _ = runs['isolated'][0](x_te)
        fig, axes = plt.subplots(2, 1, figsize=(7, 5))
        for label, (_, history) in runs.items():
            axes[0].plot([e['boundary_std'] for e in history], label=label)
        axes[0].axhline(0.02, ls='--', c='grey', lw=0.8)
        axes[0].set_ylabel('boundary std')
        axes[0].set_xlabel('epoch')
        axes[0].legend()
        n_tokens = b.shape[1]
        axes[1].plot(b[0].numpy())
        for c in cuts_te[0]:
            axes[1].axvline(c * n_tokens, c='red', ls='--', lw=0.8)
        axes[1].set_ylabel('b(t), isolated')
        axes[1].set_xlabel('token')
        fig.tight_layout()
        fig.savefig(args.plot, dpi=150)
        print(f"\nSaved figure to {args.plot}")


if __name__ == '__main__':
    main()
