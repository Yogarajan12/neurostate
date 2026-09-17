"""Attention-Contrastive Boundary Learning (ACBL) losses.

Four terms shape the boundary head, matching equation 4 of the camera-ready
paper:

    L_ACBL = a * L_pseudo + b * (L_Gauss + g * L_KL) + d * L_var

with a = 0.3, b = 0.5, g = 0.5, d = 1.0, applied at lambda = 0.3. The
consistency term inside the boundary module (weight 0.01) sits outside this
objective and is added separately in the training loop.

Terminology note: the camera-ready uses delta for the variance weight. The
conference poster uses nu for the same quantity. This package follows the
paper.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["PseudoBoundaryLoss", "AttentionPriorLoss", "ACBLLoss"]


class PseudoBoundaryLoss(nn.Module):
    """Self-supervised boundary signal derived from encoder representations.

    Cosine distance between adjacent tokens is min-max normalised per window
    and passed through a sigmoid with an annealed temperature to give soft
    targets. Targets are detached, which stops the encoder from trivially
    maximising adjacent dissimilarity to satisfy the boundary head.

    Positions at epoch junctions are masked out of the distance computation,
    so this term carries no information about where the junctions are. It
    responds to representational change only.
    """

    def __init__(self, temperature=2.0, tokens_per_epoch=196, n_epochs=3):
        super().__init__()
        self.temperature = temperature
        self.tokens_per_epoch = tokens_per_epoch
        self.n_epochs = n_epochs
        t = tokens_per_epoch * n_epochs
        mask = torch.ones(t - 1)
        for e in range(1, n_epochs):
            mask[e * tokens_per_epoch - 1] = 0.0
        self.register_buffer('epoch_boundary_mask', mask)

    def compute_pseudo_targets(self, h):
        h_det = h.detach()
        h_norm = F.normalize(h_det, dim=-1)
        cos_sim = (h_norm[:, :-1] * h_norm[:, 1:]).sum(dim=-1)
        cos_dist = (1.0 - cos_sim) * self.epoch_boundary_mask.to(h.device)
        d_min = cos_dist.min(dim=-1, keepdim=True).values
        d_max = cos_dist.max(dim=-1, keepdim=True).values
        d_range = (d_max - d_min).clamp(min=1e-6)
        cos_dist_norm = (cos_dist - d_min) / d_range
        pseudo = torch.sigmoid((cos_dist_norm - 0.5) * self.temperature)
        return F.pad(pseudo, (0, 1), mode='replicate')

    def forward(self, boundary_probs, h):
        pseudo_targets = self.compute_pseudo_targets(h)
        return F.binary_cross_entropy(
            boundary_probs.clamp(1e-6, 1 - 1e-6),
            pseudo_targets, reduction='mean')


class AttentionPriorLoss(nn.Module):
    """Label-derived boundary and attention supervision.

    Two components:

    L_Gauss places Gaussian peaks (sigma = 5 tokens) at epoch junctions where
    the epoch label changes, supervising the boundary head directly.

    L_KL compares the head-averaged attention of the final block against a
    block-diagonal prior built from the epoch labels (0.8 for same-stage token
    pairs, 0.2 otherwise, row normalised). This routes gradient to the
    boundary head through the attention mechanism rather than through its
    output.

    Both are averaged over the windows in a batch that contain at least one
    label transition; windows with no transition contribute nothing.

    This is the only ACBL term that uses labels, and it anchors the head to
    epoch junctions. The ablation in Table 3 shows that removing it collapses
    the boundary head entirely (0.0001 on Sleep-EDF), and the contiguity
    analysis in docs/reproducibility.md shows that what the head learns is
    tied more closely to junction position than to signal content.
    """

    def __init__(self, tokens_per_epoch=196, n_epochs=3,
                 sigma=5.0, attn_prior_weight=0.5):
        super().__init__()
        self.tokens_per_epoch = tokens_per_epoch
        self.n_epochs = n_epochs
        self.sigma = sigma
        self.attn_prior_weight = attn_prior_weight
        self.T = tokens_per_epoch * n_epochs

    def _build_boundary_targets(self, labels, device):
        b = labels.shape[0]
        t = self.T
        tpe = self.tokens_per_epoch
        trans_01 = (labels[:, 0] != labels[:, 1]).float()
        trans_12 = (labels[:, 1] != labels[:, 2]).float()
        has_transition = (trans_01 + trans_12) > 0
        positions = torch.arange(t, device=device, dtype=torch.float32)
        targets = torch.zeros(b, t, device=device)
        junction_01 = float(tpe) - 0.5
        junction_12 = float(2 * tpe) - 0.5
        gauss_01 = torch.exp(
            -0.5 * ((positions - junction_01) / self.sigma) ** 2)
        gauss_12 = torch.exp(
            -0.5 * ((positions - junction_12) / self.sigma) ** 2)
        targets += trans_01.unsqueeze(1) * gauss_01.unsqueeze(0)
        targets += trans_12.unsqueeze(1) * gauss_12.unsqueeze(0)
        t_max = targets.max(dim=-1, keepdim=True).values.clamp(min=1e-6)
        targets = targets / t_max
        targets = targets * has_transition.float().unsqueeze(1)
        return targets, has_transition

    def _build_attention_prior(self, labels, device):
        b = labels.shape[0]
        t = self.T
        tpe = self.tokens_per_epoch
        token_labels = torch.zeros(b, t, dtype=labels.dtype, device=device)
        for e in range(self.n_epochs):
            token_labels[:, e * tpe:(e + 1) * tpe] = labels[:, e].unsqueeze(1)
        same_stage = (token_labels.unsqueeze(2) ==
                      token_labels.unsqueeze(1)).float()
        prior = same_stage * 0.8 + (1 - same_stage) * 0.2
        prior = prior / prior.sum(dim=-1, keepdim=True)
        return prior

    def forward(self, boundary_probs, regime_attention, labels):
        device = boundary_probs.device
        boundary_targets, has_transition = self._build_boundary_targets(
            labels, device)
        n_transitions = has_transition.sum().item()
        results = {
            'n_transitions': n_transitions,
            'boundary_target_loss': torch.tensor(0.0, device=device),
            'attention_prior_loss': torch.tensor(0.0, device=device),
        }
        if n_transitions == 0:
            results['total'] = torch.tensor(0.0, device=device)
            return results
        mask = has_transition.float()
        bce = F.binary_cross_entropy(
            boundary_probs.clamp(1e-6, 1 - 1e-6),
            boundary_targets, reduction='none')
        bce_per_sample = bce.mean(dim=-1)
        results['boundary_target_loss'] = (
            (bce_per_sample * mask).sum() / mask.sum())
        if regime_attention is not None:
            if regime_attention.dim() == 4:
                attn = regime_attention.mean(dim=1)
            else:
                attn = regime_attention
            prior = self._build_attention_prior(labels, device)
            log_attn = torch.log(attn.clamp(min=1e-8))
            log_prior = torch.log(prior.clamp(min=1e-8))
            kl = prior * (log_prior - log_attn)
            kl_per_sample = kl.sum(dim=-1).mean(dim=-1)
            results['attention_prior_loss'] = (
                (kl_per_sample * mask).sum() / mask.sum())
        results['total'] = (
            results['boundary_target_loss'] +
            self.attn_prior_weight * results['attention_prior_loss'])
        return results


class ACBLLoss(nn.Module):
    """Combined ACBL objective with temperature annealing and the variance
    penalty.

    The variance penalty is the negative per-window variance of b(t), which
    penalises constant outputs directly. Table 1 of the paper shows it is
    necessary but not sufficient: adding it alone still collapses (0.007),
    because penalising flatness creates no gradient that distinguishes one
    boundary position from another. Only combining it with gradient isolation
    produces a non-degenerate head.

    Set `variance_weight=0.0` to reproduce the TUAB configuration, which ran
    three terms rather than four.
    """

    def __init__(self, tokens_per_epoch=196, n_epochs=3,
                 pseudo_weight=0.3, pseudo_temperature=2.0,
                 pseudo_temp_min=0.5, pseudo_temp_anneal_epochs=15,
                 prior_weight=0.5, sigma=5.0,
                 attn_prior_weight=0.5,
                 variance_weight=1.0):
        super().__init__()
        self.pseudo_weight = pseudo_weight
        self.prior_weight = prior_weight
        self.variance_weight = variance_weight
        self.pseudo_temp_min = pseudo_temp_min
        self.pseudo_temp_anneal_epochs = pseudo_temp_anneal_epochs
        self.pseudo_temperature_init = pseudo_temperature
        self.prong1 = PseudoBoundaryLoss(
            temperature=pseudo_temperature,
            tokens_per_epoch=tokens_per_epoch,
            n_epochs=n_epochs)
        self.prong2 = AttentionPriorLoss(
            tokens_per_epoch=tokens_per_epoch,
            n_epochs=n_epochs,
            sigma=sigma,
            attn_prior_weight=attn_prior_weight)

    def anneal_temperature(self, epoch):
        if self.pseudo_temp_anneal_epochs <= 0:
            return
        progress = min(epoch / self.pseudo_temp_anneal_epochs, 1.0)
        self.prong1.temperature = (
            self.pseudo_temperature_init -
            progress * (self.pseudo_temperature_init - self.pseudo_temp_min))

    def forward(self, boundary_probs, encoder_h, regime_attention,
                epoch_labels, epoch=0):
        self.anneal_temperature(epoch)
        pseudo_loss = self.prong1(boundary_probs, encoder_h)
        prior_results = self.prong2(
            boundary_probs, regime_attention, epoch_labels)
        var_per_sample = boundary_probs.var(dim=-1)
        variance_loss = -var_per_sample.mean()
        total = (self.pseudo_weight * pseudo_loss +
                 self.prior_weight * prior_results['total'] +
                 self.variance_weight * variance_loss)
        return {
            'acbl_total': total,
            'pseudo_boundary_loss': pseudo_loss,
            'boundary_target_loss': prior_results['boundary_target_loss'],
            'attention_prior_loss': prior_results['attention_prior_loss'],
            'variance_loss': variance_loss,
            'boundary_variance': var_per_sample.mean(),
            'n_transitions': prior_results['n_transitions'],
            'pseudo_temperature': self.prong1.temperature,
        }
