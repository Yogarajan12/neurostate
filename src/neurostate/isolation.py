"""Gradient isolation: the phase schedule and a standalone utility.

The central claim of the paper is an optimisation one. An auxiliary head whose
output feeds back into the primary pathway will collapse to a near-constant
output whenever the primary task does not structurally require the auxiliary
signal. Detaching the auxiliary output on the path into the primary pathway,
while leaving the auxiliary objective's own gradient intact, is the only
intervention among eleven that avoided collapse.

The paper scopes this finding to the EEG settings measured here. What was
shown to carry across the four boundary parameterisations is the collapse,
not the remedy: the differentiable HMM and query-based variants produce no
comparable boundary statistic, so their degeneracy rests on accuracy and
inspection. The utility below is provided because the mechanism is
architecture-independent in principle, not because generality has been
demonstrated.
"""

from dataclasses import dataclass

__all__ = ["PhaseSchedule", "isolate"]


def isolate(auxiliary_output, active=True):
    """Apply a directional stop-gradient to an auxiliary head's output.

    Returns the tensor to feed into the primary pathway. Use the original,
    undetached tensor for the auxiliary objective.

        b = boundary_head(h)
        b_masked = isolate(b, active=schedule.detach_boundaries(epoch))
        logits = classifier(h, b_masked)
        loss = ce(logits, y) + lam * aux_loss(b, ...)

    With `active=True` the classification gradient cannot reach the auxiliary
    head, while the auxiliary loss still updates it. Detaching in both
    directions instead leaves the head untrainable and decoupled from the
    encoder; coupling in both directions reproduces collapse. The asymmetry is
    the mechanism.
    """
    return auxiliary_output.detach() if active else auxiliary_output


@dataclass
class PhaseSchedule:
    """Three-phase ACBL curriculum.

    Warmup (epochs 0 to warmup-1): classification loss only. The auxiliary
    head receives no signal, but the encoder develops representations for it
    to fit.

    Formation (warmup to warmup+formation-1): the auxiliary losses are active
    and the auxiliary output is detached on the path into the primary
    pathway.

    Full (warmup+formation onward): the auxiliary output is reconnected. The
    structure formed during the previous phase survives because the regime
    mask is now content-dependent rather than a passive distance filter.

    Early stopping is enabled only once the full phase begins, so a run cannot
    terminate during formation and be evaluated with the head still detached.
    The published runs use warmup=3, formation=12 on all three datasets, fixed
    in advance and never selected on test data.
    """

    warmup_epochs: int = 3
    formation_epochs: int = 12

    def phase(self, epoch):
        if epoch < self.warmup_epochs:
            return 'warmup'
        if epoch < self.warmup_epochs + self.formation_epochs:
            return 'formation'
        return 'full'

    def use_auxiliary_loss(self, epoch):
        return epoch >= self.warmup_epochs

    def detach_boundaries(self, epoch):
        return self.phase(epoch) == 'formation'

    @property
    def min_epochs(self):
        """First epoch at which early stopping may fire."""
        return self.warmup_epochs + self.formation_epochs + 1
