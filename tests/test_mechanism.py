"""Tests for the two claims the paper rests on.

These check the mechanism itself rather than only that code runs: that a flat
boundary output degenerates the regime mask into distance decay, and that
gradient isolation blocks exactly one direction of gradient flow.
"""

import pytest
import torch

from neurostate.isolation import PhaseSchedule, isolate
from neurostate.losses import ACBLLoss
from neurostate.model import (
    MultiResContrastiveNeuroState,
    RegimeStructuredAttention,
    forward_with_intermediates,
)


@pytest.fixture
def small_model():
    """A small model with real token geometry but few samples, so the tests
    run on CPU in seconds.
    """
    torch.manual_seed(0)
    return MultiResContrastiveNeuroState(
        n_channels=3, n_samples=600, n_classes=5, embed_dim=32,
        n_layers=2, n_intra=2, n_inter=1, n_cross=1, cp_hidden=16,
        n_context_epochs=3)


def test_flat_boundaries_give_distance_decay_mask():
    """The useful-when-flat property that causes collapse.

    With a constant b(t), the cumulative sum is linear in position, so
    S_ij = exp(-|C_i - C_j|) depends only on |i - j|. The mask becomes a
    passive distance filter that is identical for every input, which already
    supports above-0.70 accuracy on Sleep-EDF. The classifier therefore gains
    nothing from an informative boundary head, which is why joint training
    drives it to a constant.
    """
    attn = RegimeStructuredAttention(embed_dim=32, n_intra=2, n_inter=1,
                                     n_cross=1)
    t = 12
    flat = torch.full((1, t), 0.3)
    mask = attn._build_regime_mask(flat)[0]

    for offset in range(t):
        diagonal = torch.diagonal(mask, offset=offset)
        assert torch.allclose(diagonal, diagonal[0], atol=1e-6), (
            "a constant boundary output must give a mask that depends only "
            "on token distance")

    other = attn._build_regime_mask(torch.full((1, t), 0.8))[0]
    assert not torch.allclose(mask, other, atol=1e-3), (
        "different constant values should still scale the decay rate")

    structured = torch.full((1, t), 0.1)
    structured[0, t // 2] = 0.9
    varied = attn._build_regime_mask(structured)[0]
    diag = torch.diagonal(varied, offset=3)
    assert diag.std() > 1e-3, (
        "a non-constant boundary output must break distance invariance")


def test_gradient_isolation_blocks_only_classification_path(small_model):
    """Gradient isolation is directional.

    During formation, the classification loss must not reach the boundary
    head, while the ACBL losses still must. Detaching both directions would
    leave the head untrainable; coupling both reproduces collapse. The
    asymmetry is the mechanism, so it is worth asserting exactly.
    """
    model = small_model
    x = torch.randn(2, 3, 3, 600)
    y = torch.tensor([0, 1])
    boundary_params = [p for n, p in model.named_parameters()
                       if 'changepoint' in n]
    assert boundary_params, "expected parameters in the boundary module"

    model.zero_grad()
    out = forward_with_intermediates(model, x, detach_boundaries=True)
    torch.nn.functional.cross_entropy(out['logits'], y).backward()
    for p in boundary_params:
        assert p.grad is None or torch.all(p.grad == 0), (
            "classification loss reached the boundary head during formation")

    model.zero_grad()
    out = forward_with_intermediates(model, x, detach_boundaries=True)
    out['boundary_probs'].var(dim=-1).mean().neg().backward()
    assert any(p.grad is not None and torch.any(p.grad != 0)
               for p in boundary_params), (
        "the auxiliary objective must still update the boundary head")

    model.zero_grad()
    out = forward_with_intermediates(model, x, detach_boundaries=False)
    torch.nn.functional.cross_entropy(out['logits'], y).backward()
    assert any(p.grad is not None and torch.any(p.grad != 0)
               for p in boundary_params), (
        "outside formation the classification gradient must flow again")


def test_isolate_utility_matches_model_behaviour():
    """The standalone utility does the same thing as the inline detach.

    The gradient is routed through a parameter, as it would be in a real
    auxiliary head, rather than through a leaf tensor.
    """
    head = torch.nn.Linear(4, 1)
    h = torch.randn(2, 4)

    head.zero_grad()
    isolated = isolate(torch.sigmoid(head(h)), active=True)
    assert not isolated.requires_grad, (
        "an isolated auxiliary output must carry no gradient history")
    (isolated.sum() + 0.0 * head.weight.sum()).backward()
    assert torch.all(head.weight.grad == 0), (
        "no gradient may reach the head through an isolated output")

    head.zero_grad()
    connected = isolate(torch.sigmoid(head(h)), active=False)
    assert connected.requires_grad
    connected.sum().backward()
    assert torch.any(head.weight.grad != 0), (
        "with isolation off the gradient must reach the head again")


def test_phase_schedule_matches_published_curriculum():
    schedule = PhaseSchedule(warmup_epochs=3, formation_epochs=12)
    assert schedule.phase(0) == 'warmup'
    assert schedule.phase(2) == 'warmup'
    assert schedule.phase(3) == 'formation'
    assert schedule.phase(14) == 'formation'
    assert schedule.phase(15) == 'full'
    assert not schedule.use_auxiliary_loss(2)
    assert schedule.use_auxiliary_loss(3)
    assert schedule.detach_boundaries(14)
    assert not schedule.detach_boundaries(15)
    assert schedule.min_epochs == 16, (
        "early stopping must not fire before the full phase begins")


def test_acbl_variance_term_penalises_constant_output(small_model):
    """The variance penalty should be at its worst for a constant output."""
    loss_fn = ACBLLoss(tokens_per_epoch=small_model.tokens_per_epoch,
                       n_epochs=3)
    h = torch.randn(2, small_model.tokens_per_epoch * 3, 32)
    labels = torch.tensor([[0, 1, 1], [2, 2, 3]])
    flat = torch.full((2, small_model.tokens_per_epoch * 3), 0.5)
    varied = torch.rand(2, small_model.tokens_per_epoch * 3)
    flat_var = loss_fn(flat, h, None, labels)['variance_loss']
    varied_var = loss_fn(varied, h, None, labels)['variance_loss']
    assert flat_var > varied_var, (
        "a constant boundary output must incur the larger variance penalty")


def test_acbl_skips_windows_without_transitions(small_model):
    """Label-derived terms contribute nothing when no window has a
    transition, which is the situation on TUAB by construction."""
    loss_fn = ACBLLoss(tokens_per_epoch=small_model.tokens_per_epoch,
                       n_epochs=3)
    h = torch.randn(2, small_model.tokens_per_epoch * 3, 32)
    b = torch.rand(2, small_model.tokens_per_epoch * 3)
    no_transition = torch.tensor([[1, 1, 1], [0, 0, 0]])
    out = loss_fn(b, h, None, no_transition)
    assert out['n_transitions'] == 0
    assert float(out['boundary_target_loss']) == 0.0
