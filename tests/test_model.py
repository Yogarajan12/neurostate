"""Shape, parameter-count and checkpoint-compatibility tests."""

import torch

from neurostate.model import (
    ContrastiveBoundaryModule,
    MultiResContrastiveNeuroState,
)


def test_parameter_counts_match_paper():
    """The paper states 1.00 M parameters on Sleep-EDF and 1.07 M on the
    17-channel montage. The difference comes from the spatial convolution,
    which spans all input channels and is instantiated once per resolution
    branch: 3 x (40 x 40 x 17 + 40) - 3 x (40 x 40 x 3 + 40) = 67,200.
    """
    sleep = MultiResContrastiveNeuroState(n_channels=3, n_classes=5)
    chb = MultiResContrastiveNeuroState(n_channels=17, n_classes=2)
    assert round(sleep.n_params / 1e6, 2) == 1.00
    assert round(chb.n_params / 1e6, 2) == 1.07
    assert chb.n_params - sleep.n_params == 67200 + (
        chb.head[-1].weight.numel() + chb.head[-1].bias.numel()
        - sleep.head[-1].weight.numel() - sleep.head[-1].bias.numel())


def test_token_geometry():
    """196 tokens per epoch, 588 per three-epoch window, from
    (3000 - 75) // 15 + 1.
    """
    model = MultiResContrastiveNeuroState(n_channels=3)
    assert model.tokens_per_epoch == 196
    assert model.pos_embed.shape[1] == 588


def test_forward_shapes_and_boundary_range():
    model = MultiResContrastiveNeuroState(n_channels=3, n_samples=600,
                                          embed_dim=32, n_layers=2,
                                          n_intra=2, n_inter=1, n_cross=1,
                                          cp_hidden=16)
    out = model(torch.randn(2, 3, 3, 600), return_boundaries=True)
    assert out['logits'].shape == (2, 5)
    assert out['boundaries'].shape == (2, model.tokens_per_epoch * 3)
    assert out['boundaries'].min() >= 0.0
    assert out['boundaries'].max() <= 1.0


def test_centre_epoch_pooling():
    """Classification must read the centre epoch, with the flanking epochs
    acting as context. Perturbing only a flanking epoch should change the
    logits less than perturbing the centre one.
    """
    torch.manual_seed(0)
    model = MultiResContrastiveNeuroState(n_channels=3, n_samples=600,
                                          embed_dim=32, n_layers=2,
                                          n_intra=2, n_inter=1, n_cross=1,
                                          cp_hidden=16).eval()
    x = torch.randn(1, 3, 3, 600)
    with torch.no_grad():
        base = model(x)['logits']
        centre = x.clone()
        centre[:, 1] += 5.0
        flank = x.clone()
        flank[:, 0] += 5.0
        d_centre = (model(centre)['logits'] - base).abs().sum()
        d_flank = (model(flank)['logits'] - base).abs().sum()
    assert d_centre > d_flank


def test_state_dict_keys_are_stable():
    """Checkpoint compatibility. These names are what the published runs
    saved; renaming a module would silently break checkpoint loading.
    """
    model = MultiResContrastiveNeuroState(n_channels=3)
    keys = set(model.state_dict().keys())
    for expected in [
        'mr_encoder.enc_100hz.temporal_conv.0.weight',
        'mr_encoder.enc_50hz.spatial_conv.0.weight',
        'mr_encoder.merge.0.weight',
        'changepoint_module.temperature',
        'changepoint_module.projections.0.0.weight',
        'changepoint_module.fusion.0.weight',
        'blocks.0.attn.qkv.weight',
        'pos_embed', 'epoch_embed', 'head.0.weight',
    ]:
        assert expected in keys, f"missing expected checkpoint key {expected}"


def test_boundary_module_consistency_term_is_non_negative():
    module = ContrastiveBoundaryModule(embed_dim=32, hidden_dim=16)
    out = module(torch.randn(2, 60, 32))
    assert out['boundary_loss'] >= 0.0
    assert len(out['per_scale']) == 3
