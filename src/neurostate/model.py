"""NeuroState architecture: multi-resolution EEG encoder with a contrastive
boundary module and regime-structured attention.

Module and parameter names are kept identical to the notebooks that produced
the published results, so checkpoints saved from those notebooks load into
this package without remapping keys.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = [
    "PatchEmbedding",
    "MultiResolutionEncoder",
    "ContrastiveBoundaryModule",
    "RegimeStructuredAttention",
    "NeuroStateBlock",
    "MultiResContrastiveNeuroState",
    "forward_with_intermediates",
]


class PatchEmbedding(nn.Module):
    """Single-resolution encoder: temporal convolution, spatial convolution
    across channels, average pooling, then projection to the embedding width.

    The spatial convolution spans all input channels, so its parameter count
    scales with the montage. This is why the model has 1.00 M parameters on
    the 3-channel Sleep-EDF montage and 1.07 M on the 17-channel CHB-MIT one.
    """

    def __init__(self, n_channels=22, n_samples=3000, embed_dim=128,
                 temporal_kernel=25, pool_kernel=75, pool_stride=15,
                 dropout=0.1):
        super().__init__()
        self.temporal_conv = nn.Sequential(
            nn.Conv2d(1, 40, (1, temporal_kernel),
                      padding=(0, temporal_kernel // 2)),
            nn.BatchNorm2d(40), nn.GELU())
        self.spatial_conv = nn.Sequential(
            nn.Conv2d(40, 40, (n_channels, 1)),
            nn.BatchNorm2d(40), nn.GELU())
        self.pool = nn.AvgPool2d((1, pool_kernel), stride=(1, pool_stride))
        self.projection = nn.Sequential(
            nn.Conv2d(40, embed_dim, (1, 1)), nn.Dropout(dropout))
        self.seq_len = (n_samples - pool_kernel) // pool_stride + 1

    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.temporal_conv(x)
        x = self.spatial_conv(x)
        x = self.pool(x)
        x = self.projection(x)
        return x.squeeze(2).permute(0, 2, 1)


class MultiResolutionEncoder(nn.Module):
    """Encodes one 30-second epoch at 100, 50 and 25 Hz, then merges the three
    branches by linear interpolation to the 100 Hz token length.

    The three rates are chosen on physiological grounds: sleep spindles
    (11-16 Hz) need sampling above 50 Hz to avoid aliasing, slow waves that
    dominate N3 (0.5-4 Hz) are well captured at 25 Hz, and seizure onset
    transients need the 100 Hz branch.
    """

    def __init__(self, n_channels=3, n_samples=3000, embed_dim=128,
                 dropout=0.1):
        super().__init__()
        self.enc_100hz = PatchEmbedding(
            n_channels, n_samples, embed_dim, dropout=dropout)
        self.enc_50hz = PatchEmbedding(
            n_channels, n_samples // 2, embed_dim, dropout=dropout)
        self.enc_25hz = PatchEmbedding(
            n_channels, n_samples // 4, embed_dim, dropout=dropout)
        self.merge = nn.Sequential(
            nn.Linear(embed_dim * 3, embed_dim), nn.GELU(),
            nn.Dropout(dropout))
        self.seq_len_100 = self.enc_100hz.seq_len
        self.seq_len_50 = self.enc_50hz.seq_len
        self.seq_len_25 = self.enc_25hz.seq_len

    def forward(self, x):
        x_50 = x[:, :, ::2]
        x_25 = x[:, :, ::4]
        emb_100 = self.enc_100hz(x)
        emb_50 = self.enc_50hz(x_50)
        emb_25 = self.enc_25hz(x_25)
        t1 = emb_100.shape[1]
        emb_50_up = F.interpolate(
            emb_50.permute(0, 2, 1), size=t1,
            mode='linear', align_corners=False).permute(0, 2, 1)
        emb_25_up = F.interpolate(
            emb_25.permute(0, 2, 1), size=t1,
            mode='linear', align_corners=False).permute(0, 2, 1)
        merged = torch.cat([emb_100, emb_50_up, emb_25_up], dim=-1)
        return self.merge(merged)


class ContrastiveBoundaryModule(nn.Module):
    """Estimates per-token changepoint probabilities from representation
    contrast rather than from absolute token content.

    For each offset s in `scales`, tokens are passed through a per-scale
    projection MLP, L2 normalised, and compared with the token s positions
    ahead by cosine similarity. The contrast 1 - (sim + 1) / 2 is passed
    through a sigmoid with a learned temperature shared across scales. A
    two-layer fusion MLP combines the three per-scale contrasts into b(t).

    This matches equations 1 and 2 of the camera-ready paper. An internal
    consistency term (weight 0.01) keeps the per-scale estimates near the
    fused output; it is returned as `boundary_loss` and is added to the total
    objective outside the lambda-weighted ACBL term.
    """

    def __init__(self, embed_dim=128, hidden_dim=64,
                 scales=(1, 4, 16), dropout=0.1):
        super().__init__()
        self.scales = scales
        self.n_scales = len(scales)
        self.projections = nn.ModuleList([
            nn.Sequential(
                nn.Linear(embed_dim, hidden_dim), nn.GELU(),
                nn.Dropout(dropout), nn.Linear(hidden_dim, hidden_dim))
            for _ in scales])
        self.fusion = nn.Sequential(
            nn.Linear(self.n_scales, self.n_scales * 2), nn.GELU(),
            nn.Linear(self.n_scales * 2, 1))
        self.temperature = nn.Parameter(torch.tensor(1.0))

    def _compute_contrast(self, x, proj, offset):
        b, t, _ = x.shape
        h = proj(x)
        h_norm = F.normalize(h, dim=-1)
        if offset < t:
            h_shifted = torch.roll(h_norm, -offset, dims=1)
            h_shifted[:, -offset:, :] = h_norm[:, -offset:, :]
            similarity = (h_norm * h_shifted).sum(dim=-1)
            contrast = 1.0 - (similarity + 1.0) / 2.0
            contrast = torch.sigmoid(
                (contrast - 0.5) * self.temperature.abs().clamp(min=0.1))
        else:
            contrast = torch.zeros(b, t, device=x.device)
        return contrast

    def forward(self, x):
        per_scale = [self._compute_contrast(x, proj, off)
                     for proj, off in zip(self.projections, self.scales)]
        stacked = torch.stack(per_scale, dim=-1)
        fused = torch.sigmoid(self.fusion(stacked).squeeze(-1))
        consistency = sum(F.mse_loss(ps, fused.detach())
                          for ps in per_scale) / len(per_scale)
        return {'boundaries': fused, 'per_scale': per_scale,
                'boundary_loss': 0.01 * consistency}


class RegimeStructuredAttention(nn.Module):
    """Attention whose heads are partitioned into three functional groups by
    the boundary probabilities.

    The same-regime affinity mask is S_ij = exp(-|C_i - C_j|) where C is the
    cumulative sum of b(t). Four intra-regime heads use S, two inter-regime
    heads use 1 - S, and two cross-scale heads are unmasked.

    The mask is rebuilt inside every transformer block rather than computed
    once, because `_build_regime_mask` lives here and `forward` is called per
    block with the same boundary vector.

    The collapse mechanism is visible in `_build_regime_mask`: when b(t) is
    constant, C becomes linear in position and S degenerates into a pure
    exponential distance decay, which already supports above-0.70 accuracy on
    Sleep-EDF. The classification objective therefore has no incentive to make
    the boundary head informative. This is the useful-when-flat property that
    Section 4.2 of the paper identifies as the architectural cause.
    """

    def __init__(self, embed_dim=128, n_intra=4, n_inter=2,
                 n_cross=2, dropout=0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.n_intra = n_intra
        self.n_inter = n_inter
        self.n_cross = n_cross
        self.n_heads = n_intra + n_inter + n_cross
        self.head_dim = embed_dim // self.n_heads
        assert embed_dim % self.n_heads == 0
        self.qkv = nn.Linear(embed_dim, 3 * embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.attn_drop = nn.Dropout(dropout)
        self.proj_drop = nn.Dropout(dropout)
        self.scale = self.head_dim ** -0.5

    def _build_regime_mask(self, boundaries):
        cum = torch.cumsum(boundaries, dim=1)
        return torch.exp(-torch.abs(cum.unsqueeze(2) - cum.unsqueeze(1)))

    def forward(self, x, boundaries, return_attention=False):
        b, t, d = x.shape
        qkv = self.qkv(x).reshape(b, t, 3, self.n_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        same = self._build_regime_mask(boundaries).unsqueeze(1)
        cross = 1.0 - same
        h_ie = self.n_intra
        h_ce = h_ie + self.n_inter
        mask = torch.ones_like(attn)
        mask[:, :h_ie] = same.expand(b, self.n_intra, t, t)
        mask[:, h_ie:h_ce] = cross.expand(b, self.n_inter, t, t)
        attn = attn + torch.log(mask + 1e-6)
        attn_w = F.softmax(attn, dim=-1)
        attn_w = self.attn_drop(attn_w)
        out = (attn_w @ v).transpose(1, 2).reshape(b, t, d)
        out = self.proj_drop(self.out_proj(out))
        return (out, attn_w) if return_attention else out


class NeuroStateBlock(nn.Module):
    """Pre-norm transformer block using regime-structured attention."""

    def __init__(self, embed_dim=128, n_intra=4, n_inter=2,
                 n_cross=2, mlp_ratio=4.0, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = RegimeStructuredAttention(
            embed_dim, n_intra, n_inter, n_cross, dropout)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, int(embed_dim * mlp_ratio)),
            nn.GELU(), nn.Dropout(dropout),
            nn.Linear(int(embed_dim * mlp_ratio), embed_dim),
            nn.Dropout(dropout))

    def forward(self, x, boundaries, return_attention=False):
        normed = self.norm1(x)
        if return_attention:
            a, w = self.attn(normed, boundaries, True)
            x = x + a
            x = x + self.mlp(self.norm2(x))
            return x, w
        x = x + self.attn(normed, boundaries)
        x = x + self.mlp(self.norm2(x))
        return x


class MultiResContrastiveNeuroState(nn.Module):
    """Full NeuroState model.

    Three consecutive 30-second epochs form the input window (588 tokens at
    196 tokens per epoch). Classification reads a mean over the centre epoch's
    tokens only, with the flanking epochs acting as context.
    """

    def __init__(self, n_channels=3, n_samples=3000, n_classes=5,
                 embed_dim=128, n_layers=4, dropout=0.1,
                 n_intra=4, n_inter=2, n_cross=2,
                 contrast_scales=(1, 4, 16), cp_hidden=64,
                 n_context_epochs=3, verbose=False):
        super().__init__()
        self.n_classes = n_classes
        self.n_context = n_context_epochs
        self.mr_encoder = MultiResolutionEncoder(
            n_channels, n_samples, embed_dim, dropout)
        tokens_per_epoch = self.mr_encoder.seq_len_100
        total_tokens = tokens_per_epoch * n_context_epochs
        self.pos_embed = nn.Parameter(
            torch.randn(1, total_tokens, embed_dim) * 0.02)
        self.pos_drop = nn.Dropout(dropout)
        self.epoch_embed = nn.Parameter(
            torch.randn(1, n_context_epochs, 1, embed_dim) * 0.02)
        self.changepoint_module = ContrastiveBoundaryModule(
            embed_dim=embed_dim, hidden_dim=cp_hidden,
            scales=contrast_scales, dropout=dropout)
        self.blocks = nn.ModuleList([
            NeuroStateBlock(embed_dim, n_intra, n_inter, n_cross,
                            dropout=dropout)
            for _ in range(n_layers)])
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2), nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim // 2, n_classes))
        self.tokens_per_epoch = tokens_per_epoch
        self.apply(self._init_weights)
        self.n_params = sum(p.numel() for p in self.parameters())
        if verbose:
            print(f"MultiResContrastiveNeuroState: "
                  f"{self.n_params / 1e6:.2f}M params, {total_tokens} tokens")

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.LayerNorm):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)

    def forward(self, x, return_boundaries=False):
        _, n, _, _ = x.shape
        epoch_embs = []
        for i in range(n):
            emb = self.mr_encoder(x[:, i])
            emb = emb + self.epoch_embed[:, i]
            epoch_embs.append(emb)
        full_seq = torch.cat(epoch_embs, dim=1)
        full_seq = self.pos_drop(full_seq + self.pos_embed)
        cp_out = self.changepoint_module(full_seq)
        boundaries = cp_out['boundaries']
        boundary_loss = cp_out['boundary_loss']
        for block in self.blocks:
            full_seq = block(full_seq, boundaries)
        full_seq = self.norm(full_seq)
        start = self.tokens_per_epoch * (n // 2)
        end = start + self.tokens_per_epoch
        pooled = full_seq[:, start:end, :].mean(dim=1)
        logits = self.head(pooled)
        out = {'logits': logits, 'boundary_loss': boundary_loss}
        if return_boundaries:
            out['boundaries'] = boundaries
            out['per_scale'] = cp_out['per_scale']
        return out


def forward_with_intermediates(model, x, detach_boundaries=False):
    """Forward pass exposing the tensors the ACBL losses need.

    Returns the encoder output before boundary detection, the boundary head
    output, and the attention weights of the final block.

    When `detach_boundaries` is True, the boundary vector is detached before
    it enters regime attention. The classification loss then cannot send
    gradients back into the boundary head, while the ACBL losses still can,
    because they are applied to the undetached `boundary_probs`. That one-way
    block is gradient isolation, and it is applied only during the formation
    phase.
    """
    _, n, _, _ = x.shape
    epoch_embs = []
    for i in range(n):
        emb = model.mr_encoder(x[:, i])
        emb = emb + model.epoch_embed[:, i]
        epoch_embs.append(emb)
    full_seq = torch.cat(epoch_embs, dim=1)
    full_seq = model.pos_drop(full_seq + model.pos_embed)
    encoder_h = full_seq
    cp_out = model.changepoint_module(full_seq)
    boundaries = cp_out['boundaries']
    boundary_loss = cp_out['boundary_loss']
    boundaries_for_attn = (boundaries.detach() if detach_boundaries
                           else boundaries)
    regime_attention = None
    for i, block in enumerate(model.blocks):
        if i == len(model.blocks) - 1:
            full_seq, attn_w = block(
                full_seq, boundaries_for_attn, return_attention=True)
            regime_attention = attn_w
        else:
            full_seq = block(full_seq, boundaries_for_attn)
    full_seq = model.norm(full_seq)
    tpe = model.tokens_per_epoch
    start = tpe * (n // 2)
    pooled = full_seq[:, start:start + tpe, :].mean(dim=1)
    logits = model.head(pooled)
    return {
        'logits': logits,
        'boundary_loss': boundary_loss,
        'boundaries': boundaries,
        'encoder_h': encoder_h,
        'boundary_probs': boundaries,
        'regime_attention': regime_attention,
    }
