# Method

The architecture and training curriculum as implemented. Where a reasonable
reading of the paper would differ from what the code does, that is stated
rather than left for a reproducer to discover.

## Architecture

### Multi-resolution encoder

Each 30-second epoch is encoded at 100, 50 and 25 Hz. Each branch applies a
temporal convolution, a spatial convolution across all channels, average
pooling and a linear projection. The three branches are interpolated to the
100 Hz token length (196 tokens) and merged by a learned linear projection.

The rates are chosen physiologically. Sleep spindles (11 to 16 Hz) need
sampling above 50 Hz to avoid aliasing; slow waves dominating N3 (0.5 to 4 Hz)
are well captured at 25 Hz; seizure onset transients need the 100 Hz branch.

Three consecutive epochs form the input window, giving 588 tokens.

**Parameter count is montage-dependent.** The spatial convolution is
`Conv2d(40, 40, (n_channels, 1))` and is instantiated once per resolution
branch, so its size scales with the number of input channels:

```
3 × (40 × 40 × 17 + 40) − 3 × (40 × 40 × 3 + 40) = 67,200
```

The model is 1.00 M parameters on the 3-channel Sleep-EDF montage and 1.07 M
on the 17-channel CHB-MIT and 19-channel TUAB montages. `tests/test_model.py`
asserts both.

### Boundary module

Boundaries come from representation *contrast* rather than absolute token
content, so the head cannot simply learn to fire on high-amplitude tokens.

For each offset `s ∈ {1, 4, 16}` (roughly 150 ms to 2.4 s), tokens pass
through a per-scale projection MLP, are L2 normalised, and compared with the
token `s` positions ahead by cosine similarity. The contrast
`1 − (sim + 1)/2` passes through a sigmoid with a **learned temperature**
shared across scales. A **two-layer fusion MLP** combines the three per-scale
contrasts into `b(t)`.

Cosine similarity replaces Euclidean distance so the head is scale-invariant
across electrodes and recording sessions.

An **internal consistency term**, `0.01 · Σ_s MSE(d_s, sg[b])`, keeps the
per-scale estimates near the fused output. It belongs to the boundary module
and is added to the total objective *outside* the λ-weighted ACBL term. It is
easy to miss when reading the loss equation alone.

> **Note.** Earlier drafts wrote equation 1 as a single linear fusion,
> `b(t) = σ(W_f · [d₁, d₄, d₁₆]ᵀ + b_f)`. That was a schematic. The
> camera-ready corrects it to match the implementation: per-scale projections,
> a learned temperature, and the two-layer fusion. If working from an older
> copy of the paper, trust the code and the camera-ready.

### Regime-structured attention

Boundary probabilities condition attention through a cumulative sum and a
same-regime affinity mask:

```
C_i = Σ_{j≤i} b(j)          S_ij = exp(−|C_i − C_j|)
```

The eight heads are partitioned into three groups: four *intra-regime* heads
use `S`, two *inter-regime* heads use `1 − S`, and two *cross-scale* heads are
unmasked. Attention logits become
`ã_ij = a_ij + log(M_ij + ε)` with `ε = 1e-6`.

The name "cross-scale" describes the design intent rather than the behaviour;
those heads apply standard unmasked attention for global context, and both the
paper and this documentation say so outright.

Two behaviours worth stating because they are invisible in the equations:

- **The mask is rebuilt inside every transformer block**, not computed once.
  `_build_regime_mask` lives in the attention module and `forward` is called
  per block with the same boundary vector.
- **Pooling is centre-epoch, not global.** After the final block, the
  classifier reads a mean over the 196 centre-epoch tokens only, with the
  flanking epochs acting as context. `tests/test_model.py` asserts that
  perturbing the centre epoch moves the logits more than perturbing a flank.

## Why the head collapses

When `b(t)` is constant, `C` is linear in position, so `S_ij` depends only on
`|i − j|`. The mask becomes an exponential distance-decay filter, identical
for every input and carrying no information about the signal.

That filter is *already a useful attention prior*. It supports above-0.70
accuracy on Sleep-EDF on its own. So the classification objective can reach
good accuracy without the boundary head ever becoming informative, and the
gradient reaching the head carries almost no pressure to make it so. This is
the shortcut pathology: the classifier takes the shortcut, and the auxiliary
head is left with no reason to develop content.

A second pathology compounds it. Classification gradients push encoder
representations toward configurations that make the boundary head's job
harder, so the head is fitting a moving target.

`tests/test_mechanism.py::test_flat_boundaries_give_distance_decay_mask`
asserts the first property directly, and `demo/collapse_demo.py` shows that a
deliberately frozen flat boundary reaches essentially the same accuracy as a
learned one.

## ACBL with gradient isolation

### The curriculum

| Phase | Epochs | Behaviour |
|---|---|---|
| Warmup | 0 to 2 | Classification loss only. The head gets no signal, but the encoder develops representations for it to fit. |
| Formation | 3 to 14 | ACBL losses active. The boundary output is **detached** on the path into attention. |
| Full | 15 onward | Boundaries reconnected. The structure formed during the previous phase survives, because the mask is now content-dependent rather than a passive distance filter. |

Early stopping is enabled **only once the full phase begins**, so a run cannot
terminate during formation and be evaluated with the head still detached.

### The directional stop-gradient

During formation, `∇_θbnd L_ACBL` is preserved while the classification
gradient into the boundary head is blocked. The asymmetry is essential:

- Detach **both** directions: the head is untrainable and decoupled from the
  encoder.
- Couple **both** directions: collapse returns.
- Detach **one** direction: the head develops content independently before
  being folded into the joint optimisation.

The directionality is the mechanism, not a hyperparameter. It makes the
curriculum a property of *how gradients flow* rather than *what they are
computed on*, so nothing about it is specific to this boundary head. That is
the argument for expecting it to carry to other parameterisations, not a
result: the collapse was observed under sigmoid masks, discrete masks, HMMs
and segmentation queries, but gradient isolation was only applied to the
sigmoid one. Extending it to the others is unrun work, not a reported
finding.

`tests/test_mechanism.py::test_gradient_isolation_blocks_only_classification_path`
asserts all three conditions.

### The four ACBL terms

```
L_ACBL = α·L_pseudo + β·(L_Gauss + γ·L_KL) + δ·L_var
α = 0.3,  β = 0.5,  γ = 0.5,  δ = 1.0,  applied at λ = 0.3
```

| Term | What it does |
|---|---|
| `L_pseudo` | BCE against self-supervised targets from adjacent-token cosine distance, min-max normalised and passed through a sigmoid with temperature annealed 2.0 → 0.5 over 15 epochs. Targets are detached so the encoder cannot satisfy the head by inflating adjacent dissimilarity. Epoch junctions are masked out, so this term carries no information about where junctions are. |
| `L_Gauss` | BCE against Gaussian targets (σ = 5 tokens) placed at epoch junctions where the epoch label changes. |
| `L_KL` | KL divergence between the head-averaged attention of the final block and a block-diagonal prior from epoch labels (0.8 same-stage, 0.2 otherwise, row normalised). Routes gradient to the head *through* attention. |
| `L_var` | Negative per-window variance of `b(t)`, penalising constant outputs directly. |

`L_Gauss` and `L_KL` are averaged over the windows in a batch containing at
least one label transition; windows without one contribute nothing. On TUAB
no window contains a transition by construction, so both terms are inactive
throughout.

Two things worth knowing:

- **λ = 0.3, not 1.0.** Earlier drafts stated λ = 1.0; the code uses
  `acbl_weight = 0.3` and the camera-ready corrects it.
- **There are four terms, not three.** The variance penalty is live in every
  reported run. A three-term reading would not reproduce the numbers.
- **Notation.** The camera-ready uses δ for the variance weight; the
  conference poster uses ν. This package follows the paper.

### Why the variance penalty is not enough on its own

Table 1 shows boundary losses alone at 0.011 and with the variance penalty at
0.007, both degenerate. Only adding gradient isolation reaches 0.028.
Penalising flatness rewards *spread* but creates no gradient distinguishing
one boundary position from another, so the head can satisfy it with noise
rather than structure. The ACBL rows in Table 1 are cumulative, not
alternatives.

## Per-dataset recipes

The paper's setup line reads as a single recipe. The runs differ in ways that
matter for reproduction:

| | Sleep-EDF | CHB-MIT | TUAB |
|---|---|---|---|
| Batch size | 16 | 32 | 32 |
| Optimiser groups | two (boundary lr 3e-4, wd 1e-5) | one | one |
| Class-weighted loss | no | yes | yes |
| Plateau scheduler | yes | yes | no |
| ACBL terms | four | four | **three** (no variance penalty) |
| Split | 70/15/15 over identifiers, seeded | deterministic round-robin | random 60/20/20 over epochs |

Phase lengths (warmup 3, formation 12) and the ACBL weights are identical
across all three, fixed in advance and never selected on test data. That
reuse is the evidence against per-task tuning.

The configs in `configs/` encode these differences explicitly. Setting
`variance_weight: 1.0` on TUAB, for instance, would not reproduce the reported
numbers.
