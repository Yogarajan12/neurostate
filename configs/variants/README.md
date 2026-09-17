# Collapse-table variants

Each file reproduces one row of Table 1 by overriding the Sleep-EDF config.
Only variants expressible as configuration changes appear here. Interventions
that need a different forward pass or module (Gumbel-Softmax masking, hard
thresholding, cumulative-product masking, the differentiable HMM, query-based
segmentation) require code changes and are documented in
`docs/collapse_experiments.md`.

Run one with:

    neurostate-train --config configs/variants/no_isolation.yaml --seeds 42
