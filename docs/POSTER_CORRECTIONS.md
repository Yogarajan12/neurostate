# Poster corrections against the camera-ready

Checked `neurostate_poster_v6.tex` against `mlsp_template_camera_ready_4.tex`.
Line numbers refer to the poster source. One correction is substantive, the
rest are accuracy and scope.

## 1. ACBL equation is missing a term and states lambda wrongly

**Line 311.** The poster gives three terms and `lambda = 1.0`:

    L_ACBL = alpha*L_pseudo + beta*(L_Gaussian + gamma*L_KL)
    alpha=0.3, beta=0.5, gamma=0.5, lambda=1.0

The camera-ready (line 100) has four terms and `lambda = 0.3`:

    L_ACBL = alpha*L_pseudo + beta*(L_Gauss + gamma*L_KL) + delta*L_var
    alpha=0.3, beta=0.5, gamma=0.5, delta=1.0, lambda=0.3

Restore `+ delta*L_var` and set the gloss to `alpha=0.3, beta=0.5, gamma=0.5,
delta=1.0, lambda=0.3`. The probable origin is `delta=1.0` mistyped as
`lambda=1.0`.

This matters beyond notation: the variance penalty is a row in the collapse
table the poster is built around, and "boundary losses alone" versus "plus
variance penalty" versus "plus gradient isolation" is the cumulative argument
the panel makes.

## 2. "subject-level split" is accurate for only one of the three datasets

**Line ~332, results table footnote.** Only CHB-MIT splits by subject, through
the deterministic round-robin. Sleep-EDF splits on recording identifiers, and
two nights of the same person carry different identifiers. TUAB splits
randomly over epoch indices.

The camera-ready says "a fixed split". Use that; it fits the same space and is
true of all three rows.

## 3. The collapse-range claim excludes two rows and overstates the threshold

**Line 317.** "Every other variant stays under the 0.020 degeneracy threshold,
at 0.003--0.018."

Two problems. The differentiable HMM and query-based segmentation rows report
`n/a` for the boundary statistic, so they are not inside that range. And the
camera-ready (line 127) states the 0.020 cutoff is "a descriptive label rather
than a test", with bimodal scheduling at 0.014 and cumulative-product at 0.018
judged degenerate on training behaviour and accuracy rather than on the
statistic.

Suggested replacement, same length: "The nine variants with a comparable
statistic all fall between 0.003 and 0.018, below the 0.020 line that labels a
degenerate distribution."

## 4. Two independent findings joined by a causal "so"

**Line ~349, CHB-MIT bullet.** The text runs the centre-versus-surround
geometry into the onset latency with "so the peaks track epoch joins: a median
30.0 s from onset against 10.5 s for PELT".

These are separate measurements. The geometry result explains why the centre
epoch scores above its neighbours; it does not produce the latency figure.
Replace the colon construction with a full stop and a separate sentence.

## 5. Section 5 generalises further than the paper does

**Lines ~370--375.** The poster says the failure "comes from training two jobs
together, not from EEG", and that any sequence model with a second head the
main loss does not need can reach the same degenerate solution.

The camera-ready reaches the same intuition but closes by scoping it: the
pathology "lies in the joint optimization itself rather than in any particular
representation of the boundary. We scope the conclusion to the EEG settings
measured here."

Keep the reach, add the limit. For example: "the failure comes from training
two jobs together rather than from any particular way of writing boundaries
down; whether it generalises beyond the EEG settings measured here is
untested."

This is the claim most likely to be challenged at the session, and conceding
the limit unprompted is a stronger position than defending the stronger
version.

## 6. Placeholders still unset

**Lines 34--35.** `\codeurl` is `https://github.com/FILL-IN-HANDLE/neurostate`
and `\email` is `FILL.IN@your-address.com`. Both print as dead references.
`\paperurl` is already correct.

Set `\codeurl` to:

    \newcommand{\codeurl}{https://github.com/Yogarajan12/neurostate}

This is the path the repository is published at, and it matches
`pyproject.toml`, `CITATION.cff` and the README clone line. Do not rename the
repository afterwards: GitHub's redirect after a rename is a courtesy that
lapses if anything later occupies the old path, and a printed QR code cannot
be reissued.

## Verified as correct, no change needed

- Finding 2 describes gradient isolation as cutting the classification
  gradient into the boundary head, matching the camera-ready. The epochs
  3--14 window and the twelve-epoch figure in Section 5 are both right.
- All twelve numbers in the results table match Table 2 of the paper.
- The null statement, 5/5 seeds at 1.47--1.94x, matches line 127.
- The CHB-MIT conditional figures (n=59 against n=1,437, p >= 0.21,
  d = -0.25 +/- 0.20) and the centre-surround percentages (98.9 and 99.9 over
  1,443 windows) match line 166.
- The Sleep-EDF spectral claim (2.1x beta, 3.3x gamma, p < 10^-22) matches
  line 262.
- Calling the interpretability result preliminary matches the abstract.
