"""Generate the README figures from the stored result files.

    python scripts/make_figures.py

Figures are written to assets/ with an explicit white background, because
GitHub renders READMEs in both light and dark themes and a transparent
background turns matplotlib's black text invisible in dark mode.

Every figure is built from files in results/, so each one regenerates
alongside the tables rather than being a static image whose provenance has
been lost.
"""

import argparse
import json
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / 'results'
ASSETS = ROOT / 'assets'

WHITE = 'white'
ISOLATED = '#1f5c8b'
NULL = '#b0b0b0'
THRESHOLD = '#c44536'


def _style(ax):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(alpha=0.25, linewidth=0.6)
    ax.set_axisbelow(True)


def figure_boundary_trajectories(out_path):
    """Boundary std per epoch, five seeds, against the empirical null.

    The paper's collapse evidence is Table 1, which is single-seed and has no
    stored per-epoch traces. This figure therefore shows what the *isolated*
    condition does across seeds, with the shuffled-input null and the 0.020
    degeneracy threshold as reference lines, rather than a collapsed
    trajectory that was never saved. The phase boundaries make the effect of
    the formation phase visible: the boundary statistic is essentially zero
    during warmup and rises once ACBL activates under isolation.
    """
    traces = []
    for seed in range(42, 47):
        path = RESULTS / f'sleep_edf_trace_seed{seed}.json'
        if path.exists():
            traces.append((seed, json.loads(path.read_text())))
    if not traces:
        print('no traces found, skipping trajectories figure')
        return False

    null = json.loads((RESULTS / 'null_distribution.json').read_text())

    fig, ax = plt.subplots(figsize=(7.2, 4.0), facecolor=WHITE)
    ax.set_facecolor(WHITE)

    # The published traces label the middle phase 'form'; PhaseSchedule
    # emits 'formation'. Both are accepted so figures regenerate from either
    # the shipped files or a fresh run.
    formation_names = ('form', 'formation')
    warmup = sum(1 for e in traces[0][1] if e['phase'] == 'warmup')
    formation_end = sum(1 for e in traces[0][1]
                        if e['phase'] in ('warmup',) + formation_names)
    ax.axvspan(0, warmup - 0.5, color='#f0f0f0', zorder=0)
    ax.axvspan(warmup - 0.5, formation_end - 0.5, color='#e3eef5', zorder=0)

    for seed, trace in traces:
        ax.plot([e['epoch'] for e in trace], [e['bnd_std'] for e in trace],
                color=ISOLATED, alpha=0.75, linewidth=1.4,
                label='gradient isolation (5 seeds)' if seed == 42 else None)

    ax.axhline(null['null_mean'], color=NULL, linestyle='-', linewidth=1.2,
               label=f"shuffled-input null ({null['null_mean']:.4f})")
    ax.axhline(0.020, color=THRESHOLD, linestyle='--', linewidth=1.2,
               label='degeneracy threshold (0.020)')

    ymax = max(e['bnd_std'] for _, t in traces for e in t) * 1.12
    ax.text(warmup / 2, ymax * 0.95, 'warmup', ha='center', va='top',
            fontsize=8.5, color='#555555')
    ax.text((warmup + formation_end) / 2, ymax * 0.95, 'formation',
            ha='center', va='top', fontsize=8.5, color='#555555')
    ax.text(formation_end + 4, ymax * 0.95, 'full', ha='center', va='top',
            fontsize=8.5, color='#555555')

    ax.set_xlabel('training epoch')
    ax.set_ylabel('boundary standard deviation')
    ax.set_title('Boundary structure forms during the isolated phase '
                 '(Sleep-EDF)', fontsize=11)
    ax.set_ylim(0, ymax)
    ax.set_xlim(0, max(e['epoch'] for _, t in traces for e in t))
    ax.legend(frameon=False, fontsize=8.5, loc='lower right')
    _style(ax)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160, facecolor=WHITE)
    plt.close(fig)
    print(f'wrote {out_path}')
    return True


def figure_scope(out_path):
    """The three analyses that constrain the interpretability claims.

    Each panel is a comparison the boundary head would have to pass for a
    transition-detection reading to hold, and in each one the two populations
    are indistinguishable. Putting the negative results in the README rather
    than burying them in docs/ is deliberate.
    """
    cond = json.loads((RESULTS / 'chb_mit_conditional.json').read_text())
    cont = json.loads((RESULTS / 'sleep_edf_contiguity.json').read_text())
    cs = json.loads((RESULTS / 'chb_mit_centre_surround.json').read_text())

    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.5), facecolor=WHITE)
    for ax in axes:
        ax.set_facecolor(WHITE)
        _style(ax)

    ax = axes[0]
    a = [r['mean_window_std_transition'] for r in cond]
    b = [r['mean_window_std_flat'] for r in cond]
    x = np.arange(2)
    ax.bar(x, [np.mean(a), np.mean(b)],
           yerr=[np.std(a, ddof=1), np.std(b, ddof=1)],
           color=[ISOLATED, NULL], width=0.55, capsize=4)
    ax.set_xticks(x)
    ax.set_xticklabels(['contains\ntransition', 'no\ntransition'],
                       fontsize=9)
    ax.set_ylabel('mean window boundary std')
    pmin = min(r['p_value'] for r in cond)
    ax.set_title(f'CHB-MIT by window type\n'
                 f'no difference: p $\\geq$ {pmin:.2f}, 5 seeds',
                 fontsize=9.5)

    ax = axes[1]
    vals = [cont['contiguous']['mean_window_std'],
            cont['spliced']['mean_window_std']]
    ax.bar(x, vals, color=[ISOLATED, NULL], width=0.55)
    ax.set_xticks(x)
    ax.set_xticklabels([f"contiguous\n(n={cont['contiguous']['n']})",
                        f"spliced\n(n={cont['spliced']['n']})"], fontsize=9)
    ax.set_ylabel('mean window boundary std')
    ax.set_title('Sleep-EDF by recording contiguity\n'
                 'no difference: 0.0214 vs 0.0220', fontsize=9.5)

    ax = axes[2]
    seiz = [r['positive_seizure'] * 100 for r in cs]
    norm = [r['positive_normal'] * 100 for r in cs]
    ax.bar(x, [np.mean(seiz), np.mean(norm)],
           yerr=[np.std(seiz, ddof=1), np.std(norm, ddof=1)],
           color=[ISOLATED, NULL], width=0.55, capsize=4)
    ax.set_xticks(x)
    ax.set_xticklabels([f'seizure\n(n={cs[0]["n_seizure"]})',
                        f'non-seizure\n(n={cs[0]["n_normal"]})'], fontsize=9)
    ax.set_ylabel('windows with centre > surround (%)')
    ax.set_ylim(90, 101)
    ax.set_title('CHB-MIT centre vs surround\n'
                 'effect is window geometry, not seizure', fontsize=9.5)

    fig.suptitle('What the boundary output has not been shown to do',
                 fontsize=11, y=1.02)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160, facecolor=WHITE, bbox_inches='tight')
    plt.close(fig)
    print(f'wrote {out_path}')
    return True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out-dir', default=str(ASSETS))
    args = parser.parse_args()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    figure_boundary_trajectories(out / 'boundary_trajectories.png')
    figure_scope(out / 'scope_analyses.png')


if __name__ == '__main__':
    main()
