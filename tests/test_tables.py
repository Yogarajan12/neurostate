"""Verify that the shipped result files regenerate the published tables.

This is the provenance check: it confirms the numbers in the paper follow
from the stored runs rather than having been transcribed by hand.
"""

import json
from pathlib import Path

import numpy as np
import pytest

RESULTS = Path(__file__).resolve().parents[1] / 'results'

PUBLISHED = {
    'sleep_edf': {'accuracy': (0.698, 0.013), 'f1_macro': (0.612, 0.019),
                  'kappa': (0.596, 0.018), 'boundary_std': (0.026, 0.003)},
    'chb_mit': {'accuracy': (0.901, 0.043), 'auroc': (0.924, 0.014),
                'kappa': (0.356, 0.132), 'sensitivity': (0.796, 0.047),
                'boundary_std': (0.014, 0.006)},
}


@pytest.mark.parametrize('tag', sorted(PUBLISHED))
def test_results_regenerate_published_table(tag):
    path = RESULTS / f'{tag}_results.json'
    if not path.exists():
        pytest.skip(f'{path.name} not present')
    results = json.loads(path.read_text())
    assert len(results) == 5, 'the published table uses five training seeds'
    for key, (pm, ps) in PUBLISHED[tag].items():
        vals = np.array([r[key] for r in results], dtype=float)
        # The paper prints three decimals, so the rounded statistic is what
        # must match, not an absolute tolerance on the raw value.
        assert round(float(vals.mean()), 3) == pm, (
            f'{tag} {key}: mean {vals.mean():.4f} does not round to {pm}')
        assert round(float(vals.std(ddof=1)), 3) == ps, (
            f'{tag} {key}: std {vals.std(ddof=1):.4f} does not round to {ps}')


def test_chbmit_conditional_is_a_non_result():
    """The conditional analysis found no difference between windows with and
    without a label transition. If a future run made this significant, the
    paper's wording would need to change, so it is asserted rather than left
    as prose.
    """
    path = RESULTS / 'chb_mit_conditional.json'
    if not path.exists():
        pytest.skip('chb_mit_conditional.json not present')
    per_seed = json.loads(path.read_text())
    p_values = [r['p_value'] for r in per_seed if 'p_value' in r]
    assert p_values, 'expected per-seed p-values'
    assert min(p_values) >= 0.20, (
        'the paper reports no detectable difference (p >= 0.21); a '
        'significant result here contradicts the published text')
