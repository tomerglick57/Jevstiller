import numpy as np

from jevstiller.calibrate import clopper_pearson_upper, fit_policy


def test_cp_zero_failures_matches_closed_form():
    # P(X=0 | n, p) = (1-p)^n = delta  ->  p = 1 - delta^(1/n)
    for n in (10, 100, 1000):
        assert abs(clopper_pearson_upper(0, n, 0.05) - (1 - 0.05 ** (1 / n))) < 1e-6


def test_cp_monotone_and_bounded():
    assert clopper_pearson_upper(5, 100) > clopper_pearson_upper(2, 100) > 0.02
    assert clopper_pearson_upper(100, 100) == 1.0
    assert clopper_pearson_upper(0, 0) == 1.0
    assert 0.05 < clopper_pearson_upper(5, 100) < 0.12


def test_fit_policy_respects_budget():
    rng = np.random.default_rng(0)
    N = 5000
    conf = rng.uniform(0.2, 1.0, N)
    # disagreement more likely when confidence is low: ~0.8% at conf 0.9, ~7% at conf 0.7
    p_disagree = 0.5 * ((1 - conf) / 0.8) ** 2
    agree = rng.uniform(size=N) >= p_disagree
    ood = rng.uniform(size=N)
    pol = fit_policy(conf, agree, ood, budget=0.02, delta=0.05)
    assert pol.usable
    assert pol.expected_system_disagreement <= 0.02
    sel = pol.accepts(conf, ood)
    realised = (sel & ~agree).sum() / N
    assert realised <= 0.02 + 0.005
    assert pol.expected_coverage > 0.2


def test_fit_policy_no_solution():
    conf = np.full(200, 0.9)
    agree = np.zeros(200, bool)          # student always wrong
    pol = fit_policy(conf, agree, np.zeros(200), budget=0.02)
    assert not pol.usable and pol.expected_coverage == 0.0
