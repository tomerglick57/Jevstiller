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


def test_the_guarantee_holds_at_delta():
    """Many calibration sets from a known distribution: the chosen threshold's true rate of answered-and-
    disagreeing requests exceeds the budget in at most ~delta of them. (The old rule, which kept the widest of
    ~200 thresholds each tested at the full delta and bounded coverage x selective rate, gave no such bound.)"""
    rng = np.random.default_rng(1)
    budget, delta, N = 0.03, 0.1, 600

    def p_dis(c):                         # disagreement probability given confidence, conf ~ U(0.5, 1)
        return 0.6 * ((1 - c) / 0.5) ** 2

    fine = np.linspace(0.5, 1, 20001)

    def true_rate(t):                     # P(conf >= t and disagree)
        m = fine >= t
        trapezoid = getattr(np, "trapezoid", None) or np.trapz   # numpy < 2 has only trapz
        return float(trapezoid(np.where(m, p_dis(fine), 0), fine) / 0.5)

    grid = np.linspace(0.99, 0.5, 60)
    sims, bad = 300, 0
    for _ in range(sims):
        conf = rng.uniform(0.5, 1, N)
        agree = rng.uniform(size=N) >= p_dis(conf)
        pol = fit_policy(conf, agree, np.zeros(N), budget=budget, delta=delta, candidates=grid)
        if pol.usable and true_rate(pol.conf_threshold) > budget:
            bad += 1
    assert bad / sims <= delta + 3 * (delta * (1 - delta) / sims) ** 0.5, bad


def test_scan_stops_at_the_first_failure():
    # a cluster of confident mistakes: the bound fails at 0.95 and never recovers, however many
    # easy rows lie below it
    conf = np.concatenate([np.full(50, 0.97), np.full(950, 0.6)])
    agree = np.concatenate([np.zeros(50, bool), np.ones(950, bool)])
    pol = fit_policy(conf, agree, np.zeros(1000), budget=0.02, candidates=[0.99, 0.95, 0.5])
    assert not pol.usable and pol.expected_coverage == 0.0   # only 0.99 passed, and it answers nothing


def test_threshold_grid_is_fixed_and_strictest_first():
    from jevstiller.calibrate import threshold_grid
    g = threshold_grid(5)
    assert (np.diff(g) < 0).all() and g[0] > 0.999 and abs(g[-1] - 0.2) < 1e-9


def test_ood_threshold_comes_from_the_reference_itself():
    from jevstiller.ood import KnnOOD
    rng = np.random.default_rng(0)
    X = rng.normal(size=(400, 16)) + np.repeat(np.eye(16)[:2] * 6, 200, axis=0)
    X /= np.linalg.norm(X, axis=1, keepdims=True)
    o = KnnOOD(5)
    o.fit(X)
    t = o.threshold(0.99)
    near = o.score(X[:10] + 0.01)
    far = o.score(-X[:10])                               # the opposite direction: nothing like it seen
    assert np.isfinite(t) and (near <= t).all() and (far > t).all()
    tiny = KnnOOD(5)
    tiny.fit(X[:3])
    assert tiny.threshold(0.99) == float("inf")
