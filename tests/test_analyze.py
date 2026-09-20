import numpy as np
import pytest

import analyze
import protocol as P


# --------------------------------------------------------------------------
# Temperature fitting
# --------------------------------------------------------------------------

def _make_calibration_data(true_T, n=4000, k=3, seed=0):
    """Synthetic well-calibrated q_true (Dirichlet draws), labels sampled
    FROM q_true (so some items are genuinely wrong -- without that, the
    argmax is always the true label and the NLL-minimizing T degenerates to
    the sharpest allowed value regardless of true_T). The OBSERVED
    (miscalibrated) probs are p = normalize(q_true^true_T) -- the exact
    inverse of section 6's model q = normalize(p^(1/T)) -- so
    fit_temperature(p, y) should recover true_T."""
    rng = np.random.default_rng(seed)
    q_true = rng.dirichlet(np.full(k, 1.5), size=n)
    cdf = np.cumsum(q_true, axis=1)
    u = rng.uniform(size=(n, 1))
    y_idx = (u < cdf).argmax(axis=1)
    p = q_true ** true_T
    p = p / p.sum(axis=1, keepdims=True)
    p = np.vstack([P.floor_renorm(row) for row in p])
    return p, y_idx


def test_temperature_fitting_recovers_known_T():
    p, y_idx = _make_calibration_data(true_T=3.0)
    T_hat = analyze.fit_temperature(p, y_idx)
    assert T_hat == pytest.approx(3.0, abs=0.3)


def test_temperature_fitting_recovers_known_T_other_direction():
    p, y_idx = _make_calibration_data(true_T=0.3)
    T_hat = analyze.fit_temperature(p, y_idx)
    assert T_hat == pytest.approx(0.3, abs=0.05)


def test_apply_temperature_T_equals_one_is_a_no_op():
    rng = np.random.default_rng(1)
    raw = rng.dirichlet(np.ones(P.K), size=25)
    raw = np.vstack([P.floor_renorm(row) for row in raw])
    scaled = analyze.apply_temperature(raw, 1.0)
    assert scaled == pytest.approx(raw, abs=1e-9)


# --------------------------------------------------------------------------
# ECE
# --------------------------------------------------------------------------

def test_ece_zero_for_perfectly_calibrated_set():
    # 15 confidence levels j/20 (j=1..15), each with EXACTLY j of every 20
    # items correct -- chosen so avg_confidence == avg_accuracy exactly in
    # every one of the 15 equal-mass bins, no rounding slack.
    rng = np.random.default_rng(2)
    per_level = 20
    conf, correct = [], []
    for j in range(1, 16):
        lvl = j / per_level
        c = np.array([True] * j + [False] * (per_level - j))
        rng.shuffle(c)
        conf.extend([lvl] * per_level)
        correct.extend(c.tolist())
    conf = np.array(conf)
    correct = np.array(correct)
    ece, _ = analyze.ece_equal_mass(conf, correct, n_bins=15)
    assert ece == pytest.approx(0.0, abs=1e-9)


def test_ece_large_for_confidently_wrong_set():
    conf = np.full(300, 0.99)
    correct = np.zeros(300, dtype=bool)  # always wrong despite high confidence
    ece, _ = analyze.ece_equal_mass(conf, correct, n_bins=15)
    assert ece == pytest.approx(0.99, abs=1e-6)


# --------------------------------------------------------------------------
# Brier
# --------------------------------------------------------------------------

def test_brier_matches_hand_computed_value():
    probs = np.array([
        [0.7, 0.2, 0.1],
        [0.2, 0.2, 0.6],
    ])
    y_idx = np.array([0, 2])
    # item 0: (0.7-1)^2 + (0.2-0)^2 + (0.1-0)^2 = 0.09+0.04+0.01 = 0.14
    # item 1: (0.2-0)^2 + (0.2-0)^2 + (0.6-1)^2 = 0.04+0.04+0.16 = 0.24
    # mean = 0.19
    assert analyze.multiclass_brier(probs, y_idx) == pytest.approx(0.19)


def test_brier_per_item_matches_mean():
    probs = np.array([[1.0, 0.0], [0.5, 0.5]])
    y_idx = np.array([0, 1])
    items = analyze.brier_per_item(probs, y_idx)
    assert items[0] == pytest.approx(0.0)
    assert items[1] == pytest.approx(0.5)
    assert np.mean(items) == pytest.approx(analyze.multiclass_brier(probs, y_idx))


# --------------------------------------------------------------------------
# tau fitting
# --------------------------------------------------------------------------

def test_tau_infinite_when_k_below_min():
    # Only 40 items total; even accepting all of them can't reach k=50.
    rng = np.random.default_rng(3)
    conf = rng.uniform(0.5, 1.0, size=40)
    correct = np.ones(40, dtype=bool)  # perfect -- still can't hit min_k
    tau = analyze.fit_tau(conf, correct, target_error=0.05, min_k=50)
    assert tau == float("inf")


def test_tau_finite_and_coverage_positive_when_k_exceeds_min():
    rng = np.random.default_rng(4)
    n = 500
    conf = rng.uniform(0.0, 1.0, size=n)
    # correctness strongly increasing in confidence, so a large high-confidence
    # prefix has low error
    correct = rng.uniform(0, 1, size=n) < conf
    tau = analyze.fit_tau(conf, correct, target_error=0.05, min_k=50)
    assert np.isfinite(tau)
    accepted = conf >= tau
    assert accepted.sum() >= 50
    error_rate = (~correct[accepted]).mean()
    assert error_rate <= 0.05 + 1e-9


def test_tau_zero_coverage_when_infinite():
    conf = np.array([0.9] * 200)
    correct = np.array([False] * 200)  # 100% error at every k
    tau = analyze.fit_tau(conf, correct, target_error=0.05, min_k=50)
    assert tau == float("inf")
    accepted = conf >= tau
    assert accepted.sum() == 0


# --------------------------------------------------------------------------
# Bootstrap reproducibility
# --------------------------------------------------------------------------

def test_bootstrap_reproducible_under_fixed_seed():
    rng = np.random.default_rng(5)
    n = 200
    a = rng.normal(0.3, 0.1, size=n)
    b = rng.normal(0.25, 0.1, size=n)
    idx1 = analyze.build_resample_matrix(n, 500, seed=42)
    idx2 = analyze.build_resample_matrix(n, 500, seed=42)
    assert np.array_equal(idx1, idx2)
    r1 = analyze.bootstrap_diff_ci(a, b, idx1)
    r2 = analyze.bootstrap_diff_ci(a, b, idx2)
    assert r1 == r2


def test_bootstrap_different_seed_gives_different_matrix():
    idx1 = analyze.build_resample_matrix(100, 50, seed=1)
    idx2 = analyze.build_resample_matrix(100, 50, seed=2)
    assert not np.array_equal(idx1, idx2)


# --------------------------------------------------------------------------
# Cross-fitting never leaks a fold's own data into its own fitted parameter
# --------------------------------------------------------------------------

def test_cross_fitting_fold_T_varies_and_excludes_own_fold():
    """Construct 5 blocks of items, each block internally miscalibrated at a
    distinct, well-separated temperature. Assign block i to fold i directly
    (bypassing make_folds' randomness, since we need a KNOWN fold structure).
    If the fold i's fitted T depended on fold i's own (excluded) data, it
    would be pulled toward block i's distinctive temperature; since it must
    be fit on the other four blocks only, holding out different folds must
    change the fitted T (the leave-one-block-out training mixture differs
    every time), and none of the fold_T values may equal the temperature
    that would fit fold i's own block best in isolation.
    """
    rng = np.random.default_rng(7)
    k = 3
    block_Ts = [0.15, 1.0, 4.0, 10.0, 18.0]
    n_per_item = 300
    all_p, all_y, all_fold = [], [], []
    for block_idx, T in enumerate(block_Ts):
        # Labels are SAMPLED from q_true (real prediction noise) rather than
        # forced to equal argmax(q_true): without that, every item's true
        # label always matches the ranking regardless of T, and the
        # NLL-minimizing T degenerates to the sharpest allowed value for
        # every block, making solo-fit and cross-fit indistinguishable for
        # the wrong reason.
        q_true = rng.dirichlet(np.full(k, 1.5), size=n_per_item)
        cdf = np.cumsum(q_true, axis=1)
        u = rng.uniform(size=(n_per_item, 1))
        y_idx = (u < cdf).argmax(axis=1)
        p = q_true ** T
        p = p / p.sum(axis=1, keepdims=True)
        p = np.vstack([P.floor_renorm(row) for row in p])
        all_p.append(p)
        all_y.append(y_idx)
        all_fold.append(np.full(n_per_item, block_idx))

    probs = np.vstack(all_p)
    y_idx = np.concatenate(all_y)
    folds = np.concatenate(all_fold)

    # Fit T per fold "in isolation" (using ONLY that fold's own data) as the
    # forbidden reference point.
    solo_T = {}
    for block_idx, T in enumerate(block_Ts):
        mask = folds == block_idx
        solo_T[block_idx] = analyze.fit_temperature(probs[mask], y_idx[mask])

    result = analyze.cross_fit_arm(probs, y_idx, folds, n_folds=5)
    fold_T = result["fold_T"]

    # The out-of-fold fitted T must differ from what fitting on the held-out
    # fold's OWN data alone would give -- the whole point of cross-fitting.
    for block_idx in range(5):
        assert fold_T[block_idx] != pytest.approx(solo_T[block_idx], rel=1e-2), (
            f"fold {block_idx}'s out-of-fold T matches its own solo-fit T -- "
            "looks like the fold saw its own data")

    # And, since each held-out fold changes which four blocks are pooled for
    # training, the fitted T values across folds must not all collapse to one
    # number.
    assert len(set(round(t, 2) for t in fold_T)) > 1
