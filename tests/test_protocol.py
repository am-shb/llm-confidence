import numpy as np
import pytest
import protocol as P


def test_labels_are_the_seven_in_canonical_order():
    assert P.LABELS == ["cs.CV", "cs.LG", "cs.AI", "cs.RO", "cs.CL", "cs.CR", "cs.IR"]
    assert P.K == 7


def test_seeds_match_the_protocol():
    assert P.SEEDS == {
        "master": 20260919,
        "evaluation": 20260920,
        "dev": 20260921,
        "shuffle": 20260922,
        "folds": 20260923,
        "bootstrap": 20260924,
    }


def test_epsilon_and_start_date():
    assert P.EPSILON == 0.001
    assert P.START_DATE == "2026-09-05"


@pytest.mark.parametrize("eps", [0.0001, 0.001, 0.01])
def test_floor_renorm_sums_to_one_and_respects_the_exact_floor(eps):
    """Flooring adds at most K*eps before renormalizing, so the smallest
    entry lands at exactly eps/(1 + K*eps) -- just below eps, not above it.
    Section 7 recomputes the primary metric at all three of these epsilons.
    """
    exact_floor = eps / (1 + P.K * eps)
    for raw in ([1.0, 0, 0, 0, 0, 0, 0],
                [0.5, 0.5, 0, 0, 0, 0, 0],
                [1e-12, 1, 0, 0, 0, 0, 0]):
        out = P.floor_renorm(np.array(raw), eps=eps)
        assert out.sum() == pytest.approx(1.0)
        assert out.min() >= exact_floor * (1 - 1e-9)
        assert (out > 0).all(), "log-space operations must be defined everywhere"


def test_floor_renorm_handles_all_zero_vector():
    out = P.floor_renorm(np.zeros(7))
    assert out.sum() == pytest.approx(1.0)
    assert out == pytest.approx(np.full(7, 1 / 7))


def test_floor_renorm_converges_and_preserves_its_invariants():
    """Section 5 re-applies the floor after temperature scaling, so what
    matters is that re-application is well-defined and stable -- not that it
    is a no-op. Repeated application converges to a fixed point at min == eps.
    """
    once = P.floor_renorm(np.array([0.9, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0]))
    twice = P.floor_renorm(once)

    assert twice.sum() == pytest.approx(1.0)
    assert twice.min() >= P.EPSILON / (1 + P.K * P.EPSILON) * (1 - 1e-9)
    assert np.abs(twice - once).max() < 1e-4, "re-application must be stable"

    x = np.array([1.0, 0, 0, 0, 0, 0, 0])
    for _ in range(8):
        x = P.floor_renorm(x)
    assert x.min() == pytest.approx(P.EPSILON, rel=1e-6)
    assert np.abs(P.floor_renorm(x) - x).max() < 1e-12, "fixed point reached"


def test_floor_renorm_preserves_argmax():
    out = P.floor_renorm(np.array([0.05, 0.60, 0.35, 0.0, 0.0, 0.0, 0.0]))
    assert int(out.argmax()) == 1


def test_arms_registry_pins_every_llm_arm():
    for arm in ["J", "F1-V", "F2-V", "Q-V", "Q-L"]:
        assert P.ARMS[arm]["provider"], f"{arm} must pin a provider"
    assert P.ARMS["Q-V"]["provider"] == P.ARMS["Q-L"]["provider"] == "Parasail"
    assert P.ARMS["Q-V"]["model"] == P.ARMS["Q-L"]["model"]
    assert P.ARMS["J"]["model"] == "typesafe/jev-1.13"
