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


def test_floor_renorm_sums_to_one_and_respects_epsilon():
    out = P.floor_renorm(np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]))
    assert out.sum() == pytest.approx(1.0)
    assert (out >= P.EPSILON * 0.999).all()


def test_floor_renorm_handles_all_zero_vector():
    out = P.floor_renorm(np.zeros(7))
    assert out.sum() == pytest.approx(1.0)
    assert out == pytest.approx(np.full(7, 1 / 7))


def test_floor_renorm_is_idempotent():
    once = P.floor_renorm(np.array([0.9, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0]))
    assert P.floor_renorm(once) == pytest.approx(once)


def test_floor_renorm_preserves_argmax():
    out = P.floor_renorm(np.array([0.05, 0.60, 0.35, 0.0, 0.0, 0.0, 0.0]))
    assert int(out.argmax()) == 1


def test_arms_registry_pins_every_llm_arm():
    for arm in ["J", "F1-V", "F2-V", "Q-V", "Q-L"]:
        assert P.ARMS[arm]["provider"], f"{arm} must pin a provider"
    assert P.ARMS["Q-V"]["provider"] == P.ARMS["Q-L"]["provider"] == "Parasail"
    assert P.ARMS["Q-V"]["model"] == P.ARMS["Q-L"]["model"]
    assert P.ARMS["J"]["model"] == "typesafe/jev-1.13"
