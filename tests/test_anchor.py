"""Real tests for the three non-LLM reference arms (PROTOCOL.md section 3).

A -- TF-IDF + multinomial logistic regression, fit on the anchor pool only.
P -- anchor-pool base rates, constant, degenerate by design (section 7).
P* -- evaluation-pool base rates, constant, an oracle (sections 3 and 8).
"""

from collections import Counter

import numpy as np
import pytest

import anchor
import pools
import protocol as P


@pytest.fixture(scope="module")
def anchor_items():
    return pools.load_pool("anchor")


@pytest.fixture(scope="module")
def evaluation_items():
    return pools.load_pool("evaluation")


@pytest.fixture(scope="module")
def dev_items():
    return pools.load_pool("dev")


@pytest.fixture(scope="module")
def arm_p_eval(evaluation_items):
    return anchor.build("P", "evaluation")


@pytest.fixture(scope="module")
def arm_pstar_eval(evaluation_items):
    return anchor.build("P-star", "evaluation")


@pytest.fixture(scope="module")
def arm_a_eval(evaluation_items):
    return anchor.build("A", "evaluation")


# ---------------------------------------------------------------------------
# Arm P: anchor-pool base rates, constant.
# ---------------------------------------------------------------------------

def test_p_vector_identical_for_every_item(arm_p_eval):
    ids, probs, statuses, labels, extra = arm_p_eval
    first_row = probs[0]
    assert np.allclose(probs, first_row)


def test_p_vector_sums_to_one(arm_p_eval):
    ids, probs, statuses, labels, extra = arm_p_eval
    assert np.allclose(probs.sum(axis=1), 1.0)


def test_p_vector_matches_anchor_frequencies_after_flooring(arm_p_eval, anchor_items):
    ids, probs, statuses, labels, extra = arm_p_eval
    counts = np.zeros(P.K)
    for it in anchor_items:
        counts[P.LABELS.index(it["label"])] += 1
    expected = P.floor_renorm(counts / counts.sum())
    assert np.allclose(probs[0], expected)


def test_p_argmax_is_constant_across_all_items(arm_p_eval):
    """Regression guard for near-zero resolution (section 7): P must predict
    the same label for every single item, always."""
    ids, probs, statuses, labels, extra = arm_p_eval
    argmaxes = np.argmax(probs, axis=1)
    assert len(set(argmaxes.tolist())) == 1


# ---------------------------------------------------------------------------
# Arm P*: evaluation-pool base rates, constant, an oracle.
# ---------------------------------------------------------------------------

def test_pstar_vector_matches_evaluation_frequencies(arm_pstar_eval, evaluation_items):
    ids, probs, statuses, labels, extra = arm_pstar_eval
    counts = np.zeros(P.K)
    for it in evaluation_items:
        counts[P.LABELS.index(it["label"])] += 1
    expected = P.floor_renorm(counts / counts.sum())
    assert np.allclose(probs[0], expected)


def test_pstar_differs_from_p(arm_p_eval, arm_pstar_eval):
    """Section 2 documents a non-stationary class mix (the cs.RO ramp), so
    the anchor-pool and evaluation-pool base rates must genuinely differ --
    this is not supposed to be a no-op oracle."""
    _, p_probs, _, _, _ = arm_p_eval
    _, pstar_probs, _, _, _ = arm_pstar_eval
    assert not np.allclose(p_probs[0], pstar_probs[0])
    # The documented direction: cs.RO's share is higher in evaluation.
    ro = P.LABELS.index("cs.RO")
    assert pstar_probs[0, ro] > p_probs[0, ro]


def test_pstar_report_is_labelled_an_oracle():
    ids, probs, statuses, labels, extra = anchor.build("P-star", "dev")
    assert extra["oracle"] is True
    assert "oracle" in extra["note"].lower()


# ---------------------------------------------------------------------------
# Arm A: TF-IDF + multinomial logistic regression, anchor-only.
# ---------------------------------------------------------------------------

def test_a_trains_only_on_anchor_pool_items(monkeypatch, anchor_items,
                                             evaluation_items):
    captured = {}
    real_fit = anchor.fit_arm_a

    def spy(items):
        captured["items"] = items
        return real_fit(items)

    monkeypatch.setattr(anchor, "fit_arm_a", spy)
    anchor.build("A", "evaluation")

    trained_ids = {it["id"] for it in captured["items"]}
    evaluation_ids = {it["id"] for it in evaluation_items}
    anchor_ids = {it["id"] for it in anchor_items}

    assert trained_ids == anchor_ids
    assert not (trained_ids & evaluation_ids)


def test_a_beats_majority_class_on_evaluation(arm_a_eval, evaluation_items):
    ids, probs, statuses, labels, extra = arm_a_eval
    majority_rate = max(Counter(labels.tolist()).values()) / len(labels)
    assert extra["accuracy"] > majority_rate


def test_a_report_carries_hyperparameters_and_anchor_size(arm_a_eval, anchor_items):
    ids, probs, statuses, labels, extra = arm_a_eval
    assert extra["anchor_pool_size"] == len(anchor_items) == P.EXPECTED["anchor"]
    assert extra["vectorizer_params"]
    assert extra["classifier_params"]["random_state"] == P.SEEDS["master"]


# ---------------------------------------------------------------------------
# Cross-arm invariants: every arm goes through the same probability handling.
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("arm", ["A", "P", "P-star"])
@pytest.mark.parametrize("pool", ["evaluation", "dev"])
def test_every_row_sums_to_one_and_respects_the_floor(arm, pool):
    ids, probs, statuses, labels, extra = anchor.build(arm, pool)
    assert np.allclose(probs.sum(axis=1), 1.0, atol=1e-9)
    assert (probs >= P.EPSILON - 1e-12).all()
    assert np.all(statuses == "ok")


@pytest.mark.parametrize("arm", ["A", "P", "P-star"])
@pytest.mark.parametrize("pool", ["evaluation", "dev"])
def test_output_shape_and_ids_match_the_requested_pool(arm, pool):
    pool_items = pools.load_pool(pool)
    ids, probs, statuses, labels, extra = anchor.build(arm, pool)
    assert len(ids) == len(pool_items)
    assert probs.shape == (len(pool_items), P.K)
    assert set(ids.tolist()) == {it["id"] for it in pool_items}
