import numpy as np
import pytest
import protocol as P
import pools


@pytest.fixture(scope="module")
def derived():
    return pools.derive_pools(P.in_set(P.load_rows()))


def test_pool_sizes_hit_the_protocol_exactly(derived):
    assert len(derived["evaluation"]) == 2000
    assert len(derived["dev"]) == 200
    assert len(derived["anchor"]) == P.EXPECTED["anchor"]


def test_pools_are_disjoint(derived):
    e = {r["id"] for r in derived["evaluation"]}
    d = {r["id"] for r in derived["dev"]}
    a = {r["id"] for r in derived["anchor"]}
    assert not (e & d) and not (e & a) and not (d & a)


def test_evaluation_is_all_post_start_and_others_all_pre_start(derived):
    assert all(r["published"][:10] >= P.START_DATE for r in derived["evaluation"])
    for name in ("dev", "anchor"):
        assert all(r["published"][:10] < P.START_DATE for r in derived[name])


def test_derivation_is_reproducible(derived):
    again = pools.derive_pools(P.in_set(P.load_rows()))
    for name in ("evaluation", "dev", "anchor"):
        assert [r["id"] for r in again[name]] == [r["id"] for r in derived[name]]


def test_every_item_carries_a_full_permutation_of_the_seven_labels(derived):
    for r in derived["evaluation"][:50]:
        assert sorted(r["options"]) == sorted(P.LABELS)
        assert len(r["options"]) == 7


def test_permutation_is_stable_per_item_and_varies_across_items(derived):
    first = derived["evaluation"][0]
    assert pools.permutation_for(first["id"]) == first["options"]
    orders = {tuple(r["options"]) for r in derived["evaluation"][:200]}
    assert len(orders) > 100, "shuffle should not collapse to a few orders"


def test_shuffle_round_trip_is_identity(derived):
    opts = derived["evaluation"][0]["options"]
    vec = np.arange(7, dtype=float)
    shuffled = pools.to_shuffled(vec, opts)
    assert pools.to_canonical(shuffled, opts) == pytest.approx(vec)


def test_gold_label_is_the_primary_category(derived):
    assert all(r["label"] in P.LABELS for r in derived["evaluation"])


def test_realized_class_counts_are_near_the_protocol_expectations(derived):
    from collections import Counter
    c = Counter(r["label"] for r in derived["evaluation"])
    expected = {"cs.CV": 442, "cs.LG": 429, "cs.AI": 343,
                "cs.RO": 292, "cs.CL": 291, "cs.CR": 160, "cs.IR": 44}
    for label, exp in expected.items():
        assert abs(c[label] - exp) <= 40, f"{label}: {c[label]} vs ~{exp}"
