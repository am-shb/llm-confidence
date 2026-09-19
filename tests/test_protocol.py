import csv as _csv
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
    """A unit-sum vector always has an entry >= 1/K > eps, so at most K-1
    entries can be floored. The floored value after renormalization is
    therefore eps/(1 + (K-1)*eps) -- just below eps, and attained by a
    one-hot input. Section 7 recomputes the primary metric at all three
    of these epsilons.
    """
    tight_floor = eps / (1 + (P.K - 1) * eps)

    one_hot = P.floor_renorm(np.array([1.0, 0, 0, 0, 0, 0, 0]), eps=eps)
    assert one_hot.min() == pytest.approx(tight_floor), "bound is attained here"

    for raw in ([1.0, 0, 0, 0, 0, 0, 0],
                [0.5, 0.5, 0, 0, 0, 0, 0],
                [1e-12, 1, 0, 0, 0, 0, 0],
                [0.2, 0.2, 0.2, 0.2, 0.2, 0.0, 0.0]):
        out = P.floor_renorm(np.array(raw), eps=eps)
        assert out.sum() == pytest.approx(1.0)
        assert out.min() >= tight_floor * (1 - 1e-9)
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
    assert twice.min() >= P.EPSILON / (1 + (P.K - 1) * P.EPSILON) * (1 - 1e-9)
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
    expected = {"J": "TypeSafe", "F1-V": "Anthropic", "F2-V": "OpenAI",
                "Q-V": "Parasail", "Q-L": "Parasail"}
    for arm, provider in expected.items():
        assert P.ARMS[arm]["provider"] == provider
    assert P.ARMS["Q-V"]["model"] == P.ARMS["Q-L"]["model"]
    assert P.ARMS["J"]["model"] == "typesafe/jev-1.13"


def test_provider_block_pins_and_disables_fallbacks():
    for arm in P.LLM_ARMS:
        block = P.provider_block(arm)
        assert block["allow_fallbacks"] is False
        assert block["order"] == [P.ARMS[arm]["provider"]]
    assert P.provider_block("Q-V") == P.provider_block("Q-L"), \
        "C3 requires both Qwen arms on one identical backend"


def test_csv_sha256_matches_the_recorded_hash():
    assert P.csv_sha256(P.CSV_PATH) == P.CSV_SHA256


def test_load_rows_rejects_a_file_whose_hash_differs(tmp_path):
    bad = tmp_path / "bad.csv"
    bad.write_text("id,primary_category,published,abstract,title\n1,cs.AI,2026-09-06T00:00:00Z,x,y\n")
    with pytest.raises(ValueError, match="SHA-256"):
        P.load_rows(str(bad))


def test_load_rows_can_skip_verification_for_fixtures(tmp_path):
    f = tmp_path / "f.csv"
    f.write_text("id,primary_category,published,abstract,title\n1,cs.AI,2026-09-06T00:00:00Z,x,y\n")
    assert len(P.load_rows(str(f), verify=False)) == 1


def test_in_set_keeps_only_the_seven_primary_categories():
    rows = [{"primary_category": c} for c in ["cs.AI", "cs.SE", "cs.CV", "stat.ML"]]
    assert [r["primary_category"] for r in P.in_set(rows)] == ["cs.AI", "cs.CV"]


def test_real_csv_reproduces_the_protocol_totals():
    rows = P.in_set(P.load_rows())
    assert len(rows) == P.EXPECTED["in_set"]
    post = [r for r in rows if r["published"][:10] >= P.START_DATE]
    assert len(post) == P.EXPECTED["post_start"]
    pre = [r for r in rows if r["published"][:10] < P.START_DATE]
    assert len(pre) - P.POOL_SIZES["dev"] == P.EXPECTED["anchor"]


def test_integrity_report_finds_no_duplicates_or_short_abstracts():
    rep = P.integrity_report(P.in_set(P.load_rows()))
    assert rep["duplicate_ids"] == 0
    assert rep["short_abstracts"] == 0
    assert 185 <= rep["median_words"] <= 195
