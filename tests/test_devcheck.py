import numpy as np
import pytest

import devcheck


def test_extract_usage_handles_chat_shape():
    usage = {"prompt_tokens": 1921, "completion_tokens": 87, "cost": 0.004712,
              "completion_tokens_details": {"reasoning_tokens": 84}}
    out = devcheck.extract_usage(usage)
    assert out == {"prompt_tokens": 1921, "completion_tokens": 87,
                   "reasoning_tokens": 84, "cost": 0.004712}


def test_extract_usage_handles_decisions_shape():
    # Arm J's response shape: input_tokens/output_tokens, no reasoning
    # breakdown, cost reported directly.
    usage = {"input_tokens": 1038, "output_tokens": 78, "cost": 4.3596e-05}
    out = devcheck.extract_usage(usage)
    assert out["prompt_tokens"] == 1038
    assert out["completion_tokens"] == 78
    assert out["reasoning_tokens"] == 0
    assert out["cost"] == pytest.approx(4.3596e-05)


def test_extract_usage_prefers_reported_cost_when_present():
    # Even if it disagreed with a tokens*price estimate, the reported cost
    # wins -- it is never recomputed from tokens.
    usage = {"prompt_tokens": 100, "completion_tokens": 10, "cost": 0.5}
    assert devcheck.extract_usage(usage)["cost"] == 0.5


def test_extract_usage_tolerates_missing_usage():
    assert devcheck.extract_usage(None)["prompt_tokens"] == 0
    assert devcheck.extract_usage({})["cost"] is None


def _ql_record(top1_token, top_logprobs_tokens):
    return {"response": {"choices": [{"logprobs": {"content": [{
        "token": top1_token,
        "top_logprobs": [{"token": t, "logprob": -1.0}
                         for t in top_logprobs_tokens],
    }]}}]}}


def test_letter_checks_does_not_flag_glued_vocabulary_tokens():
    # The old (wrong) check flagged any top-k token whose FIRST CHARACTER
    # was in A-G -- "cs" starts with C, "AI" starts with A, "Based" starts
    # with B, "Claude" starts with C, "CV" starts with C. None of these are
    # single-character letter tokens and none should count toward any
    # letter's bare-token tally.
    records = [_ql_record("D", ["D", "cs", "AI", "Based", "Claude", "CV"])]
    result = devcheck.letter_checks(records)
    assert result["bare_counts"] == {"D": 1}
    for false_positive_letter in ("A", "B", "C"):
        assert false_positive_letter not in result["bare_counts"]


def test_letter_checks_all_seven_requires_every_letter_present():
    records = [_ql_record(letter, [letter]) for letter in "ABCDEF"]  # missing G
    result = devcheck.letter_checks(records)
    assert not result["all_seven_single_token"]
    records.append(_ql_record("G", ["G"]))
    result = devcheck.letter_checks(records)
    assert result["all_seven_single_token"]


def test_letter_checks_counts_top1_bare_letter_choices():
    records = [_ql_record("D", ["D", "A"]), _ql_record("cs", ["cs", "D"])]
    result = devcheck.letter_checks(records)
    # Only the first record's top-1 token is a bare letter.
    assert result["top1_bare_total"] == 1
    assert result["top1_counts"] == {"D": 1}
    assert result["n_with_logprobs"] == 2


def test_letter_position_bias_uniform_gives_high_p_value():
    uniform_counts = {letter: 100 for letter in devcheck.LETTERS}
    stat, p, obs = devcheck.letter_position_bias(uniform_counts)
    assert stat == pytest.approx(0.0, abs=1e-9)
    assert p == pytest.approx(1.0, abs=1e-9)


def test_letter_position_bias_skewed_gives_low_p_value():
    skewed = {"A": 18, "B": 26, "C": 13, "D": 49, "E": 24, "F": 13, "G": 57}
    stat, p, obs = devcheck.letter_position_bias(skewed)
    assert p < 0.001


def test_shuffle_uniformity_flags_a_biased_shuffle():
    # A degenerate "shuffle" that always keeps every label at its own index
    # (identity permutation for every item) should fail hard against
    # uniform: every count piles onto the diagonal.
    items = [{"options": ["cs.CV", "cs.LG", "cs.AI", "cs.RO", "cs.CL",
                          "cs.CR", "cs.IR"]} for _ in range(50)]
    table, stat, p = devcheck.shuffle_uniformity(items)
    assert p < 1e-6
    assert table.trace() == 50 * 7  # every label always at its own position


def test_cost_extrapolation_multiplies_by_pool_size_ratio():
    rows = [{"arm": "J", "n": 200, "cost_total": 1.0},
            {"arm": "F1-V", "n": 200, "cost_total": 2.0}]
    per_arm, total_projected, j_projected, grand_total = \
        devcheck.cost_extrapolation(rows, target_n=2000)
    j_row = next(r for r in per_arm if r["arm"] == "J")
    f1v_row = next(r for r in per_arm if r["arm"] == "F1-V")
    assert j_row["factor"] == pytest.approx(10.0)
    assert j_row["projected_cost"] == pytest.approx(10.0)
    assert f1v_row["projected_cost"] == pytest.approx(20.0)
    assert total_projected == pytest.approx(30.0)
    # Grand total adds one more J re-run (PC2) on top of the summed arms.
    assert j_projected == pytest.approx(10.0)
    assert grand_total == pytest.approx(40.0)


def test_cost_extrapolation_respects_a_different_dev_pool_size():
    # If dev had been 100 items instead of 200, the factor to reach 2000
    # should be 20x, not hardcoded at 10x.
    rows = [{"arm": "J", "n": 100, "cost_total": 1.0}]
    per_arm, total_projected, j_projected, grand_total = \
        devcheck.cost_extrapolation(rows, target_n=2000)
    assert per_arm[0]["factor"] == pytest.approx(20.0)
    assert per_arm[0]["projected_cost"] == pytest.approx(20.0)


def test_per_class_accuracy_isolates_one_class(tmp_path, monkeypatch):
    # Build a tiny synthetic preds/X_dev.npz and confirm per_class_accuracy
    # reads it back correctly, rather than trusting real data alone.
    monkeypatch.chdir(tmp_path)
    (tmp_path / "preds").mkdir()
    import protocol as P
    labels = np.array(["cs.AI", "cs.AI", "cs.LG", "cs.LG"])
    probs = np.zeros((4, P.K))
    # Predict cs.AI correctly once, wrong once; cs.LG correct both times.
    probs[0, P.LABELS.index("cs.AI")] = 1.0
    probs[1, P.LABELS.index("cs.LG")] = 1.0
    probs[2, P.LABELS.index("cs.LG")] = 1.0
    probs[3, P.LABELS.index("cs.LG")] = 1.0
    np.savez(tmp_path / "preds" / "J_dev.npz", ids=np.arange(4), probs=probs,
             statuses=np.array(["ok"] * 4), labels=labels)
    monkeypatch.setattr(devcheck, "ARMS", ["J"])
    result = devcheck.per_class_accuracy()
    assert result["J"]["cs.AI"] == pytest.approx(0.5)
    assert result["J"]["cs.LG"] == pytest.approx(1.0)
