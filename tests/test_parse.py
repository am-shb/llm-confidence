import json
import numpy as np
import pytest
import protocol as P
import parse


ITEM = {
    "id": "x",
    "label": "cs.LG",
    "options": ["cs.RO", "cs.IR", "cs.LG", "cs.CV", "cs.CR", "cs.AI", "cs.CL"],
}
UNIFORM = np.full(P.K, 1 / P.K)


def _verbalized(content, finish="stop"):
    return {"id": "x", "arm": "Q-V", "status": "ok",
            "response": {"choices": [{"finish_reason": finish,
                                      "message": {"content": content}}]}}


def test_verbalized_maps_keys_to_canonical_order():
    payload = {"cs.RO": 0.0, "cs.IR": 0.0, "cs.LG": 0.7, "cs.CV": 0.3,
               "cs.CR": 0.0, "cs.AI": 0.0, "cs.CL": 0.0}
    vec, status = parse.parse_record(_verbalized(json.dumps(payload)), ITEM)
    assert status == "ok"
    assert vec.sum() == pytest.approx(1.0)
    assert int(vec.argmax()) == P.LABELS.index("cs.LG")
    assert vec[P.LABELS.index("cs.CV")] > vec[P.LABELS.index("cs.AI")]


def test_verbalized_unnormalized_input_is_renormalized():
    payload = {l: 2.0 for l in P.LABELS}
    vec, status = parse.parse_record(_verbalized(json.dumps(payload)), ITEM)
    assert status == "ok"
    assert vec.sum() == pytest.approx(1.0)


def test_epsilon_floor_is_applied():
    payload = {l: 0.0 for l in P.LABELS}
    payload["cs.LG"] = 1.0
    vec, _ = parse.parse_record(_verbalized(json.dumps(payload)), ITEM)
    # A single application of protocol.floor_renorm leaves floored entries at
    # eps / (1 + (K-1)*eps), which is slightly BELOW eps (0.0009940... at the
    # default epsilon for K=7) -- not at eps itself. A threshold of 0.999*eps
    # is tighter than that true value and would reject a correct single-floor
    # implementation while accepting a double-floored one (double-flooring
    # pushes the floored entries back up toward eps, since eps/(1+(K-1)*eps)
    # < eps, so a second floor() call raises them again). Compare against the
    # derived formula instead of an arbitrary near-eps fraction.
    floor = P.EPSILON / (1 + (P.K - 1) * P.EPSILON)
    assert (vec >= floor - 1e-12).all()


def test_malformed_json_is_a_failure_scored_uniform():
    vec, status = parse.parse_record(_verbalized("not json at all"), ITEM)
    assert status == "parse_error"
    assert vec == pytest.approx(UNIFORM)


def test_missing_key_is_a_failure():
    payload = {l: 1 / 6 for l in P.LABELS if l != "cs.IR"}
    vec, status = parse.parse_record(_verbalized(json.dumps(payload)), ITEM)
    assert status == "parse_error"
    assert vec == pytest.approx(UNIFORM)


def test_negative_and_nonfinite_values_are_failures():
    for bad in (-0.5, float("inf"), float("nan")):
        payload = {l: 0.1 for l in P.LABELS}
        payload["cs.LG"] = bad
        vec, status = parse.parse_record(
            _verbalized(json.dumps(payload).replace("NaN", '"NaN"')), ITEM)
        assert status in ("parse_error", "bad_value")
        assert vec == pytest.approx(UNIFORM)


def test_truncation_is_a_failure():
    payload = {l: 1 / 7 for l in P.LABELS}
    vec, status = parse.parse_record(
        _verbalized(json.dumps(payload), finish="length"), ITEM)
    assert status == "truncated"
    assert vec == pytest.approx(UNIFORM)


def test_refusal_is_a_failure():
    rec = {"id": "x", "arm": "F1-V", "status": "ok",
           "response": {"choices": [{"finish_reason": "content_filter",
                                     "message": {"refusal": "no"}}]}}
    vec, status = parse.parse_record(rec, ITEM)
    assert status == "refusal"
    assert vec == pytest.approx(UNIFORM)


def test_runner_level_failure_is_scored_uniform():
    rec = {"id": "x", "arm": "J", "status": "failure", "response": None}
    vec, status = parse.parse_record(rec, ITEM)
    assert status == "timeout"
    assert vec == pytest.approx(UNIFORM)


def _letter(top):
    return {"id": "x", "arm": "Q-L", "status": "ok",
            "response": {"choices": [{
                "finish_reason": "stop",
                "message": {"content": top[0][0]},
                "logprobs": {"content": [{
                    "top_logprobs": [{"token": t, "logprob": lp}
                                     for t, lp in top]}]}}]}}


def test_letter_renormalizes_over_the_seven_label_tokens():
    top = [("C", -0.1), ("A", -2.0), ("D", -3.0), (" the", -5.0)]
    vec, status = parse.parse_record(_letter(top), ITEM)
    assert status == "ok"
    assert vec.sum() == pytest.approx(1.0)
    # C is position 2 in options -> cs.LG
    assert int(vec.argmax()) == P.LABELS.index("cs.LG")


def test_letter_labels_absent_from_topk_get_zero_before_the_floor():
    top = [("C", -0.1), ("A", -2.0)]
    vec, status = parse.parse_record(_letter(top), ITEM)
    assert status == "ok"
    # Only A and C were returned; the other five sit at the floor.
    floored = [v for v in vec if v <= P.EPSILON * 1.01]
    assert len(floored) == 5


def test_letter_outside_a_to_g_is_a_failure():
    # Fix round 1: the emitted-text fallback that used to classify this case
    # as "bad_letter" was removed (it fabricated confidence off raw text --
    # see test_letter_with_no_valid_letter_in_topk_is_a_counted_failure).
    # With no valid A-G token anywhere in top-k, there is no distribution to
    # read at all, so this now falls under "no_logprobs" like any other
    # unreadable top-k. "bad_letter" stays in FAILURE_KINDS defensively but is
    # no longer produced by _parse_letter.
    vec, status = parse.parse_record(_letter([("Z", -0.1)]), ITEM)
    assert status == "no_logprobs"
    assert status in parse.FAILURE_KINDS
    assert vec == pytest.approx(UNIFORM)


def test_letter_ignores_case_and_surrounding_whitespace():
    vec, status = parse.parse_record(_letter([(" c ", -0.1), ("A", -2.0)]), ITEM)
    assert status == "ok"
    assert int(vec.argmax()) == P.LABELS.index("cs.LG")


def test_letter_with_no_valid_letter_in_topk_is_a_counted_failure():
    """Section 4 ties Q-L to the first token's logprobs. If top-k holds no
    valid letter there is no distribution to read, so this is a failure scored
    uniform -- NOT an excuse to read a probability off the emitted text, which
    would fabricate confidence the model never expressed.
    """
    rec = {"id": "x", "arm": "Q-L", "status": "ok",
           "response": {"choices": [{
               "finish_reason": "stop",
               "message": {"content": "C"},          # a valid letter in the TEXT
               "logprobs": {"content": [{"top_logprobs": [
                   {"token": " the", "logprob": -0.1},
                   {"token": "\n", "logprob": -2.0}]}]}}]}}   # but none in top-k
    vec, status = parse.parse_record(rec, ITEM)
    assert status == "no_logprobs"
    assert vec == pytest.approx(UNIFORM), \
        "must be uniform, not a fabricated one-hot on the emitted letter"
    assert status in parse.FAILURE_KINDS


def test_no_parse_path_ever_returns_a_one_hot_from_emitted_text():
    """Guard the general property: nothing may invent a maximally confident
    vector from text when logprobs are unreadable.
    """
    for emitted in ("A", "G", "d"):
        rec = {"id": "x", "arm": "Q-L", "status": "ok",
               "response": {"choices": [{
                   "finish_reason": "stop",
                   "message": {"content": emitted},
                   "logprobs": {"content": [{"top_logprobs": [
                       {"token": "zzz", "logprob": -0.1}]}]}}]}}
        vec, status = parse.parse_record(rec, ITEM)
        assert status in parse.FAILURE_KINDS
        assert vec.max() < 0.5, f"emitted {emitted!r} produced a confident vector"


def test_duplicated_identical_logprob_entry_does_not_double_count():
    single = [("C", -0.1), ("A", -2.0)]
    dup = [("C", -0.1), ("C", -0.1), ("A", -2.0)]
    vec_single, status_single = parse.parse_record(_letter(single), ITEM)
    vec_dup, status_dup = parse.parse_record(_letter(dup), ITEM)
    assert status_single == status_dup == "ok"
    assert vec_single == pytest.approx(vec_dup)


def test_empty_content_is_a_counted_failure():
    for content in ("", "   ", None):
        rec = {"id": "x", "arm": "Q-V", "status": "ok",
               "response": {"choices": [{"finish_reason": "stop",
                                         "message": {"content": content}}]}}
        vec, status = parse.parse_record(rec, ITEM)
        assert status in parse.FAILURE_KINDS
        assert vec == pytest.approx(UNIFORM)


def test_record_with_no_choices_at_all_is_a_counted_failure():
    rec = {"id": "x", "arm": "Q-V", "status": "ok", "response": {}}
    vec, status = parse.parse_record(rec, ITEM)
    assert status in parse.FAILURE_KINDS
    assert vec == pytest.approx(UNIFORM)


def test_jev_bad_values_are_counted_failures():
    """J is the subject arm; a malformed probability from it must not slip
    through as data.
    """
    base = {"cs.CV": 0.5, "cs.LG": 0.5, "cs.AI": 0, "cs.RO": 0,
            "cs.CL": 0, "cs.CR": 0, "cs.IR": 0}
    for bad_label, bad_value in (("cs.AI", -0.2), ("cs.AI", float("inf"))):
        probs = dict(base); probs[bad_label] = bad_value
        rec = {"id": "x", "arm": "J", "status": "ok",
               "response": {"answers": {P.JEV_QUESTION_KEY: {
                   "type": "choice", "choice": "cs.CV",
                   "probabilities": probs, "confidence": 0.5}}}}
        vec, status = parse.parse_record(rec, ITEM)
        assert status in parse.FAILURE_KINDS
        assert vec == pytest.approx(UNIFORM)


def test_jev_missing_or_wrong_keys_is_a_parse_error():
    for probs in ({"cs.CV": 1.0}, {}, None):
        rec = {"id": "x", "arm": "J", "status": "ok",
               "response": {"answers": {P.JEV_QUESTION_KEY: {
                   "type": "choice", "probabilities": probs}}}}
        vec, status = parse.parse_record(rec, ITEM)
        assert status in parse.FAILURE_KINDS


def test_every_failure_kind_is_a_known_kind():
    assert "parse_error" in parse.FAILURE_KINDS
    assert "ok" not in parse.FAILURE_KINDS


def test_tag_leakage_is_detected_and_does_not_by_itself_fail_the_item():
    payload = {l: 1 / 7 for l in P.LABELS}
    content = "<thinking>hmm</thinking>" + json.dumps(payload)
    vec, status = parse.parse_record(_verbalized(content), ITEM)
    assert status == "ok"
    assert parse.has_tag_leakage(content)
    assert not parse.has_tag_leakage(json.dumps(payload))
