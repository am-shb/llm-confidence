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
    # default epsilon for K=7) -- not at eps itself. A DOUBLE application
    # raises that back up to 0.00099996 (still >= the single-floor value, so
    # a one-sided ">=" bound cannot tell them apart -- it passes for both).
    # Assert equality against the derived single-floor formula so a
    # regression to double-flooring actually fails this test.
    floor = P.EPSILON / (1 + (P.K - 1) * P.EPSILON)
    assert vec.min() == pytest.approx(floor)


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
    rec = _letter(top)
    vec, status = parse.parse_record(rec, ITEM)
    assert status == "ok"

    # Independently reconstruct the correct single-floor result by applying
    # protocol.floor_renorm exactly once to the raw (pre-floor) vector
    # _parse_letter produces. The old "<= eps * 1.01" threshold is a loose
    # one-sided bound that a double-floored implementation also satisfies
    # (double-flooring pushes already-floored entries back UP toward eps);
    # exact equality against an independently-computed single floor does not.
    choice = rec["response"]["choices"][0]
    raw, raw_status = parse._parse_letter(choice, ITEM)
    assert raw_status == "ok"
    expected = P.floor_renorm(raw)
    assert vec == pytest.approx(expected)

    # Only A and C were returned; the other five sit at exactly the floor.
    at_floor = [v for v in vec if v == pytest.approx(expected.min())]
    assert len(at_floor) == 5


def test_letter_outside_a_to_g_is_a_failure():
    # Fix round 2: classification is read from the top-k TOKEN LIST, never
    # from message.content. Top-k here contains a letter-like token ("Z")
    # that is not in A-G, which section 4's pre-freeze checks treat as a
    # model/prompt problem -- distinct from no letter-like token appearing at
    # all (a provider problem, see test_letter_with_no_valid_letter_in_topk_
    # is_a_counted_failure). Both score uniform; only the diagnosis differs.
    vec, status = parse.parse_record(_letter([("Z", -0.1)]), ITEM)
    assert status == "bad_letter"
    assert status in parse.FAILURE_KINDS
    assert vec == pytest.approx(UNIFORM)


def test_letter_ignores_case_and_surrounding_whitespace():
    vec, status = parse.parse_record(_letter([(" c ", -0.1), ("A", -2.0)]), ITEM)
    assert status == "ok"
    assert int(vec.argmax()) == P.LABELS.index("cs.LG")


def test_letter_with_no_valid_letter_in_topk_is_a_counted_failure():
    """Section 4 ties Q-L to the first token's logprobs. If top-k holds no
    letter-like token at all -- not even one outside A-G -- there is no
    distribution to read and no letter-following signal either; that is a
    provider problem (Parasail not returning the sampled token), scored
    uniform -- NOT an excuse to read a probability off the emitted text, which
    would fabricate confidence the model never expressed.
    """
    rec = {"id": "x", "arm": "Q-L", "status": "ok",
           "response": {"choices": [{
               "finish_reason": "stop",
               "message": {"content": "C"},          # a valid letter in the TEXT
               "logprobs": {"content": [{"top_logprobs": [
                   {"token": " the", "logprob": -0.1},
                   {"token": "\n", "logprob": -2.0},
                   {"token": "123", "logprob": -3.0}]}]}}]}}  # no letter-like token
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


def test_unreadable_topk_distinguishes_wrong_letter_from_no_letter():
    """Both score uniform per section 5, but section 4's pre-freeze checks act
    on WHICH failure it was: a wrong letter means fix the alphabet, no letter at
    all means investigate the provider.
    """
    def ql(topk):
        return {"id": "x", "arm": "Q-L", "status": "ok",
                "response": {"choices": [{
                    "finish_reason": "stop", "message": {"content": "?"},
                    "logprobs": {"content": [{"top_logprobs": [
                        {"token": t, "logprob": lp} for t, lp in topk]}]}}]}}

    vec_wrong, st_wrong = parse.parse_record(ql([("Z", -0.1), ("Y", -2.0)]), ITEM)
    vec_none, st_none = parse.parse_record(ql([(" the", -0.1), ("\n", -2.0)]), ITEM)

    assert st_wrong == "bad_letter"
    assert st_none == "no_logprobs"
    assert st_wrong in parse.FAILURE_KINDS and st_none in parse.FAILURE_KINDS
    # Scoring must be identical -- only the diagnosis differs.
    assert vec_wrong == pytest.approx(UNIFORM)
    assert vec_none == pytest.approx(UNIFORM)


def test_failure_classification_never_reads_the_emitted_text():
    """The emitted text must not influence status or vector. Same unreadable
    top-k, three different emitted strings -> identical outcome.
    """
    results = set()
    for emitted in ("C", "Z", "not a letter at all"):
        rec = {"id": "x", "arm": "Q-L", "status": "ok",
               "response": {"choices": [{
                   "finish_reason": "stop", "message": {"content": emitted},
                   "logprobs": {"content": [{"top_logprobs": [
                       {"token": " the", "logprob": -0.1}]}]}}]}}
        vec, status = parse.parse_record(rec, ITEM)
        results.add(status)
        assert vec == pytest.approx(UNIFORM)
    assert len(results) == 1, f"emitted text changed the outcome: {results}"


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


# --- parse_run: the function that turns a run file into every number the
# study reports (Critical 1). Zero coverage before this point.

def _pool_item(item_id, label="cs.CV"):
    return {"id": item_id, "label": label, "options": list(P.LABELS)}


def _write_pool(tmp_path, pool_name, items):
    d = tmp_path / "pools"
    d.mkdir(exist_ok=True)
    with open(d / f"{pool_name}.jsonl", "w", encoding="utf-8") as fh:
        for it in items:
            fh.write(json.dumps(it) + "\n")


def _write_run(tmp_path, arm, pool_name, records, run_id=1):
    d = tmp_path / "runs"
    d.mkdir(exist_ok=True)
    suffix = "" if run_id == 1 else f"_{run_id}"
    with open(d / f"{arm}_{pool_name}{suffix}.jsonl", "w", encoding="utf-8") as fh:
        for rec in records:
            fh.write(json.dumps(rec) + "\n")


def _ok_record(item_id, payload):
    return {"id": item_id, "arm": "Q-V", "status": "ok",
            "response": {"choices": [{"finish_reason": "stop",
                                      "message": {"content": json.dumps(payload)}}]}}


def test_parse_run_deduplicates_a_repeated_id_and_reports_the_count(
        tmp_path, monkeypatch):
    """Two concurrent writers producing more records than items must not
    double-weight an item; the count dropped must be visible, not silent.
    """
    monkeypatch.chdir(tmp_path)
    payload_a = {l: (1.0 if l == "cs.CV" else 0.0) for l in P.LABELS}
    payload_b = {l: (1.0 if l == "cs.LG" else 0.0) for l in P.LABELS}
    _write_pool(tmp_path, "dev", [_pool_item("a"), _pool_item("b")])
    _write_run(tmp_path, "Q-V", "dev", [
        _ok_record("a", payload_a),
        _ok_record("a", payload_a),   # duplicate: same id, written twice
        _ok_record("b", payload_b),
    ])

    report = parse.parse_run("Q-V", "dev")

    assert report["n"] == 2
    assert report["duplicates_dropped"] == 1
    assert sorted(np.load("preds/Q-V_dev.npz")["ids"]) == ["a", "b"]


def test_parse_run_hard_fails_on_a_missing_item(tmp_path, monkeypatch):
    """An interrupted, never-resumed run must not silently produce a
    complete-looking result with fewer items than the pool.
    """
    monkeypatch.chdir(tmp_path)
    payload_a = {l: (1.0 if l == "cs.CV" else 0.0) for l in P.LABELS}
    _write_pool(tmp_path, "dev", [_pool_item("a"), _pool_item("b")])
    _write_run(tmp_path, "Q-V", "dev", [_ok_record("a", payload_a)])  # "b" absent

    with pytest.raises(SystemExit, match="b"):
        parse.parse_run("Q-V", "dev")


def test_parse_run_hard_fails_on_an_extra_unknown_id(tmp_path, monkeypatch):
    """An id in the run file that is not part of the pool is just as much a
    coverage corruption as a missing one, and must not be silently ignored.
    """
    monkeypatch.chdir(tmp_path)
    payload_a = {l: (1.0 if l == "cs.CV" else 0.0) for l in P.LABELS}
    _write_pool(tmp_path, "dev", [_pool_item("a")])
    _write_run(tmp_path, "Q-V", "dev", [
        _ok_record("a", payload_a),
        _ok_record("unknown-item", payload_a),
    ])

    with pytest.raises(SystemExit, match="unknown-item"):
        parse.parse_run("Q-V", "dev")


def test_parse_run_computes_offsum_count_and_argmax_tie_count(
        tmp_path, monkeypatch):
    """Both diagnostics are un-reconstructable later: offsum_count needs the
    RAW pre-floor sum, and argmax_tie_count needs the exact floored values,
    neither of which survive being folded into an npz of final vectors alone
    unless parse_run computes and reports them at parse time.
    """
    monkeypatch.chdir(tmp_path)
    # "a": raw probabilities sum to 0.5, not 1 -- off-sum, but not a tie once
    # renormalized (a single label carries the mass).
    offsum_payload = {l: (0.5 if l == "cs.LG" else 0.0) for l in P.LABELS}
    # "b": two labels exactly tied at 0.5 each, raw sum is exactly 1 -- a
    # genuine argmax tie, not off-sum.
    tie_payload = {l: (0.5 if l in ("cs.CV", "cs.LG") else 0.0) for l in P.LABELS}
    # "c": control -- neither off-sum nor tied.
    control_payload = {l: (1.0 if l == "cs.AI" else 0.0) for l in P.LABELS}

    _write_pool(tmp_path, "dev",
                [_pool_item("a"), _pool_item("b"), _pool_item("c")])
    _write_run(tmp_path, "Q-V", "dev", [
        _ok_record("a", offsum_payload),
        _ok_record("b", tie_payload),
        _ok_record("c", control_payload),
    ])

    report = parse.parse_run("Q-V", "dev")

    assert report["n"] == 3
    assert report["ok"] == 3
    assert report["offsum_count"] == 1
    assert report["offsum_min_raw_sum"] == pytest.approx(0.5)
    assert report["argmax_tie_count"] == 1


def test_parse_run_with_run_id_2_reads_and_writes_the_2_paths(
        tmp_path, monkeypatch):
    """PC2's re-run is written to runs/{arm}_{pool}_2.jsonl by run.py; parse.py
    must mirror the convention on both the read and write side.
    """
    monkeypatch.chdir(tmp_path)
    payload_a = {l: (1.0 if l == "cs.CV" else 0.0) for l in P.LABELS}
    _write_pool(tmp_path, "evaluation", [_pool_item("a")])
    _write_run(tmp_path, "J", "evaluation", [
        {"id": "a", "arm": "J", "status": "ok",
         "response": {"answers": {P.JEV_QUESTION_KEY: {
             "type": "choice", "choice": "cs.CV",
             "probabilities": {l: (1.0 if l == "cs.CV" else 0.0)
                                for l in P.LABELS},
             "confidence": 0.9}}}},
    ], run_id=2)

    report = parse.parse_run("J", "evaluation", run_id=2)

    assert report["run_id"] == 2
    assert report["n"] == 1
    assert (tmp_path / "preds" / "J_evaluation_2.npz").exists()
    assert (tmp_path / "preds" / "J_evaluation_2_report.json").exists()
    # The unsuffixed run-1 paths must not have been touched.
    assert not (tmp_path / "runs" / "J_evaluation.jsonl").exists()
    assert not (tmp_path / "preds" / "J_evaluation.npz").exists()
