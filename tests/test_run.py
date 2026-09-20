import io
import json
import urllib.error

import pytest

import protocol as P
import run


ITEM = {
    "id": "2609.00002",
    "label": "cs.AI",
    "title": "T",
    "abstract": "A " * 60,
    "options": ["cs.RO", "cs.IR", "cs.LG", "cs.CV", "cs.CR", "cs.AI", "cs.CL"],
}


CHAT_ARMS = [a for a in P.LLM_ARMS if P.ARMS[a]["mechanism"] != "decisions"]


def test_every_chat_arm_pins_its_provider_and_forbids_fallbacks():
    for arm in CHAT_ARMS:
        pay = run.build_payload(arm, ITEM)
        assert pay["provider"]["allow_fallbacks"] is False
        assert pay["provider"]["order"] == [P.ARMS[arm]["provider"]]


def test_jev_payload_uses_the_decisions_schema_not_chat():
    """Arm J speaks TypeSafe's System One contract: no messages array, no
    response_format, no provider block. chat/completions returns HTTP 400 for
    this model, so a chat-shaped payload would fail every item.
    """
    pay = run.build_jev_payload(ITEM)
    assert "messages" not in pay
    assert "response_format" not in pay
    assert "provider" not in pay
    assert pay["model"] == "typesafe/jev-1.13"
    assert ITEM["title"] in pay["state"]
    assert ITEM["abstract"] in pay["state"]
    q = pay["questions"][P.JEV_QUESTION_KEY]
    assert q["type"] == "choice"
    assert set(q["criteria"]) == set(P.LABELS)


def test_jev_criteria_follow_the_items_shuffled_order():
    """Section 4 requires the same per-item permutation for every arm including
    J. Here the shuffle is carried by criteria insertion order, which Python
    preserves through json.dumps.
    """
    pay = run.build_jev_payload(ITEM)
    assert list(pay["questions"][P.JEV_QUESTION_KEY]["criteria"]) == ITEM["options"]


def test_jev_criteria_carry_the_verbatim_descriptions():
    import taxonomy
    desc = taxonomy.load_descriptions()
    crit = run.build_jev_payload(ITEM)["questions"][P.JEV_QUESTION_KEY]["criteria"]
    for label in P.LABELS:
        assert desc[label]["description"] in crit[label]
        assert desc[label]["name"] in crit[label]


def test_frontier_arms_send_no_reasoning_parameter():
    """Section 4: reasoning omitted so each model runs as it ships."""
    for arm in ("F1-V", "F2-V"):
        assert "reasoning" not in run.build_payload(arm, ITEM)


def test_f1v_sends_neither_seed_nor_temperature():
    pay = run.build_payload("F1-V", ITEM)
    assert "seed" not in pay and "temperature" not in pay


def test_f2v_sends_the_master_seed_but_no_temperature():
    pay = run.build_payload("F2-V", ITEM)
    assert pay["seed"] == 20260919
    assert "temperature" not in pay


def test_qv_is_temperature_zero_and_structured():
    pay = run.build_payload("Q-V", ITEM)
    assert pay["temperature"] == 0.0
    assert pay["response_format"]["type"] == "json_schema"


def test_ql_requests_logprobs_and_sets_no_sampling_params():
    pay = run.build_payload("Q-L", ITEM)
    assert pay["logprobs"] is True
    assert pay["top_logprobs"] >= 7
    for banned in ("temperature", "top_p", "top_k"):
        assert banned not in pay, f"section 4 forbids {banned} on Q-L"
    assert "response_format" not in pay


def test_qwen_arms_explicitly_disable_thinking():
    """Section 4: the Qwen arms stay non-thinking, because reasoning tokens
    would displace the first generated token whose logprobs Q-L reads.
    Non-thinking is NOT this endpoint's default, so it must be sent explicitly.
    """
    for arm in ("Q-V", "Q-L"):
        pay = run.build_payload(arm, ITEM)
        assert pay["reasoning"] == {"enabled": False}, f"{arm} must disable thinking"


def test_frontier_arms_still_send_no_reasoning_key():
    """Section 4 requires F1-V and F2-V to run at provider defaults, with the
    parameter omitted so each runs as it ships. A reasoning key here would
    silently change what the study measures.
    """
    for arm in ("F1-V", "F2-V"):
        assert "reasoning" not in run.build_payload(arm, ITEM)


def test_jev_payload_has_no_reasoning_key():
    assert "reasoning" not in run.build_jev_payload(ITEM)


def test_ql_and_qv_share_model_and_provider():
    a, b = run.build_payload("Q-V", ITEM), run.build_payload("Q-L", ITEM)
    assert a["model"] == b["model"]
    assert a["provider"] == b["provider"]


def test_verbalized_schema_follows_the_items_shuffled_order():
    pay = run.build_payload("Q-V", ITEM)
    props = pay["response_format"]["json_schema"]["schema"]["properties"]
    assert list(props) == ITEM["options"]


def test_completed_ids_reads_back_what_was_written(tmp_path):
    path = tmp_path / "r.jsonl"
    path.write_text(
        json.dumps({"id": "a"}) + "\n" + json.dumps({"id": "b"}) + "\n")
    assert run.completed_ids(str(path)) == {"a", "b"}


def test_completed_ids_tolerates_a_truncated_final_line(tmp_path):
    """A crash mid-write must not make the whole run unresumable."""
    path = tmp_path / "r.jsonl"
    path.write_text(json.dumps({"id": "a"}) + "\n" + '{"id": "b"')
    assert run.completed_ids(str(path)) == {"a"}


def test_completed_ids_on_missing_file_is_empty(tmp_path):
    assert run.completed_ids(str(tmp_path / "nope.jsonl")) == set()


def test_completed_ids_skips_a_run_level_failure_so_resume_retries_it(tmp_path):
    """status: "failure" is a run-level infrastructure failure (rate limiting
    exhausted, a network error) now that fatal HTTP statuses abort instead of
    becoming a record -- topping up credit and resuming must retry it, not
    leave it permanently scored uniform. status: "ok" records, even ones
    parse.py will later classify as a parse error or refusal, are legitimate
    section 5 outcomes and must stay done.
    """
    path = tmp_path / "r.jsonl"
    path.write_text(
        json.dumps({"id": "a", "status": "ok"}) + "\n" +
        json.dumps({"id": "b", "status": "failure"}) + "\n" +
        json.dumps({"id": "c", "status": "ok"}) + "\n")
    assert run.completed_ids(str(path)) == {"a", "c"}


def test_payload_never_contains_the_api_key():
    key = P.api_key()
    for arm in CHAT_ARMS:
        assert key not in json.dumps(run.build_payload(arm, ITEM))
    assert key not in json.dumps(run.build_jev_payload(ITEM))


def test_endpoint_snapshot_records_quantization_and_params(tmp_path):
    snap = run.snapshot_endpoint("Q-V", out_dir=str(tmp_path))
    assert snap["provider"] == "Parasail"
    assert "quantization" in snap
    assert "structured_outputs" in (snap["supported_parameters"] or [])


def test_endpoint_snapshot_confirms_ql_logprobs_are_advertised(tmp_path):
    snap = run.snapshot_endpoint("Q-L", out_dir=str(tmp_path))
    assert "logprobs" in (snap["supported_parameters"] or [])


def test_failure_record_has_the_full_documented_shape(monkeypatch):
    """Section 5: failures are recorded, never dropped -- and downstream code
    reads a fixed set of keys, so a failure record must carry them all.
    """
    def boom(payload, url=run.CHAT_URL):
        raise urllib.error.URLError("no network")
    monkeypatch.setattr(run, "_post", boom)
    monkeypatch.setattr(run.time, "sleep", lambda *_: None)

    rec = run.call("Q-L", ITEM)
    assert rec["status"] == "failure"
    for key in ("id", "arm", "request", "response", "status", "provider",
                "latency_s", "timestamp"):
        assert key in rec, f"failure record must carry {key}"
    assert rec["response"] is None


def test_an_unexpected_exception_still_produces_a_failure_record(monkeypatch):
    """The specific handlers cannot be exhaustive. Anything unexpected must
    become a recorded failure rather than escaping and dropping the item.
    """
    def boom(payload, url=run.CHAT_URL):
        raise RuntimeError("something nobody predicted")
    monkeypatch.setattr(run, "_post", boom)
    monkeypatch.setattr(run.time, "sleep", lambda *_: None)

    rec = run.call("Q-L", ITEM)
    assert rec["status"] == "failure"
    assert "RuntimeError" in rec["error"]
    assert "provider" in rec


def test_snapshot_endpoint_hard_stops_when_the_pin_is_not_serving(monkeypatch, tmp_path):
    """A silent reroute would change Q-L's quantization and destroy C3, so a
    missing pin must stop the run rather than fall back.
    """
    payload = {"data": {"endpoints": [
        {"provider_name": "SomeoneElse", "quantization": "fp8",
         "supported_parameters": [], "context_length": 1, "pricing": {}}]}}

    class FakeResponse:
        def read(self): return json.dumps(payload).encode("utf-8")
        def __enter__(self): return self
        def __exit__(self, *a): return False

    monkeypatch.setattr(run.urllib.request, "urlopen", lambda *a, **k: FakeResponse())
    with pytest.raises(SystemExit, match="Parasail|STOP"):
        run.snapshot_endpoint("Q-V", out_dir=str(tmp_path))


def _http_error(code, body):
    return urllib.error.HTTPError(
        "http://x", code, "err", {}, io.BytesIO(body.encode("utf-8")))


@pytest.mark.parametrize("code", [401, 402, 403, 400, 404, 422])
def test_call_treats_non_retryable_4xx_as_fatal_not_data(monkeypatch, code):
    """Out-of-credit (402), bad/forbidden key (401/403) and other non-408/429
    4xx must never become a `status: "failure"` record -- section 5 would
    score that uniform forever, and a resume would never retry it (a fixed
    401/402/403 stays broken data permanently). It must raise instead.
    """
    def boom(payload, url=run.CHAT_URL):
        raise _http_error(code, '{"error": {"message": "nope"}}')
    monkeypatch.setattr(run, "_post", boom)
    monkeypatch.setattr(run.time, "sleep", lambda *_: None)

    with pytest.raises(run.FatalRunError) as exc_info:
        run.call("Q-L", ITEM)
    assert exc_info.value.status == code


@pytest.mark.parametrize("code", [408, 429])
def test_call_still_retries_408_and_429_and_records_http_status(monkeypatch, code):
    """408/429 are transient -- still retried, still recorded as data with the
    status carried so parse.py can classify it as run_error rather than a
    genuine network timeout.
    """
    attempts = {"n": 0}

    def boom(payload, url=run.CHAT_URL):
        attempts["n"] += 1
        raise _http_error(code, '{"error": "rate limited"}')
    monkeypatch.setattr(run, "_post", boom)
    monkeypatch.setattr(run.time, "sleep", lambda *_: None)

    rec = run.call("Q-L", ITEM)
    assert attempts["n"] == run.MAX_ATTEMPTS
    assert rec["status"] == "failure"
    assert rec["http_status"] == code


def test_call_redacts_user_id_from_a_fatal_error_body(monkeypatch):
    """Section 4 publishes the raw JSONL; OpenRouter error bodies can carry a
    `user_id`. It must never survive into an exception message that could
    still end up logged or recorded.
    """
    def boom(payload, url=run.CHAT_URL):
        raise _http_error(
            402, '{"error": {"message": "insufficient credit"}, '
                 '"user_id": "user_abc123"}')
    monkeypatch.setattr(run, "_post", boom)
    monkeypatch.setattr(run.time, "sleep", lambda *_: None)

    with pytest.raises(run.FatalRunError) as exc_info:
        run.call("Q-L", ITEM)
    assert "user_abc123" not in str(exc_info.value)
    assert "user_abc123" not in exc_info.value.body


def test_call_redacts_user_id_even_when_the_error_body_is_not_json(monkeypatch):
    """The regex fallback must still catch it when the body cannot be parsed
    as JSON (e.g. an HTML error page or a truncated body).
    """
    def boom(payload, url=run.CHAT_URL):
        raise _http_error(
            403, 'not json but leaks "user_id": "user_xyz789" anyway')
    monkeypatch.setattr(run, "_post", boom)
    monkeypatch.setattr(run.time, "sleep", lambda *_: None)

    with pytest.raises(run.FatalRunError) as exc_info:
        run.call("Q-L", ITEM)
    assert "user_xyz789" not in str(exc_info.value)
    assert "user_xyz789" not in exc_info.value.body


def test_main_aborts_loudly_and_writes_no_record_on_a_fatal_error(
        monkeypatch, tmp_path):
    """End to end: a fatal HTTP error must abort main() with a clear message
    naming the status, and the item in flight must not be written to the run
    file -- not swallowed by handle()'s last-resort BaseException guard into
    a `status: "failure"` record that a resume would never retry.
    """
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(run, "snapshot_endpoint", lambda arm: None)
    monkeypatch.setattr(run.pools, "load_pool",
                         lambda pool: [dict(ITEM, id="only-item")])
    monkeypatch.setattr(
        run, "call",
        lambda arm, item: (_ for _ in ()).throw(
            run.FatalRunError(402, "insufficient credit")))
    monkeypatch.setattr(
        "sys.argv",
        ["run.py", "--arm", "Q-L", "--pool", "dev", "--sequential"])

    with pytest.raises(SystemExit, match="402"):
        run.main()

    out_path = tmp_path / "runs" / "Q-L_dev.jsonl"
    assert out_path.exists()
    assert out_path.read_text() == "", \
        "no record may be written for the item that hit the fatal error"
