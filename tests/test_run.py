import json
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
