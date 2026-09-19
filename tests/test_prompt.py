import json
import pytest
import protocol as P
import prompt


ITEM = {
    "id": "2609.00001",
    "label": "cs.LG",
    "title": "A Study of Things",
    "abstract": "We study things. " * 20,
    "options": ["cs.RO", "cs.IR", "cs.LG", "cs.CV", "cs.CR", "cs.AI", "cs.CL"],
}


def test_every_mechanism_includes_title_abstract_and_all_seven_options():
    for mech in ("verbalized", "letter", "decisions"):
        text = json.dumps(prompt.build_prompt(ITEM, mech))
        assert ITEM["title"] in text
        assert "We study things." in text
        for label in P.LABELS:
            assert label in text


def test_options_appear_in_the_items_shuffled_order():
    text = json.dumps(prompt.build_prompt(ITEM, "verbalized"))
    positions = [text.index(o) for o in ITEM["options"]]
    assert positions == sorted(positions), "options must follow the shuffle"


def test_descriptions_are_included_verbatim():
    import taxonomy
    desc = taxonomy.load_descriptions()
    text = json.dumps(prompt.build_prompt(ITEM, "verbalized"))
    assert desc["cs.RO"]["description"] in text
    assert desc["cs.RO"]["name"] in text


def test_no_base_rate_or_deadline_hints_anywhere():
    for mech in ("verbalized", "letter", "decisions"):
        text = json.dumps(prompt.build_prompt(ITEM, mech)).lower()
        for banned in ["base rate", "most common", "deadline", "prior probability",
                       "frequency", "usually"]:
            assert banned not in text


def test_all_arms_get_the_xml_tag_instruction():
    for mech in ("verbalized", "letter", "decisions"):
        text = json.dumps(prompt.build_prompt(ITEM, mech)).lower()
        assert "xml" in text


def test_no_anti_reasoning_instruction_anywhere():
    """Section 4: nothing tells a model not to think."""
    for mech in ("verbalized", "letter", "decisions"):
        text = json.dumps(prompt.build_prompt(ITEM, mech)).lower()
        for banned in ["do not think", "don't think", "no reasoning",
                       "without reasoning", "do not reason", "skip reasoning"]:
            assert banned not in text


def test_letter_mechanism_asks_for_one_letter_and_maps_a_to_g():
    text = json.dumps(prompt.build_prompt(ITEM, "letter"))
    assert "A" in text and "G" in text
    assert prompt.LETTERS == "ABCDEFG"
    assert len(prompt.LETTERS) == P.K


def test_json_schema_requires_all_seven_keys_in_shuffled_order():
    schema = prompt.json_schema(ITEM["options"])
    props = schema["schema"]["properties"]
    assert list(props) == ITEM["options"]
    assert set(schema["schema"]["required"]) == set(ITEM["options"])
    assert schema["schema"]["additionalProperties"] is False
    assert schema["strict"] is True


def test_prompt_body_is_identical_across_mechanisms_except_the_output_rule():
    bodies = {}
    for mech in ("verbalized", "letter", "decisions"):
        msgs = prompt.build_prompt(ITEM, mech)
        bodies[mech] = next(m["content"] for m in msgs if m["role"] == "user")
    shared = prompt.SHARED_BODY_MARKER
    assert all(shared in b for b in bodies.values())


def test_unknown_mechanism_raises():
    with pytest.raises(ValueError):
        prompt.build_prompt(ITEM, "telepathy")
