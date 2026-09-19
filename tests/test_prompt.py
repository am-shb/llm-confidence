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


def test_all_mechanisms_share_a_byte_identical_body():
    """Section 4 requires the SAME prompt content for every LLM arm, with only
    the final output instruction differing. Assert that exactly: everything up
    to the divergence point must be byte-identical, and the whole of the
    item-specific content must live inside that shared region. A marker
    substring check would not catch _body() being duplicated and drifting.
    """
    import functools

    bodies = {}
    for mech in ("verbalized", "letter", "decisions"):
        msgs = prompt.build_prompt(ITEM, mech)
        bodies[mech] = next(m["content"] for m in msgs if m["role"] == "user")

    def common_prefix(a, b):
        limit = min(len(a), len(b))
        i = 0
        while i < limit and a[i] == b[i]:
            i += 1
        return a[:i]

    shared = functools.reduce(common_prefix, bodies.values())

    # The entire task framing and every item-specific field must be shared.
    assert prompt.SHARED_BODY_MARKER in shared
    assert ITEM["title"] in shared
    assert ITEM["abstract"].strip()[:60] in shared
    for label in P.LABELS:
        assert label in shared, f"{label} must be in the shared body"
    for letter in prompt.LETTERS:
        assert f"{letter}." in shared, "option lettering must be shared"

    # What differs must be ONLY the trailing output instruction, and each
    # mechanism's remainder must still carry the universal XML-tag rule.
    for mech, body in bodies.items():
        tail = body[len(shared):]
        assert tail, f"{mech} must have a distinct output instruction"
        assert "XML" in tail or "XML" in shared
        assert ITEM["abstract"].strip()[:60] not in tail, \
            f"{mech}: item content leaked into the divergent tail"


def test_unknown_mechanism_raises():
    with pytest.raises(ValueError):
        prompt.build_prompt(ITEM, "telepathy")
