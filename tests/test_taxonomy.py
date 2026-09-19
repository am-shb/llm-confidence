import protocol as P
import taxonomy


def test_all_seven_labels_have_a_name_and_description():
    d = taxonomy.load_descriptions()
    assert set(d) == set(P.LABELS)
    for label, entry in d.items():
        assert entry["name"].strip()
        assert entry["description"].strip()


def test_descriptions_are_verbatim_single_line_no_markup():
    d = taxonomy.load_descriptions()
    for label, entry in d.items():
        assert "<" not in entry["description"]
        assert "\n" not in entry["description"]
        assert "  " not in entry["description"]


def test_known_values_are_exact():
    d = taxonomy.load_descriptions()
    assert d["cs.LG"]["name"] == "Machine Learning"
    assert d["cs.RO"]["name"] == "Robotics"
    # cs.RO's official description really is this short; do not pad it.
    assert d["cs.RO"]["description"] == (
        "Roughly includes material in ACM Subject Class I.2.9.")
    assert d["cs.CV"]["description"].startswith(
        "Covers image processing, computer vision, pattern recognition")


def test_no_base_rate_information_leaked_in():
    """Section 2: no arm gets base-rate information."""
    d = taxonomy.load_descriptions()
    blob = " ".join(e["description"] for e in d.values()).lower()
    for word in ["base rate", "frequency", "prior", "%", "most common"]:
        assert word not in blob
