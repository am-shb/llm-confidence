import pytest

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


def test_parse_page_omits_a_category_with_no_description_of_its_own():
    """A heading with no <p> must be omitted, never paired with the NEXT
    category's text. Silent mis-attribution would corrupt verbatim study data.
    """
    page = ("<div><h4>cs.CV <span>(Computer Vision)</span></h4></div>"
            "<div><h4>cs.LG <span>(Machine Learning)</span></h4></div>"
            "<div><p>BELONGS TO cs.LG.</p></div>")
    got = taxonomy.parse_page(page)
    assert "cs.CV" not in got
    assert got["cs.LG"]["description"] == "BELONGS TO cs.LG."


def test_parse_page_keeps_each_description_with_its_own_category():
    page = ("<div><h4>cs.CV <span>(Computer Vision)</span></h4></div>"
            "<div><p>CV text.</p></div>"
            "<div><h4>cs.LG <span>(Machine Learning)</span></h4></div>"
            "<div><p>LG text.</p></div>")
    got = taxonomy.parse_page(page)
    assert got["cs.CV"]["description"] == "CV text."
    assert got["cs.LG"]["description"] == "LG text."
    assert got["cs.CV"]["name"] == "Computer Vision"


def test_fetch_descriptions_hard_stops_when_a_study_label_is_missing(monkeypatch):
    """The guard is the only thing standing between a partial parse and
    corrupted prompts, so pin that it raises rather than returning partial data.
    """
    page = ("<div><h4>cs.CV <span>(Computer Vision)</span></h4></div>"
            "<div><p>CV text.</p></div>")

    class FakeResponse:
        def read(self): return page.encode("utf-8")
        def __enter__(self): return self
        def __exit__(self, *a): return False

    monkeypatch.setattr(taxonomy.urllib.request, "urlopen",
                        lambda *a, **k: FakeResponse())
    with pytest.raises(SystemExit, match="cs.LG|STOP"):
        taxonomy.fetch_descriptions()
