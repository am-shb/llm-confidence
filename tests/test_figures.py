"""Tests for figures.py -- the chart generator behind the post.

figures.py promises two things in its docstring, and these tests hold it to
both. First, that every number is recomputed from `preds/*.npz` and
`runs/*.jsonl` so "a figure can never disagree with results/analysis.md" --
so its local `brier`, `murphy` and `coverage` helpers must agree with
analyze.py's canonical versions on the real evaluation data. Second, that
there is "one entity->colour map for every chart" -- so the map must cover
every arm the protocol defines and keep the models visually distinct.

Then the plain question: does each figure actually render?

Importing figures loads the evaluation pool and all eight prediction files
at module scope, so these run against the committed data and need the repo
root as the working directory (pytest.ini sets `pythonpath = .`).
"""

import numpy as np
import pytest

import analyze
import figures
import protocol as P


ARMS = analyze.ALL_ARMS
TIER_ARMS = ("J", "F1-V", "F2-V")


# --------------------------------------------------------------------------
# The entity -> colour map
# --------------------------------------------------------------------------

def test_every_protocol_arm_has_a_colour():
    assert set(figures.C) == set(ARMS)


def test_every_protocol_arm_has_a_display_name():
    assert set(figures.NAME) == set(ARMS)


def test_distinct_models_get_distinct_colours():
    """P and P* are both "constant baseline" and deliberately share MUTED
    -- they are the same entity under two priors. Every other arm is a
    different system and must be told apart by colour alone."""
    distinct = [a for a in ARMS if a != "P-star"]
    colours = [figures.C[a] for a in distinct]
    assert len(set(colours)) == len(distinct)


def test_the_two_constant_baselines_share_one_colour():
    assert figures.C["P"] == figures.C["P-star"]
    assert figures.NAME["P"] == figures.NAME["P-star"]


def test_tier_matched_arms_are_the_three_the_protocol_compares():
    """PC1 (PROTOCOL.md section 8) rests on J, F1-V and F2-V being within
    8 accuracy points of each other. TIER drives line weight and z-order in
    every figure, so it must name exactly those three."""
    assert figures.TIER == TIER_ARMS


def test_colours_are_six_digit_hex():
    for arm, colour in figures.C.items():
        assert colour.startswith("#") and len(colour) == 7, arm
        int(colour[1:], 16)


# --------------------------------------------------------------------------
# The numbers must match analyze.py
# --------------------------------------------------------------------------

@pytest.mark.parametrize("arm", ARMS)
def test_brier_matches_analyze(arm):
    expected = analyze.multiclass_brier(figures.PM[arm], figures.y)
    assert figures.brier(arm) == pytest.approx(expected, abs=1e-12)


@pytest.mark.parametrize("arm", ARMS)
def test_murphy_matches_analyze(arm):
    """figures.murphy re-derives the reliability/resolution split that
    analyze.murphy_decomposition computes for results/analysis.md. Same
    bin count, same data, so the same two numbers."""
    p = figures.PM[arm]
    conf, corr = p.max(1), (p.argmax(1) == figures.y).astype(float)
    expected = analyze.murphy_decomposition(conf, corr, n_bins=15)
    rel, res = figures.murphy(arm, nbins=15)
    assert rel == pytest.approx(expected["reliability"], abs=1e-12)
    assert res == pytest.approx(expected["resolution"], abs=1e-12)


@pytest.mark.parametrize("arm", TIER_ARMS + ("A",))
def test_coverage_matches_analyze_cross_fit(arm):
    """fig4 quotes coverage at a 5% error budget. The same threshold fit,
    run through analyze.cross_fit_arm, must accept the same share."""
    probs = figures.PM[arm]
    folds = analyze.make_folds(figures.y, 5)
    fit = analyze.cross_fit_arm(probs, figures.y, folds)
    cov, err = figures.coverage(arm)
    assert cov == pytest.approx(fit["accept_raw"].mean(), abs=1e-12)

    accepted = fit["accept_raw"]
    expected_err = 1 - fit["correct"][accepted].mean()
    assert err == pytest.approx(expected_err, abs=1e-12)


# --------------------------------------------------------------------------
# The helpers themselves
# --------------------------------------------------------------------------

def test_predictions_are_aligned_to_the_evaluation_pool():
    """preds/*.npz store items in their own order; figures.preds reindexes
    to the pool's order. If that slipped, every arm's Brier would be scored
    against the wrong labels."""
    assert len(figures.ids) == P.POOL_SIZES["evaluation"]
    for arm in ARMS:
        assert figures.PM[arm].shape == (len(figures.ids), P.K)


@pytest.mark.parametrize("arm", ARMS)
def test_prediction_rows_are_distributions(arm):
    row_sums = figures.PM[arm].sum(axis=1)
    assert row_sums == pytest.approx(np.ones(len(figures.ids)), abs=1e-9)
    assert (figures.PM[arm] >= 0).all()


@pytest.mark.parametrize("nbins", [8, 15])
def test_curve_returns_one_point_per_bin(nbins):
    xs, ys = figures.curve("J", nbins)
    assert len(xs) == len(ys) == nbins
    assert ((0 <= ys) & (ys <= 1)).all()


def test_raw_vectors_are_pre_floor():
    """fig2 is about Jev's exact zeros, which only exist before
    P.floor_renorm runs. If rawvecs picked up floored data the whole
    figure would be vacuous."""
    allv = np.concatenate([figures.RAW["J"][i] for i in figures.ids])
    assert (allv == 0).any()
    assert figures.PM["J"].min() >= P.EPSILON * 0.99


def test_raw_vectors_cover_every_evaluation_item():
    for arm in ("J", "F1-V", "F2-V"):
        assert set(figures.RAW[arm]) >= set(figures.ids)


# --------------------------------------------------------------------------
# Rendering
# --------------------------------------------------------------------------

FIGURES = [
    ("fig0_hero.png", "fig_hero"),
    ("fig1_scatter.png", "fig_scatter"),
    ("fig2_grid.png", "fig_grid"),
    ("fig4_coverage.png", "fig_coverage"),
    ("fig5_deficit.png", "fig_deficit"),
    ("fig6_mechanism.png", "fig_mechanism"),
]


@pytest.mark.parametrize("filename,func", FIGURES)
def test_figure_renders_a_png(filename, func, tmp_path, monkeypatch):
    monkeypatch.setattr(figures, "OUT", str(tmp_path))
    getattr(figures, func)()
    out = tmp_path / filename
    assert out.exists(), f"{func} did not write {filename}"
    assert out.stat().st_size > 10_000
    assert out.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"


def test_the_rendered_set_is_exactly_what_the_post_uses(tmp_path, monkeypatch):
    """The post embeds six figures. Running the module's entry point must
    produce those six and no strays."""
    monkeypatch.setattr(figures, "OUT", str(tmp_path))
    for _, func in FIGURES:
        getattr(figures, func)()
    written = sorted(p.name for p in tmp_path.glob("*.png"))
    assert written == sorted(name for name, _ in FIGURES)
