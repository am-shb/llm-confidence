#!/usr/bin/env python3
"""Figures for the blog post.

Every number is recomputed here from `preds/*.npz` and `runs/*.jsonl` --
nothing is copied out of a markdown file. Cross-fitted quantities reuse
`analyze.make_folds` / `analyze.fit_tau` so a figure can never disagree
with results/analysis.md.

One palette, one style, one entity->colour map for every chart: a model is
the same colour wherever it appears. Titles are labels, not sentences.
"""
import json

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter

import protocol as P
import analyze

L, K = P.LABELS, len(P.LABELS)
OUT = "results"

# ---------------------------------------------------------------- style ----
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK2 = "#52514e"
MUTED = "#8a8a84"
GRID = "#e3e3dd"

# Categorical slots 1-6 of the validated palette, assigned to entities in
# fixed order and never cycled. Light mode: worst adjacent CVD dE 9.1,
# normal-vision 19.6. Aqua, yellow and magenta fall under 3:1 on this
# surface, so every series carries a direct label or a legend entry.
C = {
    "J":      "#2a78d6",   # Jev
    "F1-V":   "#eb6834",   # Claude Sonnet 5
    "F2-V":   "#1baf7a",   # GPT-5.6 Terra
    "Q-V":    "#eda100",   # Qwen, written
    "Q-L":    "#e87ba4",   # Qwen, logprobs
    "A":      "#008300",   # TF-IDF classifier
    "P":      MUTED,       # constant baseline
    "P-star": MUTED,
}
NAME = {
    "J": "Jev", "F1-V": "Claude Sonnet 5", "F2-V": "GPT-5.6 Terra",
    "Q-V": "Qwen, written", "Q-L": "Qwen, logprobs",
    "A": "TF-IDF classifier", "P": "constant baseline",
    "P-star": "constant baseline",
}
TIER = ("J", "F1-V", "F2-V")          # the three tier-matched arms

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE,
    "font.family": "DejaVu Sans", "font.size": 10,
    "text.color": INK, "axes.labelcolor": INK2, "axes.edgecolor": GRID,
    "xtick.color": INK2, "ytick.color": INK2,
    "xtick.labelsize": 9.5, "ytick.labelsize": 9.5,
    "axes.labelsize": 10,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
    "axes.spines.top": False, "axes.spines.right": False,
    "lines.linewidth": 2.0, "lines.solid_capstyle": "round",
})


def header(ax, title, subtitle, pad=1.10, sub=1.035):
    """Short label, then one line of provenance. Never a sentence."""
    ax.text(0, pad, title, transform=ax.transAxes, fontsize=13.5,
            fontweight="bold", color=INK, va="bottom", ha="left")
    ax.text(0, sub, subtitle, transform=ax.transAxes, fontsize=9.5,
            color=INK2, va="bottom", ha="left")


def tidy(ax, xgrid=False):
    ax.set_axisbelow(True)
    ax.grid(axis="y", alpha=1.0)
    ax.grid(axis="x", alpha=1.0 if xgrid else 0.0)
    for s in ("left", "bottom"):
        ax.spines[s].set_linewidth(0.9)
        ax.spines[s].set_color(GRID)


def bare(ax):
    """Axes for a bar chart: category labels do the work, no x scale."""
    ax.set_xticks([])
    ax.grid(False)
    ax.spines["bottom"].set_visible(False)
    ax.spines["left"].set_color(GRID)
    ax.set_axisbelow(True)


# ----------------------------------------------------------------- data ----
items = [json.loads(l) for l in open("pools/evaluation.jsonl")]
ids = [it["id"] for it in items]
y = np.array([L.index(it["label"]) for it in items])
ytrue = dict(zip(ids, y))
ONEHOT = np.zeros((len(y), K)); ONEHOT[np.arange(len(y)), y] = 1


def preds(arm):
    d = np.load(f"preds/{arm}_evaluation.npz", allow_pickle=True)
    pos = {str(i): k for k, i in enumerate(d["ids"])}
    return d["probs"][[pos[i] for i in ids]]


def rawvecs(arm):
    """Probabilities exactly as emitted, before the epsilon floor."""
    out = {}
    for line in open(f"runs/{arm}_evaluation.jsonl"):
        r = json.loads(line)
        if r.get("status") != "ok":
            continue
        if arm == "J":
            pr = r["response"]["answers"]["primary_category"]["probabilities"]
        else:
            pr = json.loads(r["response"]["choices"][0]["message"]["content"])
        out[r["id"]] = np.array([float(pr[l]) for l in L])
    return out


PM = {a: preds(a) for a in ("J", "F1-V", "F2-V", "Q-V", "Q-L", "A", "P", "P-star")}
RAW = {a: rawvecs(a) for a in ("J", "F1-V", "F2-V")}


def brier(arm):
    return ((PM[arm] - ONEHOT) ** 2).sum(1).mean()


def topline(arm):
    p = PM[arm]
    return p.max(1), (p.argmax(1) == y).astype(float)


def murphy(arm, nbins=15):
    """Delegated to analyze, not reimplemented. Equal-mass binning has to
    sort by confidence, and Jev's probabilities sit on a two-decimal grid,
    so ties are everywhere -- an unstable sort here would put items in
    different bins than analysis.md used and quietly shift reliability."""
    d = analyze.murphy_decomposition(*topline(arm), n_bins=nbins)
    return d["reliability"], d["resolution"]


def curve(arm, nbins=8):
    _, bins = analyze.ece_equal_mass(*topline(arm), n_bins=nbins)
    return (np.array([c for c, _, _ in bins]),
            np.array([a for _, a, _ in bins]))


def coverage(arm, budget=0.05, kmin=50, nfolds=5):
    """PROTOCOL.md section 6, using the pipeline's own fold split and fit."""
    conf, corr = topline(arm)
    folds = analyze.make_folds(y, nfolds)
    accept = np.zeros(len(y), dtype=bool)
    for f in range(nfolds):
        tau = analyze.fit_tau(conf[folds != f], corr[folds != f],
                              target_error=budget, min_k=kmin)
        accept[folds == f] = conf[folds == f] >= tau
    err = 1 - corr[accept].mean() if accept.any() else float("nan")
    return accept.mean(), err


# ------------------------------------------------------- 0. claimed vs actual
def fig_hero(nbins=8):
    """Eight bins, not fifteen: at 2,000 items the finer split is mostly
    sampling noise, and the noise is what makes the curves cross."""
    order = ["A", "Q-L", "Q-V", "F2-V", "F1-V", "J"]
    fig, ax = plt.subplots(figsize=(9.6, 8.8))
    lo, hi = 0.20, 1.005

    ax.fill_between([lo, hi], [lo, hi], [hi, hi], color=C["F2-V"], alpha=0.045,
                    linewidth=0, zorder=0)
    ax.fill_between([lo, hi], [lo, lo], [lo, hi], color=C["F1-V"], alpha=0.045,
                    linewidth=0, zorder=0)
    ax.plot([lo, hi], [lo, hi], color="#9a9a93", lw=1.4, ls=(0, (5, 4)), zorder=3)
    ax.text(0.307, 0.320, "perfectly calibrated", color=MUTED, fontsize=9.5,
            rotation=42.5, ha="center", va="center")
    ax.text(0.227, 0.972, "ABOVE — hedging", color="#1a8a62",
            fontsize=9.5, fontweight="bold")
    ax.text(0.725, 0.256, "BELOW — overclaiming", color="#c04a1c",
            fontsize=9.5, fontweight="bold", ha="right")

    for a in order:
        xs, ys = curve(a, nbins)
        tier = a in TIER
        ax.plot(xs, ys, "-o", color=C[a], label=NAME[a],
                lw=3.0 if tier else 1.8, ms=7.5 if tier else 5.5,
                alpha=1.0 if tier else 0.85,
                markeredgecolor=SURFACE, markeredgewidth=1.6,
                zorder=6 if tier else 5)

    p = PM["P"]
    ax.plot([p.max(1).mean()], [(p.argmax(1) == y).mean()], "o", ms=13,
            color=C["P"], zorder=7, markeredgecolor=SURFACE, markeredgewidth=2,
            label=NAME["P"])

    leg = ax.legend(loc="lower right", frameon=True, fontsize=10.5,
                    labelcolor=INK, handlelength=1.5, borderpad=0.9,
                    labelspacing=0.62, handletextpad=0.8,
                    bbox_to_anchor=(0.995, 0.028))
    leg.get_frame().set_facecolor(SURFACE)
    leg.get_frame().set_edgecolor(GRID)
    leg.get_frame().set_linewidth(1.0)
    leg.set_zorder(10)

    ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
    ax.set_aspect("equal", adjustable="box")
    ticks = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    ax.set_xticks(ticks); ax.set_yticks(ticks)
    ax.xaxis.set_major_formatter(PercentFormatter(1.0))
    ax.yaxis.set_major_formatter(PercentFormatter(1.0))
    ax.set_xlabel("confidence stated")
    ax.set_ylabel("how often it was right")
    tidy(ax, xgrid=True)
    header(ax, "Claimed vs. actual",
           "2,000 papers · 8 equal-mass confidence bins",
           pad=1.055, sub=1.018)
    fig.tight_layout()
    fig.savefig(f"{OUT}/fig0_hero.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


# ------------------------------------------------ 1. the two axes, plotted
def fig_scatter():
    order = ["J", "F1-V", "F2-V", "Q-V", "Q-L", "A", "P", "P-star"]
    M = {a: murphy(a) for a in order}
    fig, ax = plt.subplots(figsize=(9.2, 5.9))
    off = {"J": (11, 6, "left"), "F1-V": (0, 13, "center"),
           "F2-V": (-11, -10, "right"), "Q-V": (11, 4, "left"),
           "Q-L": (-12, 5, "right"), "A": (-12, 4, "right"),
           "P": (13, -5, "left"), "P-star": None}
    for a in order:
        rel, res = M[a]
        ax.plot(res, rel, "o", ms=11, color=C[a], zorder=5,
                markeredgecolor=SURFACE, markeredgewidth=2)
        if off[a] is None:
            continue
        dx, dy, ha = off[a]
        ax.annotate(f"{NAME[a]}\nBrier {brier(a):.3f}", (res, rel),
                    textcoords="offset points", xytext=(dx, dy), ha=ha,
                    fontsize=9.5, color=INK, linespacing=1.35)
    ax.set_xlim(-0.002, 0.056); ax.set_ylim(-0.003, 0.050)
    ax.set_xlabel("Resolution  →  telling its hard papers from its easy ones")
    ax.set_ylabel("Reliability  →  distance from\nwhat it claimed")
    ax.annotate("", xy=(0.0522, 0.0016), xytext=(0.0452, 0.0062),
                arrowprops=dict(arrowstyle="-|>", color=MUTED, lw=1.4,
                                shrinkA=0, shrinkB=0))
    ax.text(0.0448, 0.0076, "better", color=MUTED, fontsize=9.5, style="italic")
    tidy(ax, xgrid=True)
    header(ax, "Honesty vs. discrimination",
           "Murphy decomposition of top-label Brier · all eight arms")
    fig.tight_layout()
    fig.savefig(f"{OUT}/fig1_scatter.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


# -------------------------------------------- 2. what the truth was given
def fig_grid():
    """The section's claim is about frequency, not granularity: how often
    each model sends the TRUE category to the bottom of its scale. Plotted
    on all 14,000 slots this looks like a story about grid resolution,
    which the counterfactual says is not what costs Jev anything."""
    fig, ax = plt.subplots(figsize=(9.0, 4.4))
    lo, cut = 4e-4, 0.01
    ax.axvspan(lo, cut, color=INK, alpha=0.045, linewidth=0, zorder=0)

    counts = {}
    for a in TIER:
        v = np.array([RAW[a][i] for i in ids])[np.arange(len(y)), y]
        counts[a] = int((v <= cut).sum())
        zero = float((v == 0).mean())
        nz = np.sort(v[v > 0])
        xs = np.concatenate([[lo], nz, [1.0]])
        ys = np.concatenate([[zero], zero + np.arange(1, len(nz) + 1) / len(v), [1.0]])
        ax.step(xs, ys, where="post", color=C[a], zorder=3,
                linewidth=2.6 if a == "J" else 1.8)
        if zero > 0:
            ax.plot([lo], [zero], "o", ms=8, color=C[a], zorder=4,
                    markeredgecolor=SURFACE, markeredgewidth=2)

    ax.axvline(cut, color=INK2, lw=1.0, ls=(0, (2, 3)), alpha=0.7, zorder=2)
    ax.set_xscale("log")
    ax.set_xlim(lo, 1.0); ax.set_ylim(0, 1.02)
    ax.yaxis.set_major_formatter(PercentFormatter(1.0))
    ax.set_xlabel("probability the model gave the category the authors chose   (log scale)")
    ax.set_ylabel("share of the 2,000 papers\nat or below")

    lab = [("J",    f"Jev — {counts['J']} papers, 85 of them exactly zero", 0.335),
           ("F1-V", f"Claude Sonnet 5 — {counts['F1-V']}", 0.250),
           ("F2-V", f"GPT-5.6 Terra — {counts['F2-V']}", 0.180)]
    ax.text(5.5e-4, 0.425, "gave the right answer 0.01 or less:",
            color=INK2, fontsize=9.5, va="center")
    for a, text, ypos in lab:
        ax.text(5.5e-4, ypos, text, color=C[a], fontsize=10,
                fontweight="bold" if a == "J" else "normal", va="center")
    ax.text(cut * 1.3, 0.95, "0.01 — the smallest number\nJev can say that isn't zero",
            color=INK2, fontsize=9, va="top")
    tidy(ax)
    header(ax, "What the right answer was given",
           "Each model's probability on the authors' own category · 2,000 papers")
    fig.tight_layout()
    fig.savefig(f"{OUT}/fig2_grid.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    return counts


# ------------------------------------------------------- 4. coverage bars
def fig_coverage():
    order = ["J", "F1-V", "F2-V", "A"]
    res = {a: coverage(a) for a in order}
    fig, ax = plt.subplots(figsize=(8.0, 3.2))
    ypos = np.arange(len(order))[::-1]
    for yp, a in zip(ypos, order):
        cov, err = res[a]
        ax.barh(yp, cov, height=0.56, color=C[a], zorder=3)
        ax.text(cov + 0.008, yp, f"{cov*100:.1f}%", va="center", ha="left",
                fontsize=11, color=INK, fontweight="bold")
        ax.text(cov + 0.055, yp, f"realized error {err*100:.1f}%",
                va="center", ha="left", fontsize=9, color=MUTED)
    ax.set_yticks(ypos, [NAME[a] for a in order], fontsize=10, color=INK)
    ax.set_xlim(0, 0.46)
    ax.set_xlabel("share of the 2,000 papers auto-accepted")
    bare(ax)
    header(ax, "Coverage at a 5% error budget",
           "Threshold fitted on four folds, applied to the fifth",
           pad=1.22, sub=1.08)
    fig.tight_layout()
    fig.savefig(f"{OUT}/fig4_coverage.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    return res


# -------------------------------------------------- 5. where the gap lives
def fig_deficit():
    S = [i for i in ids if RAW["J"][i][ytrue[i]] <= 0.01]
    rest = [i for i in ids if i not in set(S)]

    def bsub(arm, sub):
        t = 0.0
        for i in sub:
            v = P.floor_renorm(RAW[arm][i].copy())
            oh = np.zeros(K); oh[ytrue[i]] = 1
            t += ((v - oh) ** 2).sum()
        return t / len(sub)

    fig, ax = plt.subplots(figsize=(7.8, 3.9))
    groups = [(f"the {len(S)} Jev ruled out\n(7% of the set)", S),
              (f"the other {len(rest):,}\n(93%)", rest)]
    w = 0.17
    for gi, (_, sub) in enumerate(groups):
        for ai, a in enumerate(TIER):
            v = bsub(a, sub)
            x = gi * 0.75 + (ai - 1) * w
            ax.bar(x, v, width=w * 0.88, color=C[a], zorder=3,
                   label=NAME[a] if gi == 0 else None)
            ax.text(x, v + 0.045, f"{v:.2f}", ha="center", color=INK,
                    fontsize=9.5, fontweight="bold")
    ax.set_xticks([0, 0.75], [g[0] for g in groups], fontsize=10, color=INK,
                  linespacing=1.5)
    ax.set_xlim(-0.34, 1.09); ax.set_ylim(0, 2.15)
    ax.set_ylabel("Brier score   (lower is better)")
    ax.legend(frameon=False, loc="upper right", fontsize=9.5, labelcolor=INK,
              handlelength=1.1, borderpad=0.2, bbox_to_anchor=(1.0, 1.02))
    tidy(ax)
    header(ax, "Where the gap lives",
           "Papers Jev scored at 0.01 or below, split from the rest",
           pad=1.13, sub=1.04)
    fig.tight_layout()
    fig.savefig(f"{OUT}/fig5_deficit.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    return len(S)


# ------------------------------------------------- 6. mechanism contrast
def fig_mechanism():
    LETTERS = "ABCDEFG"
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 3.6),
                             gridspec_kw={"width_ratios": [1, 1.5]})

    ax = axes[0]
    vals = {a: brier(a) for a in ("Q-V", "Q-L")}
    for yp, a in zip([1, 0], ("Q-V", "Q-L")):
        ax.barh(yp, vals[a], height=0.5, color=C[a], zorder=3)
        ax.text(vals[a] - 0.02, yp, f"{vals[a]:.3f}", va="center", ha="right",
                fontsize=11, color="#ffffff", fontweight="bold")
    ax.set_yticks([1, 0], ["written out", "from logprobs"], fontsize=10, color=INK)
    ax.set_xlim(0, 0.66)
    ax.set_xlabel("Brier score   (lower is better)")
    bare(ax)
    ax.text(0, 1.10, "Same model, same prompt", transform=ax.transAxes,
            fontsize=10.5, color=INK)

    ax = axes[1]
    truth = np.zeros(7)
    for it in items:
        truth[it["options"].index(it["label"])] += 1
    pick = np.zeros(7)
    for line in open("runs/Q-L_evaluation.jsonl"):
        r = json.loads(line)
        if r.get("status") != "ok":
            continue
        lp = r["response"]["choices"][0]["logprobs"]["content"][0]
        cand = {t["token"]: t["logprob"] for t in lp["top_logprobs"]
                if t["token"] in LETTERS}
        pick[LETTERS.index(max(cand, key=cand.get))] += 1
    x = np.arange(7); w = 0.38
    ax.bar(x - w / 2, truth, width=w * 0.9, color="#c9c9c2", zorder=3)
    ax.bar(x + w / 2, pick, width=w * 0.9, color=C["Q-L"], zorder=3)
    ax.axhline(2000 / 7, color=INK2, lw=1.0, ls=(0, (4, 3)), zorder=4)
    ax.set_xlim(-0.62, 7.45)
    ax.text(6.62, 2000 / 7, "even split", color=INK2, fontsize=8.5,
            ha="left", va="center")
    ax.set_xticks(x, list(LETTERS), fontsize=10, color=INK)
    ax.set_ylim(0, 470)
    ax.set_ylabel("papers", fontsize=9.5)
    ax.set_xlabel("option's position in the shuffled list")
    ax.text(0.30, 0.93, "where the answer actually sat", transform=ax.transAxes,
            fontsize=9.5, color=INK)
    ax.plot([0.275], [0.943], "s", ms=8, color="#c9c9c2",
            transform=ax.transAxes, clip_on=False)
    ax.text(0.30, 0.83, "what the logprobs picked", transform=ax.transAxes,
            fontsize=9.5, color=INK)
    ax.plot([0.275], [0.843], "s", ms=8, color=C["Q-L"],
            transform=ax.transAxes, clip_on=False)
    tidy(ax)
    ax.text(0, 1.10, "The logprobs prefer certain letters", transform=ax.transAxes,
            fontsize=10.5, color=INK)

    fig.text(0.005, 1.15, "Written beats logprobs", ha="left", fontsize=13.5,
             fontweight="bold", color=INK)
    fig.text(0.005, 1.055, "Qwen3.8-27B · same weights, same backend, one sentence changed",
             ha="left", fontsize=9.5, color=INK2)
    fig.tight_layout(w_pad=3.0)
    fig.savefig(f"{OUT}/fig6_mechanism.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    fig_hero()
    fig_scatter()
    fig_grid()
    cov = fig_coverage()
    n_bad = fig_deficit()
    fig_mechanism()
    print("resolution / reliability:")
    for a in TIER:
        rel, res = murphy(a)
        print(f"  {NAME[a]:18s} res {res:.4f}  rel {rel:.4f}  brier {brier(a):.4f}")
    print("\ncoverage @5%:")
    for a, (c, e) in cov.items():
        print(f"  {NAME[a]:18s} {c*100:5.1f}%  realized {e*100:.2f}%")
    print(f"\nruled-out set: {n_bad} papers")
    print("\nwrote 6 figures to results/")
