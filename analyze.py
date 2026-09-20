#!/usr/bin/env python3
"""PROTOCOL.md sections 6, 7 and 8: recalibration, metrics, and contrasts.

Reads the frozen prediction arrays (`preds/*.npz`), the frozen per-arm
reports (`preds/*_report.json`) and the raw response logs (`runs/*.jsonl`).
Writes `results/analysis.md` plus reliability and risk-coverage figures to
`results/`.

Nothing here edits PROTOCOL.md, DEVIATIONS.md, protocol.py, pools.py or
parse.py. It imports protocol.py and pools.py for the frozen constants and
pool manifest, and reuses parse.py's private per-mechanism parsers (never its
epsilon-applying wrapper) so the epsilon-sensitivity check re-derives
probabilities from the raw response exactly the way the frozen pipeline does,
without silently re-flooring an already-floored vector (see DEVIATIONS.md,
2026-09-20, section 7).

Usage:
    ./analyze.py
"""

import json
import os
import sys
from collections import OrderedDict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import numpy as np
from scipy.optimize import minimize_scalar
from scipy.special import logsumexp

import parse as _parse
import pools
import protocol as P

RESULTS_DIR = "results"
PRED_DIR = "preds"
RUNS_DIR = "runs"

LLM_ARMS = ["J", "F1-V", "F2-V", "Q-V", "Q-L"]
REF_ARMS = ["A", "P", "P-star"]
ALL_ARMS = LLM_ARMS + REF_ARMS
DISPLAY = {"P-star": "P*"}


def display(arm):
    return DISPLAY.get(arm, arm)


# --------------------------------------------------------------------------
# Pure metric functions (unit tested in tests/test_analyze.py)
# --------------------------------------------------------------------------

def onehot(y_idx, k=P.K):
    y_idx = np.asarray(y_idx)
    out = np.zeros((len(y_idx), k))
    out[np.arange(len(y_idx)), y_idx] = 1.0
    return out


def brier_per_item(probs, y_idx):
    """Section 7.1: sum_k (p_k - y_k)^2, per item."""
    y = onehot(y_idx, probs.shape[1])
    return np.sum((probs - y) ** 2, axis=1)


def multiclass_brier(probs, y_idx):
    return float(np.mean(brier_per_item(probs, y_idx)))


def accuracy(probs, y_idx):
    return float(np.mean(np.argmax(probs, axis=1) == np.asarray(y_idx)))


def fit_temperature(probs, y_idx, bounds=(0.05, 20.0)):
    """Section 6: T minimizing NLL of q proportional to p^(1/T)."""
    probs = np.asarray(probs, dtype=float)
    y_idx = np.asarray(y_idx)
    logp = np.log(probs)
    rows = np.arange(len(y_idx))

    def nll(T):
        z = logp / T
        logZ = logsumexp(z, axis=1)
        log_q_true = z[rows, y_idx] - logZ
        return -np.mean(log_q_true)

    res = minimize_scalar(nll, bounds=bounds, method="bounded",
                           options={"xatol": 1e-5})
    return float(res.x)


def apply_temperature(probs, T, eps=P.EPSILON):
    """q = normalize(p^(1/T)), then re-floor (section 5 step 3, section 6)."""
    probs = np.asarray(probs, dtype=float)
    logp = np.log(probs) / T
    logZ = logsumexp(logp, axis=1, keepdims=True)
    q = np.exp(logp - logZ)
    return np.vstack([P.floor_renorm(row, eps) for row in q])


def make_folds(y_idx, n_folds=5, seed=P.SEEDS["folds"]):
    """Section 6: 5 folds stratified by label, seed 20260923."""
    y_idx = np.asarray(y_idx)
    n = len(y_idx)
    fold = np.full(n, -1, dtype=int)
    rng = np.random.default_rng(seed)
    for label_idx in range(int(y_idx.max()) + 1):
        idx = np.where(y_idx == label_idx)[0]
        idx = rng.permutation(idx)
        for f, chunk in enumerate(np.array_split(idx, n_folds)):
            fold[chunk] = f
    assert (fold >= 0).all(), "every item must land in a fold"
    return fold


def fit_tau(confidence, correct, target_error=0.05, min_k=50):
    """Section 6: largest k whose top-k error <= target_error; else inf."""
    confidence = np.asarray(confidence, dtype=float)
    correct = np.asarray(correct, dtype=bool)
    n = len(confidence)
    order = np.argsort(-confidence, kind="stable")
    wrong_sorted = (~correct[order]).astype(float)
    cum_wrong = np.cumsum(wrong_sorted)
    ks = np.arange(1, n + 1)
    error_rate = cum_wrong / ks
    feasible = ks[error_rate <= target_error]
    if feasible.size == 0 or feasible.max() < min_k:
        return float("inf")
    k_star = int(feasible.max())
    conf_sorted = confidence[order]
    return float(conf_sorted[k_star - 1])


def ece_equal_mass(confidence, correct, n_bins=15):
    """Section 7.4: top-label ECE, 15 equal-mass bins.

    Returns (ece, bin_stats) where bin_stats is a list of
    (avg_confidence, avg_accuracy, bin_size) for the reliability diagram.
    """
    confidence = np.asarray(confidence, dtype=float)
    correct = np.asarray(correct, dtype=bool).astype(float)
    n = len(confidence)
    order = np.argsort(confidence, kind="stable")
    conf_sorted = confidence[order]
    correct_sorted = correct[order]
    ece = 0.0
    bin_stats = []
    for b in np.array_split(np.arange(n), n_bins):
        if len(b) == 0:
            continue
        avg_conf = float(conf_sorted[b].mean())
        avg_acc = float(correct_sorted[b].mean())
        w = len(b) / n
        ece += w * abs(avg_conf - avg_acc)
        bin_stats.append((avg_conf, avg_acc, len(b)))
    return ece, bin_stats


def murphy_decomposition(confidence, correct, n_bins=15):
    """Reliability / resolution / uncertainty decomposition of the top-label
    binary Brier score (Murphy 1973), used for PC1's fallback framing."""
    _, bin_stats = ece_equal_mass(confidence, correct, n_bins)
    n = sum(w for _, _, w in bin_stats)
    overall_acc = sum(acc * w for _, acc, w in bin_stats) / n
    reliability = sum(w * (conf - acc) ** 2 for conf, acc, w in bin_stats) / n
    resolution = sum(w * (acc - overall_acc) ** 2 for _, acc, w in bin_stats) / n
    uncertainty = overall_acc * (1 - overall_acc)
    return {
        "reliability": reliability, "resolution": resolution,
        "uncertainty": uncertainty,
        "brier_top1": reliability - resolution + uncertainty,
        "overall_acc": overall_acc,
    }


def cross_fit_arm(probs, y_idx, folds, n_folds=5, target_error=0.05, min_k=50):
    """Section 6, applied to one arm's raw probabilities.

    Per fold: fit T and tau (raw and scaled bases) on the other four folds,
    apply to the held-out fold. Never lets a fold see its own data for its
    own parameters.
    """
    n = probs.shape[0]
    oof_scaled = np.zeros_like(probs)
    fold_T, fold_tau_raw, fold_tau_scaled = [], [], []
    accept_raw = np.zeros(n, dtype=bool)
    accept_scaled = np.zeros(n, dtype=bool)
    # argmax is invariant under q = normalize(p^(1/T)) for T > 0, so
    # correctness does not depend on raw vs. scaled.
    correct = np.argmax(probs, axis=1) == np.asarray(y_idx)

    for f in range(n_folds):
        train = folds != f
        test = folds == f

        T = fit_temperature(probs[train], y_idx[train])
        fold_T.append(T)

        scaled_train = apply_temperature(probs[train], T)
        scaled_test = apply_temperature(probs[test], T)
        oof_scaled[test] = scaled_test

        raw_conf_train = probs[train].max(axis=1)
        scaled_conf_train = scaled_train.max(axis=1)
        correct_train = correct[train]

        tau_r = fit_tau(raw_conf_train, correct_train, target_error, min_k)
        tau_s = fit_tau(scaled_conf_train, correct_train, target_error, min_k)
        fold_tau_raw.append(tau_r)
        fold_tau_scaled.append(tau_s)

        accept_raw[test] = probs[test].max(axis=1) >= tau_r
        accept_scaled[test] = scaled_test.max(axis=1) >= tau_s

    return {
        "oof_scaled_probs": oof_scaled,
        "fold_T": fold_T,
        "fold_tau_raw": fold_tau_raw,
        "fold_tau_scaled": fold_tau_scaled,
        "accept_raw": accept_raw,
        "accept_scaled": accept_scaled,
        "correct": correct,
    }


def cross_fit_scalar_confidence(confidence, correct, folds, n_folds=5,
                                 target_error=0.05, min_k=50):
    """Same as the tau half of cross_fit_arm, for a scalar field (J's
    `confidence`) that carries no K-vector to temperature-scale."""
    n = len(confidence)
    accept = np.zeros(n, dtype=bool)
    fold_tau = []
    for f in range(n_folds):
        train = folds != f
        test = folds == f
        tau = fit_tau(confidence[train], correct[train], target_error, min_k)
        fold_tau.append(tau)
        accept[test] = confidence[test] >= tau
    return {"fold_tau": fold_tau, "accept": accept}


def build_resample_matrix(n_items, n_resamples, seed):
    rng = np.random.default_rng(seed)
    return rng.integers(0, n_items, size=(n_resamples, n_items))


def bootstrap_diff_ci(scores_a, scores_b, idx_matrix, alpha=0.05):
    """Paired bootstrap over items: percentile CI on mean(a) - mean(b),
    holding per-item scores fixed and resampling which items are drawn."""
    a = np.asarray(scores_a)[idx_matrix].mean(axis=1)
    b = np.asarray(scores_b)[idx_matrix].mean(axis=1)
    diffs = a - b
    lo, hi = np.percentile(diffs, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    point = float(np.mean(scores_a) - np.mean(scores_b))
    return point, float(lo), float(hi)


# --------------------------------------------------------------------------
# IO helpers
# --------------------------------------------------------------------------

def _npz_path(arm, pool, run_id):
    suffix = "" if run_id == 1 else f"_{run_id}"
    return os.path.join(PRED_DIR, f"{arm}_{pool}{suffix}.npz")


def load_preds(arm, pool="evaluation", run_id=1):
    path = _npz_path(arm, pool, run_id)
    if not os.path.exists(path):
        print(f"WARNING: {path} not found -- skipping {arm}", file=sys.stderr)
        return None
    d = np.load(path, allow_pickle=True)
    return {"ids": d["ids"].tolist(), "probs": d["probs"],
             "statuses": d["statuses"].tolist(), "labels": d["labels"].tolist()}


def align(ids_master, ids_arm, arr):
    pos = {rid: i for i, rid in enumerate(ids_arm)}
    order = [pos[rid] for rid in ids_master]
    return arr[order]


def load_report(arm, pool="evaluation", run_id=1):
    suffix = "" if run_id == 1 else f"_{run_id}"
    path = os.path.join(PRED_DIR, f"{arm}_{pool}{suffix}_report.json")
    if not os.path.exists(path):
        return None
    with open(path) as fh:
        return json.load(fh)


def resolved_records(arm, pool="evaluation", run_id=1):
    """id -> single record, applying parse.py's own dedup preference
    (an 'ok' record beats a 'failure' record for the same id; otherwise
    first occurrence wins). Mirrors parse.parse_run's resolution rule so
    the secondary, JSONL-derived stats (cost, latency, epsilon
    sensitivity) count each item exactly once and the same way the frozen
    pipeline does.
    """
    suffix = "" if run_id == 1 else f"_{run_id}"
    path = os.path.join(RUNS_DIR, f"{arm}_{pool}{suffix}.jsonl")
    if not os.path.exists(path):
        return None
    order, groups = [], {}
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            rid = rec["id"]
            if rid not in groups:
                groups[rid] = []
                order.append(rid)
            groups[rid].append(rec)
    resolved = OrderedDict()
    for rid in order:
        group = groups[rid]
        kept = next((r for r in group if r.get("status") == "ok"), group[0])
        resolved[rid] = kept
    return resolved


def _raw_vector(rec, item, arm):
    """The pre-floor K-vector a raw response record would parse to, using
    parse.py's own per-mechanism parsers (never its epsilon-floor step).
    Returns (vec_or_None, status)."""
    if rec.get("status") != "ok":
        return None, "failure"
    resp = rec.get("response") or {}
    mech = P.ARMS[arm]["mechanism"]
    if mech == "decisions":
        return _parse._parse_decisions(resp, item)
    choices = resp.get("choices") or []
    if not choices:
        return None, "empty"
    choice = choices[0]
    if mech == "letter":
        return _parse._parse_letter(choice, item)
    return _parse._parse_verbalized(choice, item)


def epsilon_sensitivity(arm, items_by_id, epsilons=(1e-4, 1e-2)):
    """Section 7 secondary: primary metric recomputed at other epsilons.

    Re-parses runs/{arm}_evaluation.jsonl from scratch (DEVIATIONS.md,
    2026-09-20): preds/*.npz already has epsilon=0.001 baked in, so
    re-flooring it would be a silent no-op at 1e-4 and a wrong number at
    1e-2. Items that fail to re-parse are skipped and counted.
    """
    recs = resolved_records(arm)
    if recs is None:
        return None
    vecs, y = [], []
    n_skipped = 0
    for rid, rec in recs.items():
        item = items_by_id[rid]
        vec, status = _raw_vector(rec, item, arm)
        if status != "ok":
            n_skipped += 1
            continue
        vecs.append(vec)
        y.append(P.LABELS.index(item["label"]))
    vecs = np.vstack(vecs)
    y = np.array(y)
    out = {"n_total": len(recs), "n_used": len(y), "n_skipped": n_skipped}
    for eps in epsilons:
        scaled = np.vstack([P.floor_renorm(v, eps) for v in vecs])
        out[f"brier_eps_{eps:g}"] = multiclass_brier(scaled, y)
    baseline = np.vstack([P.floor_renorm(v, P.EPSILON) for v in vecs])
    out["brier_eps_baseline_matched_n"] = multiclass_brier(baseline, y)
    return out


def usage_stats(arm, pool="evaluation"):
    """Secondary: cost/1000, reasoning tokens/item, latency p50/p95 (caveated)."""
    recs = resolved_records(arm, pool)
    if recs is None:
        return None
    latencies, costs, reasoning_tokens = [], [], []
    for rid, rec in recs.items():
        lat = rec.get("latency_s")
        if lat is not None:
            latencies.append(lat)
        usage = (rec.get("response") or {}).get("usage") or {}
        cost = usage.get("cost")
        if cost is not None:
            costs.append(cost)
        rt = (usage.get("completion_tokens_details") or {}).get("reasoning_tokens")
        if rt is not None:
            reasoning_tokens.append(rt)
    out = {"n": len(recs)}
    if latencies:
        arr = np.array(latencies)
        out["latency_p50"] = float(np.percentile(arr, 50))
        out["latency_p95"] = float(np.percentile(arr, 95))
    if costs:
        out["cost_per_1000"] = float(np.sum(costs) / len(recs) * 1000)
    if reasoning_tokens:
        out["reasoning_tokens_mean"] = float(np.mean(reasoning_tokens))
    return out


def j_confidence_field(items_by_id, pool="evaluation", run_id=1):
    """J's scalar `confidence` field (distinct from max(probabilities);
    see DEVIATIONS.md 2026-09-20, section 7)."""
    recs = resolved_records("J", pool, run_id)
    conf = {}
    for rid, rec in recs.items():
        resp = rec.get("response") or {}
        ans = (resp.get("answers") or {}).get(P.JEV_QUESTION_KEY) or {}
        c = ans.get("confidence")
        if c is not None:
            conf[rid] = float(c)
    return conf


# --------------------------------------------------------------------------
# Figures
# --------------------------------------------------------------------------

def plot_reliability(arm, bin_stats_raw, bin_stats_scaled, path):
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot([0, 1], [0, 1], linestyle="--", color="#888888", linewidth=1,
            label="perfect calibration")
    if bin_stats_raw:
        c, a, _ = zip(*bin_stats_raw)
        ax.plot(c, a, marker="o", color="#d95f02", label="raw")
    if bin_stats_scaled:
        c, a, _ = zip(*bin_stats_scaled)
        ax.plot(c, a, marker="s", color="#1b9e77", label="temperature-scaled")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("mean predicted confidence (bin)")
    ax.set_ylabel("empirical accuracy (bin)")
    ax.set_title(f"Reliability diagram: {display(arm)}")
    ax.legend(loc="upper left", fontsize=8)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_risk_coverage(curves, path, title):
    """curves: dict[arm] -> (coverage array, error array)."""
    fig, ax = plt.subplots(figsize=(6.5, 5))
    colors = plt.cm.tab10(np.linspace(0, 1, max(len(curves), 1)))
    for (arm, (cov, err)), color in zip(curves.items(), colors):
        ax.plot(cov, err, label=display(arm), color=color, linewidth=1.5)
    ax.axhline(0.05, linestyle="--", color="#888888", linewidth=1,
               label="5% target error")
    ax.set_xlabel("coverage (fraction of items accepted)")
    ax.set_ylabel("error rate among accepted items")
    ax.set_title(title)
    ax.set_xlim(0, 1)
    ax.legend(loc="upper left", fontsize=8)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def risk_coverage_curve(confidence, correct):
    order = np.argsort(-confidence, kind="stable")
    wrong_sorted = (~correct[order]).astype(float)
    n = len(confidence)
    ks = np.arange(1, n + 1)
    err = np.cumsum(wrong_sorted) / ks
    cov = ks / n
    return cov, err


# --------------------------------------------------------------------------
# Report assembly
# --------------------------------------------------------------------------

def fnum(x, nd=4):
    if x is None:
        return "n/a"
    if isinstance(x, float) and np.isnan(x):
        return "n/a (0 coverage)"
    if isinstance(x, float) and not np.isfinite(x):
        return "inf" if x > 0 else "-inf"
    return f"{x:.{nd}f}"


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)

    items = pools.load_pool("evaluation")
    items_by_id = {it["id"]: it for it in items}
    ids_master = [it["id"] for it in items]
    y_idx = np.array([P.LABELS.index(it["label"]) for it in items])
    n = len(ids_master)

    idx_matrix = build_resample_matrix(n, 2000, P.SEEDS["bootstrap"])
    folds = make_folds(y_idx, 5, P.SEEDS["folds"])

    arm_data = {}      # arm -> raw probs (n,K), aligned
    arm_fit = {}        # arm -> cross_fit_arm(...) result
    skipped_arms = []

    for arm in ALL_ARMS:
        d = load_preds(arm)
        if d is None:
            skipped_arms.append(arm)
            continue
        probs = align(ids_master, d["ids"], d["probs"])
        # Sanity: labels line up with the pool's own labels.
        labels_aligned = align(ids_master, d["ids"], np.array(d["labels"]))
        assert list(labels_aligned) == [it["label"] for it in items], (
            f"{arm}: label misalignment after id-based reindexing")
        arm_data[arm] = probs
        arm_fit[arm] = cross_fit_arm(probs, y_idx, folds)

    present_arms = [a for a in ALL_ARMS if a in arm_data]

    # ---- PC2: J's run-to-run noise floor -------------------------------
    pc2 = None
    j2 = load_preds("J", run_id=2)
    if j2 is not None and "J" in arm_data:
        probs_j2 = align(ids_master, j2["ids"], j2["probs"])
        brier_j1 = multiclass_brier(arm_data["J"], y_idx)
        brier_j2 = multiclass_brier(probs_j2, y_idx)
        pc2 = {"brier_run1": brier_j1, "brier_run2": brier_j2,
               "floor": abs(brier_j1 - brier_j2)}
    else:
        print("WARNING: J_evaluation_2.npz missing -- PC2 floor unavailable",
              file=sys.stderr)

    # ---- Per-arm metrics --------------------------------------------------
    metrics = {}
    reliability_bins = {}
    risk_curves_raw, risk_curves_scaled = {}, {}
    for arm in present_arms:
        probs = arm_data[arm]
        fit = arm_fit[arm]
        oof_scaled = fit["oof_scaled_probs"]
        correct = fit["correct"]

        raw_brier_items = brier_per_item(probs, y_idx)
        scaled_brier_items = brier_per_item(oof_scaled, y_idx)

        raw_conf = probs.max(axis=1)
        scaled_conf = oof_scaled.max(axis=1)

        ece_raw, bins_raw = ece_equal_mass(raw_conf, correct)
        ece_scaled, bins_scaled = ece_equal_mass(scaled_conf, correct)
        reliability_bins[arm] = (bins_raw, bins_scaled)

        cov_raw = float(fit["accept_raw"].mean())
        err_raw = (float((~correct[fit["accept_raw"]]).mean())
                   if fit["accept_raw"].any() else float("nan"))
        cov_scaled = float(fit["accept_scaled"].mean())
        err_scaled = (float((~correct[fit["accept_scaled"]]).mean())
                      if fit["accept_scaled"].any() else float("nan"))

        risk_curves_raw[arm] = risk_coverage_curve(raw_conf, correct)
        risk_curves_scaled[arm] = risk_coverage_curve(scaled_conf, correct)

        metrics[arm] = {
            "brier_raw": float(np.mean(raw_brier_items)),
            "brier_scaled": float(np.mean(scaled_brier_items)),
            "brier_raw_items": raw_brier_items,
            "brier_scaled_items": scaled_brier_items,
            "gap": float(np.mean(raw_brier_items) - np.mean(scaled_brier_items)),
            "gap_items": raw_brier_items - scaled_brier_items,
            "accuracy": accuracy(probs, y_idx),
            "ece_raw": ece_raw, "ece_scaled": ece_scaled,
            "coverage_raw": cov_raw, "error_raw": err_raw,
            "coverage_scaled": cov_scaled, "error_scaled": err_scaled,
            "fold_T": fit["fold_T"],
            "fold_tau_raw": fit["fold_tau_raw"],
            "fold_tau_scaled": fit["fold_tau_scaled"],
        }

        plot_reliability(arm, bins_raw, bins_scaled,
                          os.path.join(RESULTS_DIR, f"reliability_{arm}.png"))

    plot_risk_coverage(risk_curves_raw,
                        os.path.join(RESULTS_DIR, "risk_coverage_raw.png"),
                        "Risk-coverage (raw confidence)")
    plot_risk_coverage(risk_curves_scaled,
                        os.path.join(RESULTS_DIR, "risk_coverage_scaled.png"),
                        "Risk-coverage (temperature-scaled confidence)")

    # ---- J's confidence-field special case (DEVIATIONS.md 2026-09-20) ----
    j_conf_metrics = None
    if "J" in arm_data:
        conf_map = j_confidence_field(items_by_id)
        conf_arr = np.array([conf_map.get(rid, np.nan) for rid in ids_master])
        have = ~np.isnan(conf_arr)
        if have.all():
            correct_j = arm_fit["J"]["correct"]
            ece_conf, bins_conf = ece_equal_mass(conf_arr, correct_j)
            fit_conf = cross_fit_scalar_confidence(conf_arr, correct_j, folds)
            cov_conf = float(fit_conf["accept"].mean())
            err_conf = (float((~correct_j[fit_conf["accept"]]).mean())
                        if fit_conf["accept"].any() else float("nan"))
            j_conf_metrics = {
                "ece": ece_conf, "coverage": cov_conf, "error": err_conf,
                "fold_tau": fit_conf["fold_tau"], "bins": bins_conf,
                "mean_confidence": float(np.mean(conf_arr)),
                "mean_max_prob": float(arm_data["J"].max(axis=1).mean()),
            }
        else:
            print(f"WARNING: J confidence field missing for "
                  f"{(~have).sum()} items -- confidence-basis metrics skipped",
                  file=sys.stderr)

    # ---- Bootstrap: each arm's difference from J --------------------------
    bootstrap = {}
    if "J" in metrics:
        j_raw_items = metrics["J"]["brier_raw_items"]
        j_scaled_items = metrics["J"]["brier_scaled_items"]
        j_gap_items = metrics["J"]["gap_items"]
        for arm in present_arms:
            if arm == "J" or arm in REF_ARMS:
                continue
            m = metrics[arm]
            raw_pt, raw_lo, raw_hi = bootstrap_diff_ci(
                m["brier_raw_items"], j_raw_items, idx_matrix)
            scaled_pt, scaled_lo, scaled_hi = bootstrap_diff_ci(
                m["brier_scaled_items"], j_scaled_items, idx_matrix)
            gap_pt, gap_lo, gap_hi = bootstrap_diff_ci(
                m["gap_items"], j_gap_items, idx_matrix)
            bootstrap[arm] = {
                "raw": (raw_pt, raw_lo, raw_hi),
                "scaled": (scaled_pt, scaled_lo, scaled_hi),
                "gap": (gap_pt, gap_lo, gap_hi),
            }

    # C3: Q-L vs Q-V (not vs J)
    c3 = None
    if "Q-L" in metrics and "Q-V" in metrics:
        pt, lo, hi = bootstrap_diff_ci(
            metrics["Q-V"]["brier_raw_items"], metrics["Q-L"]["brier_raw_items"],
            idx_matrix)
        c3 = {"point": pt, "lo": lo, "hi": hi}  # positive => Q-L better (lower Brier)

    # ---- Secondary descriptive ---------------------------------------
    reports = {arm: load_report(arm) for arm in ALL_ARMS}
    usage = {arm: usage_stats(arm) for arm in LLM_ARMS}
    epsilons = {arm: epsilon_sensitivity(arm, items_by_id) for arm in LLM_ARMS}

    # ---- PC1: accuracy spread across J, F1-V, F2-V -------------------
    pc1_arms = [a for a in ("J", "F1-V", "F2-V") if a in metrics]
    pc1_spread = None
    pc1_dropped = None
    murphy = {}
    if len(pc1_arms) >= 2:
        accs = {a: metrics[a]["accuracy"] for a in pc1_arms}
        pc1_spread = (max(accs.values()) - min(accs.values())) * 100
        pc1_dropped = pc1_spread > 8.0
        if pc1_dropped:
            for a in pc1_arms:
                probs = arm_data[a]
                murphy[a] = murphy_decomposition(probs.max(axis=1),
                                                  arm_fit[a]["correct"])

    write_report(
        ids_master=ids_master, present_arms=present_arms,
        skipped_arms=skipped_arms, metrics=metrics, pc2=pc2,
        j_conf_metrics=j_conf_metrics, bootstrap=bootstrap, c3=c3,
        reports=reports, usage=usage, epsilons=epsilons,
        pc1_arms=pc1_arms, pc1_spread=pc1_spread, pc1_dropped=pc1_dropped,
        murphy=murphy,
    )

    # ---- Headline numbers to stdout (for the operator, not the report) --
    print("=== HEADLINE NUMBERS ===")
    for arm in present_arms:
        m = metrics[arm]
        print(f"{arm}: brier_raw={m['brier_raw']:.4f} "
              f"brier_scaled={m['brier_scaled']:.4f} gap={m['gap']:.4f} "
              f"acc={m['accuracy']:.4f} ece_raw={m['ece_raw']:.4f}")
    if pc2:
        print(f"PC2 floor (|J run1 - run2| raw Brier) = {pc2['floor']:.5f} "
              f"(run1={pc2['brier_run1']:.4f}, run2={pc2['brier_run2']:.4f})")
    if j_conf_metrics:
        print(f"J ECE: max(probabilities)={metrics['J']['ece_raw']:.4f}  "
              f"confidence-field={j_conf_metrics['ece']:.4f}")
    for arm, bs in bootstrap.items():
        print(f"J vs {arm}: raw diff(other-J)={bs['raw'][0]:.4f} "
              f"[{bs['raw'][1]:.4f}, {bs['raw'][2]:.4f}]  "
              f"scaled diff={bs['scaled'][0]:.4f} "
              f"[{bs['scaled'][1]:.4f}, {bs['scaled'][2]:.4f}]")
    if c3:
        print(f"C3 Q-L vs Q-V (Q-V minus Q-L raw brier): {c3['point']:.4f} "
              f"[{c3['lo']:.4f}, {c3['hi']:.4f}]")
    if pc1_spread is not None:
        print(f"PC1 accuracy spread (J,F1-V,F2-V): {pc1_spread:.2f} pts "
              f"(dropped={pc1_dropped})")
    print(f"Report written to {os.path.join(RESULTS_DIR, 'analysis.md')}")


def write_report(*, ids_master, present_arms, skipped_arms, metrics, pc2,
                  j_conf_metrics, bootstrap, c3, reports, usage, epsilons,
                  pc1_arms, pc1_spread, pc1_dropped, murphy):
    n = len(ids_master)
    L = []
    L.append("# Jev calibration study -- analysis report\n")
    L.append(f"2,000 evaluation items. Generated by `analyze.py` implementing "
              f"PROTOCOL.md sections 6-8. See `DEVIATIONS.md` for all "
              f"recorded departures from the frozen pre-registration.\n")

    if skipped_arms:
        L.append(f"**Skipped (predictions not found):** "
                 f"{', '.join(display(a) for a in skipped_arms)}. "
                 f"Every table below omits them; re-run once they land.\n")

    # ---- Section 7: metrics table --------------------------------------
    L.append("## Section 7 -- metrics (raw and temperature-scaled)\n")
    L.append("A, P and P* are reference points only -- never competitors, "
             "and P* is an oracle (evaluation-set base rates).\n")
    L.append("| Arm | Brier raw | Brier scaled | Calib. gap | Accuracy | "
             "ECE raw | ECE scaled | Coverage@5%err raw | realized err raw | "
             "Coverage@5%err scaled | realized err scaled |")
    L.append("| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |")
    for arm in present_arms:
        m = metrics[arm]
        label = display(arm) + (" (oracle)" if arm == "P-star" else "")
        L.append(
            f"| {label} | {fnum(m['brier_raw'])} | {fnum(m['brier_scaled'])} | "
            f"{fnum(m['gap'])} | {fnum(m['accuracy'])} | {fnum(m['ece_raw'])} | "
            f"{fnum(m['ece_scaled'])} | {fnum(m['coverage_raw'])} | "
            f"{fnum(m['error_raw'])} | {fnum(m['coverage_scaled'])} | "
            f"{fnum(m['error_scaled'])} |")
    L.append("")
    L.append("Accuracy is identical raw vs. scaled by construction: "
             "q = normalize(p^(1/T)) preserves argmax for T > 0, so "
             "temperature scaling never changes which label an arm predicts, "
             "only its confidence in that label.\n")

    L.append("Per-fold temperature and tau (section 6 cross-fitting; 5 folds, "
             "seed 20260923):\n")
    L.append("| Arm | fold T (5) | fold tau raw (5) | fold tau scaled (5) |")
    L.append("| --- | --- | --- | --- |")
    for arm in present_arms:
        m = metrics[arm]
        Ts = ", ".join(fnum(t, 3) for t in m["fold_T"])
        taus_r = ", ".join(fnum(t, 3) for t in m["fold_tau_raw"])
        taus_s = ", ".join(fnum(t, 3) for t in m["fold_tau_scaled"])
        L.append(f"| {display(arm)} | {Ts} | {taus_r} | {taus_s} |")
    L.append("")

    L.append("Reliability diagrams: " +
             ", ".join(f"`results/reliability_{a}.png`" for a in present_arms) +
             ".\n")
    L.append("Risk-coverage curves: `results/risk_coverage_raw.png`, "
             "`results/risk_coverage_scaled.png`.\n")

    # ---- J's confidence-field special case ------------------------------
    L.append("### Arm J: `max(probabilities)` vs. the `confidence` field\n")
    L.append("Required special case (DEVIATIONS.md, 2026-09-20, section 7): J "
             "returns a scalar `confidence` that disagrees with "
             "max(probabilities) on 1,338/2,000 items, mean 0.840 vs 0.865. "
             "`max(probabilities)` is the cross-arm comparable figure used "
             "everywhere else in this report; both bases are given here. "
             "The Brier and calibration gap use the probability vector only "
             "-- a scalar cannot produce the K-vector a proper scoring rule "
             "needs -- so no scaled/gap variant exists for the confidence "
             "basis.\n")
    if j_conf_metrics and "J" in metrics:
        L.append("| Basis | ECE | Coverage@5%err | realized err |")
        L.append("| --- | --- | --- | --- |")
        L.append(f"| max(probabilities) | {fnum(metrics['J']['ece_raw'])} | "
                 f"{fnum(metrics['J']['coverage_raw'])} | "
                 f"{fnum(metrics['J']['error_raw'])} |")
        L.append(f"| confidence field | {fnum(j_conf_metrics['ece'])} | "
                 f"{fnum(j_conf_metrics['coverage'])} | "
                 f"{fnum(j_conf_metrics['error'])} |")
        L.append("")
    else:
        L.append("*(confidence-field metrics unavailable -- see stderr warning)*\n")

    # ---- PC2 -------------------------------------------------------------
    L.append("## PC2 -- J's run-to-run noise floor\n")
    if pc2:
        L.append(f"J run 1 raw Brier: {fnum(pc2['brier_run1'])}. "
                 f"J run 2 raw Brier: {fnum(pc2['brier_run2'])}. "
                 f"**Floor = {fnum(pc2['floor'])}.** Per section 7: "
                 f"\"any cross-arm margin smaller than it supports no claim, "
                 f"whatever the bootstrap says.\"\n")
    else:
        L.append("*(J_evaluation_2.npz not found -- PC2 floor unavailable; "
                 "no contrast below can be floor-checked.)*\n")

    # ---- Section 8: contrasts -------------------------------------------
    L.append("## Section 8 -- contrasts\n")
    floor = pc2["floor"] if pc2 else None

    def floor_note(margin):
        if floor is None:
            return "*(PC2 floor unavailable -- cannot check)*"
        exceeds = abs(margin) > floor
        return (f"{'exceeds' if exceeds else 'does NOT exceed'} the PC2 floor "
                f"({fnum(floor)})")

    L.append("### C1 -- J vs F1-V, J vs F2-V, raw\n")
    L.append("Diff = (arm's raw Brier) - (J's raw Brier); negative means the "
             "arm has the LOWER, better Brier, i.e. J loses.\n")
    for arm in ("F1-V", "F2-V"):
        if arm in bootstrap:
            pt, lo, hi = bootstrap[arm]["raw"]
            excludes_zero = lo > 0 or hi < 0
            j_wins = pt > 0
            L.append(f"- J vs {arm}: raw Brier diff ({arm} - J) = {fnum(pt)} "
                     f"[{fnum(lo)}, {fnum(hi)}] -- "
                     f"{'J has the lower/better raw Brier' if j_wins else f'{arm} has the lower/better raw Brier (J loses)'}. "
                     f"{'Excludes' if excludes_zero else 'Includes'} zero; "
                     f"margin {floor_note(pt)}.")
    L.append("")

    L.append("### C2 -- same after scaling, plus calibration gaps\n")
    for arm in ("F1-V", "F2-V"):
        if arm in bootstrap:
            pt, lo, hi = bootstrap[arm]["scaled"]
            gpt, glo, ghi = bootstrap[arm]["gap"]
            excludes_zero = lo > 0 or hi < 0
            j_wins = pt > 0
            L.append(f"- J vs {arm}: scaled Brier diff ({arm} - J) = {fnum(pt)} "
                     f"[{fnum(lo)}, {fnum(hi)}] -- "
                     f"{'J has the lower/better scaled Brier' if j_wins else f'{arm} has the lower/better scaled Brier (J loses)'}. "
                     f"{'Excludes' if excludes_zero else 'Includes'} zero; "
                     f"margin {floor_note(pt)}. Calibration-gap diff "
                     f"(gap_{arm} - gap_J) = {fnum(gpt)} [{fnum(glo)}, {fnum(ghi)}] "
                     f"(positive means {arm} lost more to miscalibration than J did).")
    L.append("")

    L.append("### C3 -- Q-L vs Q-V (mechanism, model held fixed)\n")
    if c3:
        # c3["point"] = brier(Q-V) - brier(Q-L): positive means Q-L (the
        # logprob mechanism) has the LOWER (better) Brier; negative means
        # Q-V (verbalized) does. Stated from the sign, not assumed.
        excludes_zero = c3["lo"] > 0 or c3["hi"] < 0
        logprob_wins = c3["point"] > 0
        winner, loser = (("Q-L", "Q-V") if logprob_wins else ("Q-V", "Q-L"))
        mech_name = ("the logprob mechanism (Q-L)" if logprob_wins
                     else "the verbalized mechanism (Q-V)")
        L.append(f"Raw Brier diff (Q-V - Q-L) = {fnum(c3['point'])} "
                 f"[{fnum(c3['lo'])}, {fnum(c3['hi'])}]: {winner} has the "
                 f"lower (better) raw Brier than {loser} by "
                 f"{fnum(abs(c3['point']))}, i.e. {mech_name} wins this "
                 f"matched-model contrast. "
                 f"{'Excludes' if excludes_zero else 'Includes'} zero; "
                 f"margin {floor_note(c3['point'])}.\n")
    else:
        L.append("*(Q-V or Q-L predictions unavailable.)*\n")

    L.append("### C4 -- J vs Q-L, raw (asymmetric)\n")
    if "Q-L" in bootstrap:
        pt, lo, hi = bootstrap["Q-L"]["raw"]
        j_lower = metrics["J"]["brier_raw"] < metrics["Q-L"]["brier_raw"]
        L.append(f"Raw Brier diff (Q-L - J) = {fnum(pt)} [{fnum(lo)}, {fnum(hi)}]. "
                 f"margin {floor_note(pt)}.\n")
        if j_lower:
            L.append("**J beats Q-L on raw Brier: this is a WEAK result.** "
                     "Q-L is a 27B open-weights model, so a J win here "
                     "reintroduces the model-quality confound -- it does not "
                     "establish that J's mechanism, specifically, is what "
                     "wins.\n")
        else:
            L.append("**J loses to Q-L on raw Brier: this is DECISIVE.** "
                     "As required by section 8: \"Jev's advantage is its "
                     "probability mechanism, not its model.\"\n")
    else:
        L.append("*(Q-L predictions unavailable.)*\n")

    # ---- Attribution ------------------------------------------------------
    L.append("### Attribution: how much of C1 does C3 explain?\n")
    if c3 and "F1-V" in bootstrap and "F2-V" in bootstrap and floor is not None:
        # J's advantage over each baseline, signed so positive = J better
        # (lower Brier). Use the SMALLER of the two as "J's C1 margin" --
        # the conservative, guaranteed-against-both-baselines figure.
        j_adv_f1v = bootstrap["F1-V"]["raw"][0]
        j_adv_f2v = bootstrap["F2-V"]["raw"][0]
        c1_margin = min(j_adv_f1v, j_adv_f2v)
        # c3["point"] = brier(Q-V) - brier(Q-L): positive means the native/
        # logprob mechanism (Q-L) wins -- the direction that could, in
        # principle, explain a J win via "native probabilities help".
        mech_effect = c3["point"]

        L.append(f"J's advantage over F1-V (F1-V brier - J brier) = "
                 f"{fnum(j_adv_f1v)}; over F2-V = {fnum(j_adv_f2v)}. "
                 f"Conservative C1 margin (the smaller of the two) = "
                 f"{fnum(c1_margin)}. C3's mechanism effect (brier(Q-V) - "
                 f"brier(Q-L); positive favours the native/logprob "
                 f"mechanism) = {fnum(mech_effect)}.\n")

        if c1_margin <= 0:
            L.append(f"**Attribution does not apply: J does not beat both "
                     f"frontier baselines on raw Brier.** There is no "
                     f"positive C1 margin for a mechanism to explain -- see "
                     f"the \"better out of the box\" verdict above, which "
                     f"already fails on this basis. Separately, and without "
                     f"implying anything about J: C3 shows "
                     f"{'the native/logprob mechanism (Q-L) beating the verbalized one (Q-V)' if mech_effect > 0 else 'the verbalized mechanism (Q-V) beating the native/logprob one (Q-L)'} "
                     f"by {fnum(abs(mech_effect))} on the one model where "
                     f"both are measured.\n")
        elif mech_effect <= 0:
            L.append(f"**J's positive C1 margin cannot be attributed to the "
                     f"probability mechanism.** C3 shows the OPPOSITE "
                     f"direction on the matched 27B model -- the verbalized "
                     f"mechanism (Q-V) beat the native/logprob one (Q-L) by "
                     f"{fnum(abs(mech_effect))} there. Whatever explains J's "
                     f"margin over the frontier baselines, this study's own "
                     f"mechanism contrast does not support attributing it to "
                     f"\"native probabilities help\"; it remains unexplained "
                     f"by anything measured here.\n")
        else:
            excess = c1_margin - mech_effect
            L.append(f"Both positive: comparing {fnum(c1_margin)} to "
                     f"{fnum(mech_effect)}, difference = {fnum(excess)}, "
                     f"judged against the PC2 floor ({fnum(floor)}) as the "
                     f"only pre-existing yardstick for \"comparable\" in "
                     f"this study.\n")
            if abs(excess) <= floor:
                L.append("**Comparable to the PC2 floor: C1 is reported as "
                         "consistent with mechanism alone. No model-quality "
                         "claim is made.**\n")
            elif excess > floor:
                L.append(f"**J's margin substantially exceeds the mechanism "
                         f"effect.** The excess ({fnum(excess)}) is the part "
                         f"not explained by mechanism alone -- an estimate "
                         f"carrying the transfer caveat (C3 is measured on a "
                         f"27B model; section 9), not a clean decomposition.\n")
            else:
                L.append("**The mechanism effect exceeds C1's margin.** No "
                         "excess to attribute to anything beyond mechanism.\n")
    else:
        L.append("*(Cannot compute -- one or more of C1/C3/PC2 is unavailable.)*\n")

    # ---- Decision rules ----------------------------------------------------
    L.append("## Section 8 decision rules -- verdicts\n")

    L.append("**\"Better out of the box than typical LLM usage\"** -- holds "
             "only if J beats BOTH F1-V and F2-V on raw Brier, intervals "
             "excluding zero, margins above the PC2 floor.\n")
    if "F1-V" in bootstrap and "F2-V" in bootstrap and floor is not None:
        checks = {}
        for arm in ("F1-V", "F2-V"):
            pt, lo, hi = bootstrap[arm]["raw"]
            j_beats = pt > 0  # other arm's Brier - J's Brier > 0 means J lower
            excludes_zero = lo > 0 or hi < 0
            above_floor = abs(pt) > floor
            checks[arm] = j_beats and excludes_zero and above_floor
        verdict = all(checks.values())
        L.append(f"**Verdict: {'HOLDS' if verdict else 'DOES NOT HOLD'}.** "
                 f"F1-V check passed={checks['F1-V']}, "
                 f"F2-V check passed={checks['F2-V']}.\n")
    else:
        L.append("*(Cannot evaluate -- missing data.)*\n")

    L.append("**\"Calibrated\"** -- holds only if J's raw ECE is below 0.05 "
             "AND its calibration-gap interval includes values near zero. "
             "Reported for both of J's ECE bases.\n")
    if "J" in metrics:
        ece_maxprob = metrics["J"]["ece_raw"]
        maxprob_ok = ece_maxprob < 0.05
        L.append(f"- max(probabilities) basis: raw ECE = {fnum(ece_maxprob)} "
                 f"({'< 0.05' if maxprob_ok else '>= 0.05, FAILS the cutoff'}).")
        if j_conf_metrics:
            ece_conf = j_conf_metrics["ece"]
            conf_ok = ece_conf < 0.05
            L.append(f"- confidence-field basis: raw ECE = {fnum(ece_conf)} "
                     f"({'< 0.05' if conf_ok else '>= 0.05, FAILS the cutoff'}). "
                     f"No calibration-gap variant exists for this basis "
                     f"(scalar, cannot be temperature-scaled).")
        L.append(f"**Verdict on max(probabilities): "
                 f"{'HOLDS' if maxprob_ok else 'DOES NOT HOLD'}** "
                 f"(ECE cutoff alone determines it here; the calibration-gap "
                 f"leg is moot once the ECE cutoff already fails, and is "
                 f"reported above under C2 regardless).\n")
    else:
        L.append("*(J predictions unavailable.)*\n")

    L.append("**A, P and P\\* are excluded from every contrast and every "
             "verdict above.** They appear only in the section 7 table. "
             "P\\* is labelled an oracle throughout.\n")

    # ---- PC1 ---------------------------------------------------------------
    L.append("## PC1 -- capability matching (accuracy spread)\n")
    if pc1_spread is not None:
        L.append(f"Accuracy spread across {', '.join(display(a) for a in pc1_arms)}"
                 f" = {pc1_spread:.2f} points. Q-V and Q-L excluded (mechanism "
                 f"arms). ")
        if pc1_dropped:
            L.append("**Exceeds 8 points: the tier-matched framing is dropped.** "
                     "The comparison is reported through the Brier scores in "
                     "section 7's table (already the primary metric) rather "
                     "than through accuracy, plus a top-label binary-Brier "
                     "decomposition (Murphy 1973) below.\n")
            L.append("| Arm | reliability | resolution | uncertainty | "
                     "top-1 binary Brier | overall top-1 accuracy |")
            L.append("| --- | --- | --- | --- | --- | --- |")
            for a in pc1_arms:
                if a in murphy:
                    d = murphy[a]
                    L.append(f"| {display(a)} | {fnum(d['reliability'])} | "
                             f"{fnum(d['resolution'])} | {fnum(d['uncertainty'])} | "
                             f"{fnum(d['brier_top1'])} | {fnum(d['overall_acc'])} |")
            L.append("")
        else:
            L.append("**8 points not exceeded: the tier-matched framing "
                     "stands.**\n")
    else:
        L.append("*(Fewer than two of J, F1-V, F2-V available.)*\n")

    # ---- Secondary, descriptive ---------------------------------------
    L.append("## Secondary, descriptive\n")

    L.append("### Failure / truncation / tag-leakage / integrity counts "
             "(from `preds/*_report.json`)\n")
    L.append("| Arm | n | ok | failures (total) | tag leakage | duplicates "
             "dropped | offsum count | argmax ties |")
    L.append("| --- | --- | --- | --- | --- | --- | --- | --- |")
    for arm in ALL_ARMS:
        r = reports.get(arm)
        if not r:
            continue
        fails = r.get("failures", {})
        fail_detail = ", ".join(f"{k}:{v}" for k, v in fails.items()) or "none"
        L.append(f"| {display(arm)} | {r.get('n')} | {r.get('ok')} | "
                 f"{r.get('failure_total', 0)} ({fail_detail}) | "
                 f"{r.get('tag_leakage', 'n/a')} | "
                 f"{r.get('duplicates_dropped', 'n/a')} | "
                 f"{r.get('offsum_count', 'n/a')} | "
                 f"{r.get('argmax_tie_count', 'n/a')} |")
    L.append("")

    L.append("### Reasoning tokens, cost, latency (from `runs/*.jsonl`)\n")
    L.append("**Latency caveat (DEVIATIONS.md, 2026-09-20):** arms ran "
             "concurrently and F1-V was throttled to a 3.2s minimum interval; "
             "no dedicated 200-item sequential subsample was run. These "
             "figures are NOT comparable across arms and are NOT the section "
             "7 figure -- reported for the record only.\n")
    L.append("| Arm | n | mean reasoning tokens/item | cost per 1,000 items ($) | "
             "latency p50 (s) | latency p95 (s) |")
    L.append("| --- | --- | --- | --- | --- | --- |")
    for arm in LLM_ARMS:
        u = usage.get(arm)
        if not u:
            continue
        rt = u.get("reasoning_tokens_mean")
        L.append(f"| {display(arm)} | {u.get('n')} | "
                 f"{fnum(rt, 1) if rt is not None else 'n/a'} | "
                 f"{fnum(u.get('cost_per_1000'), 3)} | "
                 f"{fnum(u.get('latency_p50'), 3)} | "
                 f"{fnum(u.get('latency_p95'), 3)} |")
    L.append("")

    L.append("### Epsilon sensitivity (primary metric at epsilon 1e-4 and "
             "1e-2)\n")
    L.append("**Re-parsed from `runs/*.jsonl` raw responses, not from "
             "`preds/*.npz`** (DEVIATIONS.md, 2026-09-20): the stored arrays "
             "are already floored at epsilon=0.001, so re-flooring them is a "
             "silent no-op at 1e-4 and wrong at 1e-2. Items that failed to "
             "re-parse are skipped and counted below; the epsilon=0.001 "
             "column here is recomputed on the same re-parsed subset as the "
             "other two, for a same-n comparison, and can differ slightly "
             "from the section-7 baseline (computed on all 2,000 items via "
             "`preds/*.npz`) for that reason.\n")
    L.append("| Arm | n parsed / total | n skipped | Brier @1e-4 | "
             "Brier @0.001 (same subset) | Brier @1e-2 |")
    L.append("| --- | --- | --- | --- | --- | --- |")
    for arm in LLM_ARMS:
        e = epsilons.get(arm)
        if not e:
            continue
        L.append(f"| {display(arm)} | {e['n_used']}/{e['n_total']} | "
                 f"{e['n_skipped']} | {fnum(e.get('brier_eps_0.0001'))} | "
                 f"{fnum(e.get('brier_eps_baseline_matched_n'))} | "
                 f"{fnum(e.get('brier_eps_0.01'))} |")
    L.append("")

    L.append("---\n")
    L.append("Full four-contrast table and reference-arm figures are in the "
             "tables above; see `PROTOCOL.md` for the pre-registration text "
             "and `DEVIATIONS.md` for every recorded departure.\n")

    with open(os.path.join(RESULTS_DIR, "analysis.md"), "w") as fh:
        fh.write("\n".join(L))


if __name__ == "__main__":
    main()
