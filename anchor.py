#!/usr/bin/env python3
"""PROTOCOL.md section 3: the three non-LLM reference arms, A, P and P*.

- A  -- TF-IDF + multinomial logistic regression, fit only on the 5,503-item
        anchor pool. "Anchor, not a competitor."
- P  -- Anchor-pool class base rates, one constant vector for every item.
        "Degenerate reference": near-zero ECE, near-zero calibration gap,
        zero resolution, worst Brier (section 7) -- by construction.
- P* -- Evaluation-pool class base rates, one constant vector for every
        item. Deliberately uses test-set information. "Oracle, uses test
        information" (section 3); labelled an oracle wherever it appears
        (section 8).

All three are reference points, never competitors (section 8), and all three
are excluded from the cross-arm contrasts. They still go through the exact
same probability handling as every LLM arm: floor at epsilon and renormalize
(section 5 step 3), applied to every row exactly once.

Output format matches parse.py's LLM-arm output exactly, so every downstream
step (devcheck.py-style validation, the analysis stage) can read A, P and P*
the same way it reads J, F1-V, F2-V, Q-V and Q-L:

    preds/{arm}_{pool}.npz          ids, probs (n x K, canonical LABELS
                                     order), statuses, labels
    preds/{arm}_{pool}_report.json  companion metadata

`*` cannot appear in a filename people will glob and shell-quote without
thinking, so P* is written as "P-star" on disk; the report JSON still
carries the literal arm id and, for P-star, an explicit oracle flag.

Usage:
    ./anchor.py --arm A --pool evaluation
    ./anchor.py --arm P --pool dev
    ./anchor.py --arm P-star --pool evaluation
"""

import argparse
import json
import os

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

import pools
import protocol as P

PRED_DIR = "preds"
ARM_NAMES = ["A", "P", "P-star"]
POOL_NAMES = ["evaluation", "dev"]

# Arm A's hyperparameters. Sensible defaults, fixed here and echoed into the
# report so they are checkable -- not tuned against the evaluation set, and
# the evaluation set is never touched until predict() is called on it.
TFIDF_PARAMS = dict(
    lowercase=True,
    stop_words="english",
    ngram_range=(1, 2),
    min_df=2,
    max_features=50000,
    sublinear_tf=True,
)
LOGREG_PARAMS = dict(
    solver="lbfgs",
    max_iter=2000,
    C=1.0,
    random_state=P.SEEDS["master"],
)


def text_for(item):
    """Title plus abstract, plain text -- section 2's task definition."""
    return f"{item['title']} {item['abstract']}"


def class_counts(items):
    """LABELS-ordered counts of `label` over a pool's items."""
    counts = np.zeros(P.K, dtype=float)
    for it in items:
        counts[P.LABELS.index(it["label"])] += 1
    return counts


def base_rate_vector(items):
    """This pool's class base rates, floored and renormalized once.

    Every row served from this vector is identical, so applying floor_renorm
    once and tiling the result is the same as applying it to each row
    separately -- the function is a pure, deterministic map from one vector
    to one vector.
    """
    counts = class_counts(items)
    return P.floor_renorm(counts / counts.sum())


def fit_arm_a(anchor_items):
    """Fit TF-IDF + multinomial logistic regression on the anchor pool only.

    Returns (vectorizer, classifier, column_index) where column_index maps
    canonical LABELS order to the classifier's `classes_` order, since
    LogisticRegression sorts classes lexicographically and that is not
    canonical LABELS order.
    """
    texts = [text_for(it) for it in anchor_items]
    y = [it["label"] for it in anchor_items]

    vectorizer = TfidfVectorizer(**TFIDF_PARAMS)
    X = vectorizer.fit_transform(texts)

    clf = LogisticRegression(**LOGREG_PARAMS)
    clf.fit(X, y)

    column_index = [list(clf.classes_).index(label) for label in P.LABELS]
    return vectorizer, clf, column_index


def predict_arm_a(vectorizer, clf, column_index, target_items):
    """predict_proba on target_items, reindexed into canonical LABELS order."""
    texts = [text_for(it) for it in target_items]
    X = vectorizer.transform(texts)
    raw = clf.predict_proba(X)  # n x K, in clf.classes_ order
    return raw[:, column_index]  # n x K, in canonical LABELS order


def brier_score(probs, true_labels):
    """Multiclass Brier score: mean over items of sum_k (p_k - y_k)^2."""
    y = np.zeros_like(probs)
    for i, label in enumerate(true_labels):
        y[i, P.LABELS.index(label)] = 1.0
    return float(np.mean(np.sum((probs - y) ** 2, axis=1)))


def build(arm, pool):
    """Compute (ids, probs, statuses, labels, extra_report_fields) for one arm."""
    target_items = pools.load_pool(pool)
    ids = [it["id"] for it in target_items]
    true_labels = [it["label"] for it in target_items]
    n = len(target_items)

    if arm == "A":
        anchor_items = pools.load_pool("anchor")
        vectorizer, clf, column_index = fit_arm_a(anchor_items)

        anchor_ids = {it["id"] for it in anchor_items}
        eval_ids = {it["id"] for it in pools.load_pool("evaluation")}
        assert not (anchor_ids & eval_ids), (
            "anchor pool leaked evaluation ids -- arm A must never train on "
            "evaluation-pool items")

        raw = predict_arm_a(vectorizer, clf, column_index, target_items)
        probs = np.vstack([P.floor_renorm(row) for row in raw])

        pred_labels = [P.LABELS[i] for i in np.argmax(probs, axis=1)]
        accuracy = accuracy_score(true_labels, pred_labels)

        extra = {
            "vectorizer": "TfidfVectorizer",
            "vectorizer_params": {
                k: (list(v) if isinstance(v, tuple) else v)
                for k, v in TFIDF_PARAMS.items()
            },
            "classifier": "LogisticRegression",
            "classifier_params": LOGREG_PARAMS,
            "anchor_pool_size": len(anchor_items),
            "vocabulary_size": len(vectorizer.vocabulary_),
            "accuracy": accuracy,
            "brier": brier_score(probs, true_labels),
            "fit_on": "anchor",
            "no_evaluation_leakage": True,
        }

    elif arm == "P":
        anchor_items = pools.load_pool("anchor")
        vec = base_rate_vector(anchor_items)
        probs = np.tile(vec, (n, 1))

        pred_labels = [P.LABELS[i] for i in np.argmax(probs, axis=1)]
        extra = {
            "base_rate_source": "anchor",
            "base_rate_pool_size": len(anchor_items),
            "vector": vec.tolist(),
            "accuracy": accuracy_score(true_labels, pred_labels),
            "brier": brier_score(probs, true_labels),
        }

    elif arm == "P-star":
        evaluation_items = pools.load_pool("evaluation")
        vec = base_rate_vector(evaluation_items)
        probs = np.tile(vec, (n, 1))

        pred_labels = [P.LABELS[i] for i in np.argmax(probs, axis=1)]
        extra = {
            "base_rate_source": "evaluation",
            "base_rate_pool_size": len(evaluation_items),
            "vector": vec.tolist(),
            "accuracy": accuracy_score(true_labels, pred_labels),
            "brier": brier_score(probs, true_labels),
            "oracle": True,
            "note": (
                "P* uses evaluation-set base rates, i.e. test-set "
                "information. PROTOCOL.md section 3 calls it an oracle "
                "that 'uses test information', and section 8 requires it "
                "be labelled an oracle wherever it appears."
            ),
        }

    else:
        raise ValueError(f"unknown arm {arm!r}")

    statuses = np.array(["ok"] * n)
    return np.array(ids), probs, statuses, np.array(true_labels), extra


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--arm", required=True, choices=ARM_NAMES)
    ap.add_argument("--pool", required=True, choices=POOL_NAMES)
    args = ap.parse_args()

    ids, probs, statuses, labels, extra = build(args.arm, args.pool)

    assert probs.shape == (len(ids), P.K)
    assert np.all(statuses == "ok")
    assert np.allclose(probs.sum(axis=1), 1.0, atol=1e-9)
    assert (probs >= P.EPSILON - 1e-12).all()

    report = {
        "arm": args.arm,
        "pool": args.pool,
        "n": len(ids),
        "ok": len(ids),
        "epsilon": P.EPSILON,
    }
    report.update(extra)

    os.makedirs(PRED_DIR, exist_ok=True)
    np.savez(os.path.join(PRED_DIR, f"{args.arm}_{args.pool}.npz"),
              ids=ids, probs=probs, statuses=statuses, labels=labels)
    with open(os.path.join(PRED_DIR, f"{args.arm}_{args.pool}_report.json"),
              "w") as fh:
        json.dump(report, fh, indent=2)

    print(f"{args.arm} on {args.pool}: n={len(ids)} "
          f"accuracy={extra.get('accuracy'):.4f} brier={extra.get('brier'):.4f}")


if __name__ == "__main__":
    main()
