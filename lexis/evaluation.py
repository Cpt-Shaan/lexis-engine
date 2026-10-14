"""Effectiveness metrics and significance tests.

Binary metrics (P@10, AP, 11-point P/R) count grade >= 1 as relevant.
NDCG@10 uses graded gains 2^grade - 1 with a log2(rank + 1) discount; the
ideal ranking is built from every judged document of the query.  Unjudged
documents are non-relevant.
"""

import math

import numpy as np
from scipy import stats


def precision_at(ranked, qrels, k=10):
    return sum(1 for d in ranked[:k] if qrels.get(d, 0) > 0) / k


def average_precision(ranked, qrels):
    R = sum(1 for g in qrels.values() if g > 0)
    if R == 0:
        return 0.0
    hits = 0
    s = 0.0
    for i, d in enumerate(ranked, 1):
        if qrels.get(d, 0) > 0:
            hits += 1
            s += hits / i
    return s / R


def ndcg_at(ranked, qrels, k=10):
    dcg = sum((2 ** qrels.get(d, 0) - 1) / math.log2(i + 2) for i, d in enumerate(ranked[:k]))
    ideal = sorted(qrels.values(), reverse=True)[:k]
    idcg = sum((2 ** g - 1) / math.log2(i + 2) for i, g in enumerate(ideal))
    return dcg / idcg if idcg > 0 else 0.0


def interpolated_pr(ranked, qrels, points=11):
    """11-point interpolated precision (IIR Sec. 8.4)."""
    R = sum(1 for g in qrels.values() if g > 0)
    levels = np.linspace(0, 1, points)
    if R == 0:
        return np.zeros(points)
    prec, rec = [], []
    hits = 0
    for i, d in enumerate(ranked, 1):
        if qrels.get(d, 0) > 0:
            hits += 1
        prec.append(hits / i)
        rec.append(hits / R)
    prec, rec = np.array(prec), np.array(rec)
    out = []
    for r in levels:
        m = prec[rec >= r - 1e-12] if len(rec) else np.array([])
        out.append(m.max() if m.size else 0.0)
    return np.array(out)


def evaluate(runs, qrels):
    """runs: {qid: ranked docIDs} -> dict of mean metrics + per-query AP."""
    qids = sorted(runs)
    ap = np.array([average_precision(runs[q], qrels.get(q, {})) for q in qids])
    p10 = np.array([precision_at(runs[q], qrels.get(q, {})) for q in qids])
    nd = np.array([ndcg_at(runs[q], qrels.get(q, {})) for q in qids])
    pr = np.mean([interpolated_pr(runs[q], qrels.get(q, {})) for q in qids], axis=0)
    return {"MAP": ap.mean(), "P@10": p10.mean(), "NDCG@10": nd.mean(), "pr11": pr,
            "ap": dict(zip(qids, ap)), "qids": qids}


def paired_tests(a, b):
    """Two-sided paired t-test and Wilcoxon signed-rank test over per-query scores."""
    a, b = np.asarray(a), np.asarray(b)
    t = stats.ttest_rel(a, b)
    try:
        w = stats.wilcoxon(a, b).pvalue
    except ValueError:      # all differences zero
        w = 1.0
    return float(t.pvalue) if not np.isnan(t.pvalue) else 1.0, float(w)
