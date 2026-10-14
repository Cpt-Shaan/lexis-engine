"""Ranking: tune on training queries, run once on test queries; ablations with significance tests.

Systems
  Boolean (AND)   all non-stop query words ANDed by the exact engine; results in docID order
  Boolean (OR)    same words ORed (reported for context)
  raw-tf          cosine over raw-tf vectors, no idf, one zone (title+body): nnc.nnc, stem+stop
  lnc.ltc         the textbook default (single zone), for reference
  Lexis best      best training-MAP configuration from the grid below
"""

import itertools
import json
import os
import time

import numpy as np

from common import DATA, FIGS, INDEX, RESULTS, ensure_index, md_table, plt, save_json, save_md
from lexis.boolean import BooleanEngine
from lexis.corpus import load_qrels, load_queries
from lexis.evaluation import average_precision, evaluate, paired_tests
from lexis.index import Index
from lexis.ranking import Collection, RankConfig, batch_rankings, rank_from_scores
from lexis.text import STOP_WORDS, tokenize
from lexis.tolerant import KGramIndex

GRID = {
    "doc_tf": ["n", "l"],
    "doc_idf": ["n", "t"],
    "q_tf": ["n", "l"],
    "q_idf": ["n", "t", "p"],
    "slope": [1.0, 0.8, 0.6, 0.4, 0.3, 0.2],
    "g": [None, 0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],   # None = zones off
}


def boolean_runs(ix, queries, op):
    eng = BooleanEngine(ix, KGramIndex(ix.vocabulary))
    runs = {}
    for qid, q in queries:
        words = [w for w in tokenize(q) if w not in STOP_WORDS]
        runs[qid] = eng.search(f" {op} ".join(words)) if words else []
    return runs


def mean_ap(S, doc_ids, qids, qrels):
    return np.array([average_precision(rank_from_scores(S[:, j], doc_ids), qrels.get(q, {}))
                     for j, q in enumerate(qids)])


def tune(colls, train, qrels):
    """Exhaustive grid on training MAP.  Score matrices are cached per (analyzer, doc
    weighting, query weighting), so each g value is just a linear combination."""
    qids = [q for q, _ in train]
    texts = [t for _, t in train]
    results = []
    t0 = time.time()
    for (stem, stop), coll in colls.items():
        for dtf, didf, slope, qtf, qidf in itertools.product(GRID["doc_tf"], GRID["doc_idf"], GRID["slope"],
                                                             GRID["q_tf"], GRID["q_idf"]):
            base = RankConfig(stem=stem, stop=stop, doc_tf=dtf, doc_idf=didf, slope=slope, q_tf=qtf, q_idf=qidf)
            Q = coll.query_matrix(texts, base)
            St = (coll.zone_weights("title", dtf, didf, "c", slope) @ Q).toarray()
            Sb = (coll.zone_weights("body", dtf, didf, "c", slope) @ Q).toarray()
            Sa = (coll.zone_weights("all", dtf, didf, "c", slope) @ Q).toarray()
            for g in GRID["g"]:
                S = Sa if g is None else g * St + (1 - g) * Sb
                cfg = base.with_(zones=g is not None, g=0.0 if g is None else g)
                ap = mean_ap(S, coll.doc_ids, qids, qrels)
                results.append((ap.mean(), cfg))
    print(f"grid: {len(results)} configs in {time.time() - t0:.0f}s")
    return results


def complexity(cfg):
    """Tie-break: prefer fewer moving parts."""
    return (cfg.slope != 1.0) + (cfg.doc_idf != "n") + cfg.zones


def run_cfg(colls, cfg, queries):
    coll = colls[(cfg.stem, cfg.stop)]
    R = batch_rankings(coll, cfg, [q for _, q in queries])
    return {qid: r for (qid, _), r in zip(queries, R)}


def main():
    ensure_index()
    ix = Index(INDEX)
    train = load_queries(os.path.join(DATA, "queries_train.tsv"))
    test = load_queries(os.path.join(DATA, "queries_test.tsv"))
    qrels = load_qrels(os.path.join(DATA, "qrels.tsv"))
    colls = {(s, p): Collection(ix, s, p) for s in (True, False) for p in (True, False)}

    grid = tune(colls, train, qrels)
    grid.sort(key=lambda x: (-round(x[0], 6), complexity(x[1])))
    # The specification requires title and body to be weighted separately, so the
    # system is chosen among zone-weighted configurations with 0 < g < 1.  The
    # unconstrained optimum is reported alongside for transparency.
    free_map, free_best = grid[0]
    best_map, best = next((m, c) for m, c in grid if c.zones and 0 < c.g < 1)
    print("best", best.describe(), best_map, "| unconstrained", free_best.describe(), free_map)
    with open(os.path.join(INDEX, "rank_config.json"), "w") as f:
        json.dump(best.__dict__, f, indent=2)
    save_json("grid", [{"map": m, **c.__dict__} for m, c in grid])

    raw = RankConfig(doc_tf="n", doc_idf="n", q_tf="n", q_idf="n", zones=False)
    lnc = RankConfig(zones=False)
    systems = [("Boolean (AND)", None, "AND"), ("Boolean (OR)", None, "OR"),
               ("raw-tf (nnc.nnc, one zone)", raw, None), ("lnc.ltc (one zone)", lnc, None),
               (f"**Lexis best**: {best.describe()}", best, None)]

    evals = {}
    rows = []
    for name, cfg, op in systems:
        rec = {}
        for split, qs in (("train", train), ("test", test)):
            runs = boolean_runs(ix, qs, op) if cfg is None else run_cfg(colls, cfg, qs)
            rec[split] = evaluate(runs, qrels)
        evals[name] = rec
        rows.append([name] + [f"{rec[s][m]:.4f}" for s in ("train", "test") for m in ("P@10", "MAP", "NDCG@10")])
    md = md_table(["System", "train P@10", "train MAP", "train NDCG@10", "test P@10", "test MAP", "test NDCG@10"],
                  rows, ["---"] + ["---:"] * 6)

    best_name = systems[-1][0]
    sig = []
    for name, _, _ in systems[:-1]:
        a = [evals[best_name]["test"]["ap"][q] for q in evals[best_name]["test"]["qids"]]
        b = [evals[name]["test"]["ap"][q] for q in evals[name]["test"]["qids"]]
        pt, pw = paired_tests(a, b)
        sig.append([f"best vs {name}", f"{np.mean(a) - np.mean(b):+.4f}", f"{pt:.2g}", f"{pw:.2g}"])
    md += "\n\n" + md_table(["Test-set comparison (per-query AP)", "ΔMAP", "p (paired t)", "p (Wilcoxon)"], sig)

    # 11-point curves
    levels = np.linspace(0, 1, 11)
    pr_rows = []
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.4), sharey=True)
    for ax, split in zip(axes, ("train", "test")):
        for name, _, _ in systems:
            if name == "Boolean (OR)":
                continue
            ax.plot(levels, evals[name][split]["pr11"], marker="o", ms=3,
                    label=name.replace("**", "").split(":")[0])
        ax.set(xlabel="recall", ylabel="interpolated precision", title=f"11-point P/R ({split})",
               xlim=(0, 1), ylim=(0, 1.02))
        ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(f"{FIGS}/pr11.png")
    for name, _, _ in systems:
        pr_rows.append([name.split(":")[0]] + [f"{x:.3f}" for x in evals[name]["test"]["pr11"]])
    md_pr = md_table(["Test set"] + [f"{r:.1f}" for r in levels], pr_rows)

    # ---------------------------------------------------------------- ablations
    abl = [("stemming off", best.with_(stem=not best.stem)),
           ("stop words off" if best.stop else "stop words on (best has them off)", best.with_(stop=not best.stop)),
           ("idf off (doc and query)", best.with_(doc_idf="n", q_idf="n")),
           ("length normalization off", best.with_(doc_norm="n")),
           ("zone weighting off (title+body as one zone)", best.with_(zones=False))]
    ab_rows = []
    ab_data = {}
    for name, cfg in abl:
        rec = {s: evaluate(run_cfg(colls, cfg, qs), qrels) for s, qs in (("train", train), ("test", test))}
        ab_data[name] = rec
        cells = [name]
        for s in ("train", "test"):
            base = evals[best_name][s]
            a = [base["ap"][q] for q in base["qids"]]
            b = [rec[s]["ap"][q] for q in base["qids"]]
            pt, pw = paired_tests(a, b)
            cells += [f"{rec[s]['MAP']:.4f}", f"{rec[s]['MAP'] - base['MAP']:+.4f}", f"{pt:.2g}", f"{pw:.2g}"]
        ab_rows.append(cells)
    head = [f"**Lexis best** ({best.describe()})",
            f"{evals[best_name]['train']['MAP']:.4f}", "", "", "",
            f"{evals[best_name]['test']['MAP']:.4f}", "", "", ""]
    md_ab = md_table(["Variant", "train MAP", "ΔMAP", "p (t)", "p (Wilcoxon)",
                      "test MAP", "ΔMAP", "p (t)", "p (Wilcoxon)"], [head] + ab_rows)

    # ------------------------------------------------------------ sensitivity plot
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.2))
    gs = [g for g in GRID["g"] if g is not None]
    same = [(m, c) for m, c in grid if (c.stem, c.stop, c.doc_tf, c.doc_idf, c.q_tf, c.q_idf) ==
            (best.stem, best.stop, best.doc_tf, best.doc_idf, best.q_tf, best.q_idf)]
    for slope in GRID["slope"]:
        ys = [max(m for m, c in same if c.zones and c.g == g and c.slope == slope) for g in gs]
        axes[0].plot(gs, ys, marker=".", label=f"slope {slope:g}" + (" (cosine)" if slope == 1 else ""))
    axes[0].set(xlabel="title weight g", ylabel="train MAP", title="Zone weight × pivot slope")
    axes[0].legend(fontsize=6.5)
    smart = {}
    for m, c in grid:
        key = f"{c.doc_tf}{c.doc_idf}.{c.q_tf}{c.q_idf}"
        smart[key] = max(smart.get(key, 0), m)
    keys = sorted(smart, key=smart.get)
    axes[1].barh(keys, [smart[k] for k in keys], color="#4a7bb7")
    axes[1].set(xlabel="best train MAP", title="SMART weighting (doc.query, best over other params)")
    axes[1].tick_params(axis="y", labelsize=6.5)
    fig.tight_layout()
    fig.savefig(f"{FIGS}/tuning.png")

    top = [[f"{m:.4f}", c.describe()] for m, c in grid[:10]]
    md_top = md_table(["train MAP", "configuration"], top, ["---:", "---"])

    save_json("ranking", {"best": best.__dict__, "systems": {k: {s: {m: v[s][m] for m in ("MAP", "P@10", "NDCG@10", "pr11")}
                                                                    for s in v} for k, v in evals.items()},
                          "per_query_ap_test": {k: v["test"]["ap"] for k, v in evals.items()},
                          "ablations": {k: {s: v[s]["MAP"] for s in v} for k, v in ab_data.items()},
                          "grid_size": len(grid)})
    save_md("ranking_main", md)
    save_md("ranking_pr11", md_pr)
    save_md("ranking_ablations", md_ab)
    save_md("ranking_top_configs", md_top)
    save_md("grid_size", f"{len(grid):,}")
    free_rec = {s: evaluate(run_cfg(colls, free_best, qs), qrels) for s, qs in (("train", train), ("test", test))}
    save_md("ranking_unconstrained",
            f"Unconstrained grid optimum: {free_best.describe()} with train MAP {free_map:.4f} "
            f"(test MAP {free_rec['test']['MAP']:.4f}); best zone-weighted configuration: train MAP {best_map:.4f}.")


if __name__ == "__main__":
    main()
