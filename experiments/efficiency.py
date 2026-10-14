"""Efficiency: Chapter 7 pruning vs exhaustive scoring (overlap@10, docs scored, latency, quality)."""

import json
import os
import time

import numpy as np

from common import DATA, FIGS, INDEX, ensure_index, md_table, plt, save_json, save_md
from lexis.corpus import load_qrels, load_queries
from lexis.evaluation import ndcg_at, precision_at
from lexis.index import Index
from lexis.ranking import Collection, RankConfig, Scorer

METHODS = [
    ("Exhaustive TAAT (all postings of all query terms)", "exhaustive", {}),
    ("Index elimination: idf ≥ 0.5, ≥ 50% of terms", "index_elimination", {"idf_min": 0.5, "min_frac": 0.5}),
    ("Index elimination: idf ≥ 1.0, ≥ 50% of terms", "index_elimination", {"idf_min": 1.0, "min_frac": 0.5}),
    ("Index elimination: idf ≥ 0.5, all terms", "index_elimination", {"idf_min": 0.5, "min_frac": 1.0}),
    ("Champion lists r = 20", "champion", {"r": 20}),
    ("Champion lists r = 50", "champion", {"r": 50}),
    ("Champion lists r = 100", "champion", {"r": 100}),
    ("Cluster pruning √N leaders, b₁ = 1, b₂ = 1", "cluster", {"b1": 1, "b2": 1}),
    ("Cluster pruning √N leaders, b₁ = 1, b₂ = 5", "cluster", {"b1": 1, "b2": 5}),
    ("Cluster pruning √N leaders, b₁ = 3, b₂ = 3", "cluster", {"b1": 3, "b2": 3}),
    ("Cluster pruning √N leaders, b₁ = 3, b₂ = 10", "cluster", {"b1": 3, "b2": 10}),
]


def main():
    ensure_index()
    ix = Index(INDEX)
    cfg = RankConfig(**json.load(open(os.path.join(INDEX, "rank_config.json"))))
    coll = Collection(ix, cfg.stem, cfg.stop)
    t0 = time.perf_counter()
    sc = Scorer(coll, cfg, champion_r=50)
    t_champ = time.perf_counter() - t0
    t0 = time.perf_counter()
    sc.build_clusters(1)
    t_clu = time.perf_counter() - t0
    queries = load_queries(os.path.join(DATA, "queries_train.tsv")) + load_queries(os.path.join(DATA, "queries_test.tsv"))
    qrels = load_qrels(os.path.join(DATA, "qrels.tsv"))

    # warm-up
    for _, q in queries[:5]:
        sc.search(q)
    exact = {}
    rows, data = [], []
    for label, method, kw in METHODS:
        kw = dict(kw)
        if method == "champion":
            sc.build_champions(kw.pop("r"))
        ov, scored, lat, p10, nd = [], [], [], [], []
        for qid, q in queries:
            res, n, dt = sc.timed(q, 10, method, repeat=20, **kw)
            docs = [d for d, _ in res]
            if method == "exhaustive":
                exact[qid] = docs
            ov.append(len(set(docs) & set(exact[qid])) / max(1, len(exact[qid])))
            scored.append(n)
            lat.append(dt * 1000)
            p10.append(precision_at(docs, qrels.get(qid, {})))
            nd.append(ndcg_at(docs, qrels.get(qid, {})))
        rec = {"method": label, "overlap@10": np.mean(ov), "docs_scored": np.mean(scored),
               "latency_ms": np.mean(lat), "latency_p50": np.median(lat), "P@10": np.mean(p10),
               "NDCG@10": np.mean(nd)}
        data.append(rec)
        base = data[0]
        rows.append([label, f"{rec['overlap@10']:.3f}", f"{rec['docs_scored']:,.0f}",
                     f"{100 * rec['docs_scored'] / base['docs_scored']:.1f}%",
                     f"{rec['latency_ms']:.3f}", f"{base['latency_ms'] / rec['latency_ms']:.2f}×",
                     f"{rec['P@10']:.3f}", f"{rec['NDCG@10']:.3f}"])
        print(rows[-1], flush=True)
    md = md_table(["Method", "overlap@10", "docs scored / query", "vs exhaustive", "latency ms",
                   "speedup", "P@10", "NDCG@10"], rows)
    md += (f"\n\nAll 80 queries (train + test), configuration: {cfg.describe()}; collection N = {coll.N:,}. "
           f"Latency is the best of 20 runs per query, then averaged over queries. "
           f"Offline cost: champion lists (r=50) {t_champ:.2f}s, clustering ({sc.n_leaders} leaders) {t_clu:.2f}s.")
    save_md("efficiency", md)
    save_json("efficiency", data)

    fig, ax = plt.subplots(figsize=(5.4, 3.4))
    for rec, mk in zip(data, "o^^^sssDDDD"):
        ax.scatter(rec["docs_scored"], rec["overlap@10"], marker=mk, s=28)
        ax.annotate(rec["method"].split(":")[0].replace("Index elimination", "IE").replace("Champion lists", "CL")
                    .replace("Cluster pruning √N leaders,", "CP").replace("Exhaustive TAAT (all postings of all query terms)", "exhaustive")
                    + ("" if ":" not in rec["method"] else " " + rec["method"].split(":")[1].strip()),
                    (rec["docs_scored"], rec["overlap@10"]), fontsize=5.5,
                    xytext=(3, -8) if "r = 100" in rec["method"] else (3, 2), textcoords="offset points")
    ax.set_xscale("log")
    ax.set(xlabel="documents scored per query (log)", ylabel="overlap@10 with exhaustive",
           title="Pruning: work vs fidelity")
    fig.tight_layout()
    fig.savefig(f"{FIGS}/pruning.png")


if __name__ == "__main__":
    main()
