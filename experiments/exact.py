"""Exact retrieval: skip pointers, positional vs plain index size, Boolean test suite, tolerant retrieval."""

import math
import os
import time

import numpy as np

from common import (CORPUS, DATA, FIGS, INDEX, ensure_index, fmt_bytes, md_table, plt, save_json,
                    save_md)
from lexis.boolean import BooleanEngine, Counter, intersect
from lexis.bruteforce import BruteForce
from lexis.codec import vb_encode
from lexis.corpus import load_boolean_tests, load_misspellings
from lexis.index import Index
from lexis.tolerant import KGramIndex

PAIRS = [("bank", "rate"), ("oil", "opec"), ("trade", "deficit"), ("wheat", "export"),
         ("dollar", "yen"), ("stock", "split"), ("interest", "rates"), ("gold", "silver"),
         ("said", "tanker"), ("company", "coffee"), ("mln", "dlrs"), ("the", "cocoa")]
SPACINGS = ["sqrt", 4, 16, 64]


def skip_experiment(ix):
    rows, data = [], []
    for a, b in PAIRS:
        p1, p2 = ix.docs_for(a), ix.docs_for(b)
        base = Counter()
        ref = intersect(p1, p2, None, base)
        rec = {"pair": f"{a} AND {b}", "len1": len(p1), "len2": len(p2), "result": len(ref),
               "no_skips": base.n}
        cells = []
        for s in SPACINGS:
            c = Counter()
            assert intersect(p1, p2, s, c) == ref
            rec[str(s)] = c.n
            cells.append(f"{c.n:,} ({100 * (1 - c.n / base.n):+.0f}%)".replace("+", "−") if c.n <= base.n
                         else f"{c.n:,} (+{100 * (c.n / base.n - 1):.0f}%)")
        data.append(rec)
        rows.append([rec["pair"], f"{len(p1):,}", f"{len(p2):,}", f"{len(ref):,}", f"{base.n:,}"] + cells)
    tot = {k: sum(r[k] for r in data) for k in ["no_skips"] + [str(s) for s in SPACINGS]}
    rows.append(["**total**", "", "", "", f"**{tot['no_skips']:,}**"] +
                [f"**{tot[str(s)]:,} (−{100 * (1 - tot[str(s)] / tot['no_skips']):.0f}%)**" for s in SPACINGS])
    md = md_table(["Query", "|p1|", "|p2|", "|result|", "no skips"] +
                  ["skip = √L" if s == "sqrt" else f"skip = {s}" for s in SPACINGS], rows)

    # comparisons vs spacing curve for a few pairs
    fig, ax = plt.subplots(figsize=(5.2, 3.2))
    spacings = [1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64, 96, 128, 256]
    for a, b in PAIRS[:6] + [PAIRS[8]]:
        p1, p2 = ix.docs_for(a), ix.docs_for(b)
        base = Counter()
        intersect(p1, p2, None, base)
        ys = []
        for s in spacings:
            c = Counter()
            intersect(p1, p2, s if s > 1 else None, c)
            ys.append(c.n / base.n)
        ax.semilogx(spacings, ys, marker=".", lw=1, label=f"{a} ∧ {b}")
    ax.axhline(1, color="gray", lw=.8)
    ax.set(xlabel="skip spacing s", ylabel="comparisons / comparisons without skips",
           title="Skip pointers: effect of spacing")
    ax.legend(fontsize=6.5)
    fig.tight_layout()
    fig.savefig(f"{FIGS}/skips.png")
    return md, data


def size_experiment(ix):
    """Positional vs non-positional postings (same VB coding; also uncompressed 32-bit)."""
    m = ix.main
    plain_vb = plain_tf_vb = 0
    plain_raw = plain_tf_raw = pos_raw = 0
    for i, t in enumerate(m.terms):
        pl = m.positional(t)
        docs = [d for d, _ in pl]
        gaps = [docs[0]] + [b - a for a, b in zip(docs, docs[1:])]
        plain_vb += len(vb_encode(gaps))
        plain_tf_vb += len(vb_encode(gaps)) + len(vb_encode(len(p) for _, p in pl))
        plain_raw += 4 * len(docs)
        plain_tf_raw += 8 * len(docs)
        pos_raw += 8 * len(docs) + 4 * sum(len(p) for _, p in pl)
    pos_vb = os.path.getsize(os.path.join(INDEX, "postings_pos.bin"))
    zone_vb = os.path.getsize(os.path.join(INDEX, "postings_doc.bin"))
    rows = [
        ["docIDs only (Boolean index)", fmt_bytes(plain_raw), fmt_bytes(plain_vb), "1.00×"],
        ["docIDs + tf (plain index)", fmt_bytes(plain_tf_raw), fmt_bytes(plain_tf_vb),
         f"{plain_tf_vb / plain_vb:.2f}×"],
        ["docIDs + tf_title + tf_body (Lexis zone index)", fmt_bytes(12 * m.meta['num_postings']),
         fmt_bytes(zone_vb), f"{zone_vb / plain_vb:.2f}×"],
        ["docIDs + tf + positions (Lexis positional index)", fmt_bytes(pos_raw), fmt_bytes(pos_vb),
         f"{pos_vb / plain_vb:.2f}×"],
    ]
    md = md_table(["Postings", "uncompressed (32-bit ints)", "VB-coded gaps", "VB size vs docIDs only"], rows)
    md += (f"\n\nPositional / plain (docID+tf), VB-coded: **{pos_vb / plain_tf_vb:.2f}×**; "
           f"uncompressed: **{pos_raw / plain_tf_raw:.2f}×**. "
           f"Average tokens per document = {m.meta['num_tokens'] / m.meta['num_docs']:.0f}, "
           f"average tf per posting = {m.meta['num_tokens'] / m.meta['num_postings']:.2f}.")
    return md, {"plain_raw": plain_raw, "plain_vb": plain_vb, "plain_tf_raw": plain_tf_raw,
                "plain_tf_vb": plain_tf_vb, "pos_raw": pos_raw, "pos_vb": pos_vb, "zone_vb": zone_vb}


def boolean_suite(ix, kg):
    eng = BooleanEngine(ix, kg)
    bf = BruteForce(CORPUS)
    rows, data = [], []
    for t in load_boolean_tests(os.path.join(DATA, "boolean_tests.tsv")):
        t0 = time.perf_counter()
        for _ in range(5):
            got = eng.search(t["query"])
        ti = (time.perf_counter() - t0) / 5
        t0 = time.perf_counter()
        ref = bf.search(t["query"])
        tb = time.perf_counter() - t0
        ok = got == t["expected"] and ref == t["expected"]
        data.append({"id": t["id"], "query": t["query"], "type": t["type"], "n": len(got), "expected": t["n"],
                     "pass": ok, "index_ms": 1000 * ti, "brute_ms": 1000 * tb})
        rows.append([t["id"], f"`{t['query']}`", t["type"], t["n"], len(got), "✓" if ok else "✗",
                     f"{1000 * ti:.2f}", f"{1000 * tb:.0f}"])
    n_ok = sum(d["pass"] for d in data)
    md = md_table(["id", "query", "type", "expected", "Lexis", "match", "Lexis ms", "brute-force ms"], rows,
                  ["---", "---", "---", "---:", "---:", ":---:", "---:", "---:"])
    md += (f"\n\n**{n_ok}/{len(data)}** result sets are identical to the expected sets and to the "
           f"brute-force scan. Mean latency: Lexis {np.mean([d['index_ms'] for d in data]):.2f} ms, "
           f"brute force {np.mean([d['brute_ms'] for d in data]):.0f} ms.")
    return md, data


def spelling(ix):
    rows_all = load_misspellings(os.path.join(DATA, "misspellings.tsv"))
    out = {}
    lines = []
    for k in (2, 3):
        kg = KGramIndex(ix.vocabulary, k, ix.collection_frequency)
        t0 = time.perf_counter()
        res = [(bad, good[0], kg.suggest(bad, n=3)) for bad, good in rows_all]
        dt = (time.perf_counter() - t0) / len(rows_all)
        top1 = sum(s[:1] == [g] for _, g, s in res)
        top3 = sum(g in s for _, g, s in res)
        one = [r for r in res[:50]]
        two = [r for r in res[50:]]
        acc1 = sum(s[:1] == [g] for _, g, s in one)
        acc2 = sum(s[:1] == [g] for _, g, s in two)
        out[k] = {"top1": top1, "top3": top3, "ms": dt * 1000, "fails": [r for r in res if r[2][:1] != [r[1]]],
                  "one_typo": acc1, "two_typo": acc2, "size": kg.size()}
        lines.append([f"{k}-gram", f"{top1}/60", f"{acc1}/50", f"{acc2}/10", f"{top3}/60", f"{dt * 1000:.1f}",
                      f"{kg.size()['grams']:,}", f"{kg.size()['postings']:,}"])
    md = md_table(["k-gram index", "top-1", "1 typo", "2 typos", "top-3", "ms / term", "k-grams", "postings"], lines)
    fails = out[2]["fails"]
    md += "\n\nMisses (bigram index): " + ", ".join(f"*{b}* → {s[0] if s else '∅'} (intended *{g}*)"
                                                    for b, g, s in fails)
    return md, out


def wildcard_stats(kg):
    rows = []
    for pat in ["econom*", "agricultur*", "*port", "pet*chem*", "*ization", "s*ing*s"]:
        t0 = time.perf_counter()
        for _ in range(20):
            res = kg.wildcard(pat)
        dt = (time.perf_counter() - t0) / 20
        rows.append([f"`{pat}`", len(res), ", ".join(res[:5]) + (" …" if len(res) > 5 else ""), f"{dt * 1000:.2f}"])
    return md_table(["pattern", "# terms", "first matches", "ms"], rows, ["---", "---:", "---", "---:"])


def main():
    ensure_index()
    ix = Index(INDEX)
    kg = KGramIndex(ix.vocabulary, 2, ix.collection_frequency)
    md1, d1 = skip_experiment(ix)
    md2, d2 = size_experiment(ix)
    md3, d3 = boolean_suite(ix, kg)
    md4, d4 = spelling(ix)
    md5 = wildcard_stats(kg)
    save_json("exact", {"skips": d1, "sizes": d2, "boolean": d3, "spelling": d4})
    save_md("exact_skips", md1)
    save_md("exact_sizes", md2)
    save_md("exact_boolean", md3)
    save_md("exact_spelling", md4)
    save_md("exact_wildcard", md5)


if __name__ == "__main__":
    main()
