"""Heaps' and Zipf's law fits (IIR Sec. 5.1)."""

from collections import Counter

import numpy as np

from common import CORPUS, FIGS, md_table, plt, save_json, save_md
from lexis.corpus import stream_documents
from lexis.text import Analyzer, tokenize

RCV1 = {"k": 44, "b": 0.49, "T": 1_000_020, "M": 38_323}   # IIR Sec. 5.1.1, Fig. 5.1


def fit_line(x, y):
    A = np.vstack([x, np.ones_like(x)]).T
    (slope, icpt), res, *_ = np.linalg.lstsq(A, y, rcond=None)
    yhat = slope * x + icpt
    r2 = 1 - ((y - yhat) ** 2).sum() / ((y - y.mean()) ** 2).sum()
    return slope, icpt, r2


def main():
    settings = {
        "case-folded": Analyzer(stem=False, stop=False),
        "case-folded + stemmed": Analyzer(stem=True, stop=False),
        "case-folded + stemmed + 25 stop words": Analyzer(stem=True, stop=True),
    }
    seen = {s: set() for s in settings}
    cf = {s: Counter() for s in settings}
    T = {s: 0 for s in settings}
    curve = {s: ([], []) for s in settings}
    for doc in stream_documents(CORPUS):
        toks = tokenize(doc.title) + tokenize(doc.body)
        for s, an in settings.items():
            terms = [n for n in (an.normalize(t) for t in toks) if n is not None]
            seen[s].update(terms)
            cf[s].update(terms)
            T[s] += len(terms)
            curve[s][0].append(T[s])
            curve[s][1].append(len(seen[s]))

    out = {}
    rows = []
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.6))
    for s in settings:
        t, m = np.array(curve[s][0], float), np.array(curve[s][1], float)
        b, logk, r2 = fit_line(np.log10(t), np.log10(m))
        k = 10 ** logk
        freqs = np.array(sorted(cf[s].values(), reverse=True), float)
        ranks = np.arange(1, len(freqs) + 1, dtype=float)
        a_all, c_all, r2z = fit_line(np.log10(ranks), np.log10(freqs))
        # log-binned fit: 60 ranks equally spaced in log(rank), so the long tail of
        # hapax legomena does not dominate the least-squares fit
        pts = np.unique(np.round(np.logspace(0, np.log10(len(freqs)), 60)).astype(int)) - 1
        a_mid, c_mid, r2m = fit_line(np.log10(ranks[pts]), np.log10(freqs[pts]))
        head = ranks <= 1000
        a_head, c_head, r2h = fit_line(np.log10(ranks[head]), np.log10(freqs[head]))
        out[s] = {"T": int(T[s]), "M": len(seen[s]), "heaps_k": k, "heaps_b": b, "heaps_r2": r2,
                  "zipf_alpha_all": -a_all, "zipf_logc_all": c_all, "zipf_r2_all": r2z,
                  "zipf_alpha_logbin": -a_mid, "zipf_logc_logbin": c_mid, "zipf_r2_logbin": r2m,
                  "zipf_alpha_head1000": -a_head, "zipf_r2_head1000": r2h, "hapax": int((freqs == 1).sum()),
                  "cf1": int(freqs[0]), "top10": cf[s].most_common(10),
                  "M_at_rcv1_T": k * RCV1["T"] ** b}
        rows.append([s, f"{T[s]:,}", f"{len(seen[s]):,}", f"{k:.1f}", f"{b:.3f}", f"{r2:.4f}",
                     f"{-a_mid:.3f}", f"{r2m:.3f}", f"{-a_head:.3f}", f"{-a_all:.3f}"])
        axes[0].loglog(t, m, lw=1.2, label=f"{s}")
        axes[1].loglog(ranks, freqs, lw=1.2, label=s)
        if s == "case-folded + stemmed":
            axes[0].loglog(t, k * t ** b, "k--", lw=.8, label=f"fit: {k:.1f}·T^{b:.3f}")
            axes[0].loglog(t, RCV1["k"] * t ** RCV1["b"], ":", color="gray", lw=1,
                           label="RCV1: 44·T^0.49")
            axes[1].loglog(ranks, 10 ** c_mid * ranks ** a_mid, "k--", lw=.8,
                           label=f"log-binned fit: α={-a_mid:.2f}")
            axes[1].loglog(ranks, freqs[0] / ranks, ":", color="gray", lw=1, label="ideal Zipf cf₁/i")
    axes[0].set(xlabel="tokens T", ylabel="vocabulary M", title="Heaps' law")
    axes[1].set(xlabel="rank i", ylabel="collection frequency cfᵢ", title="Zipf's law")
    for ax in axes:
        ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(f"{FIGS}/heaps_zipf.png")
    save_json("vocabulary", out)

    md = md_table(["Setting", "T (tokens)", "M (terms)", "Heaps k", "Heaps b", "R²",
                   "Zipf α (log-binned)", "R²", "Zipf α (ranks 1–1000)", "Zipf α (every rank)"], rows)
    md += "\n\n" + md_table(["", "k", "b"], [["Reuters-RCV1 (IIR Fig. 5.1)", RCV1["k"], RCV1["b"]]])
    s = out["case-folded + stemmed"]
    md += (f"\n\nWith our stemmed fit, a collection the size of the RCV1 sample (T = 1,000,020) "
           f"would have M ≈ {s['M_at_rcv1_T']:,.0f} terms; the textbook reports 38,323 actual "
           f"and 38,365 predicted.")
    save_md("vocabulary", md)


if __name__ == "__main__":
    main()
