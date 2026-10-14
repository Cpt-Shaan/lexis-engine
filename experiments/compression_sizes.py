"""Compression: postings size table (uncompressed / gaps / VB / gamma) and dictionary compression."""

import math
import os

import numpy as np

from common import CORPUS, INDEX, ensure_index, fmt_bytes, md_table, save_json, save_md
from lexis.codec import gamma_bits, gamma_decode, gamma_encode, packed_bits, vb_encode
from lexis.index import Index


def list_sizes(gaps):
    """Bytes of one postings list under each method (each list padded to a whole byte)."""
    return {
        "uncompressed": 4 * len(gaps),
        "gaps": math.ceil(packed_bits(gaps) / 8),
        "vb": len(vb_encode(gaps)),
        "gamma": math.ceil(sum(gamma_bits(g) for g in gaps) / 8),
    }


def main():
    ensure_index()
    ix = Index(INDEX)
    m = ix.main
    df = np.array(m.df)
    n = len(m.terms)
    order = np.lexsort((np.arange(n), -df))          # most frequent first (by df)
    top100 = set(order[:100].tolist())
    rare10k = set(order[-10_000:].tolist())
    groups = {"all": set(range(n)), "top100": top100, "rare10k": rare10k}
    METHODS = ["uncompressed", "gaps", "vb", "gamma"]
    acc = {kind: {g: dict.fromkeys(METHODS, 0) for g in groups} for kind in ("docids", "positions")}
    counts = {kind: dict.fromkeys(groups, 0) for kind in ("docids", "positions")}
    check_gaps = []
    for i, t in enumerate(m.terms):
        pl = m.positional(t)
        docs = [d for d, _ in pl]
        dgaps = [docs[0]] + [b - a for a, b in zip(docs, docs[1:])]
        pgaps = []
        for _, pos in pl:
            # first position stored +1 so every gap is >= 1 (gamma cannot code 0)
            pgaps.append(pos[0] + 1)
            pgaps.extend(b - a for a, b in zip(pos, pos[1:]))
        sd, sp_ = list_sizes(dgaps), list_sizes(pgaps)
        if i % 97 == 0:
            check_gaps.extend(dgaps)
        for g, members in groups.items():
            if i in members:
                for meth in METHODS:
                    acc["docids"][g][meth] += sd[meth]
                    acc["positions"][g][meth] += sp_[meth]
                counts["docids"][g] += len(dgaps)
                counts["positions"][g] += len(pgaps)
    # sanity: the gamma coder really produces the size we account for
    data, nbits = gamma_encode(check_gaps)
    assert gamma_decode(data, nbits) == check_gaps
    assert nbits == sum(gamma_bits(g) for g in check_gaps)

    names = {"uncompressed": "Uncompressed (32-bit docIDs)", "gaps": "Gaps, fixed width per list (bit-packed)",
             "vb": "Gaps, variable byte", "gamma": "Gaps, Elias γ"}
    md = ""
    for kind, label in (("docids", "docID postings"), ("positions", "position postings")):
        rows = []
        for meth in METHODS:
            row = [names[meth].replace("docIDs", "docIDs" if kind == "docids" else "positions")]
            for g in ("all", "top100", "rare10k"):
                b = acc[kind][g][meth]
                u = acc[kind][g]["uncompressed"]
                row += [fmt_bytes(b), f"{8 * b / counts[kind][g]:.2f}", f"{u / b:.2f}×"]
            rows.append(row)
        hdr = ["Method", "All terms", "bits/entry", "ratio", "Top-100 terms", "bits/entry", "ratio",
               "Rarest 10,000", "bits/entry", "ratio"]
        md += f"**{label}**  (all: {counts[kind]['all']:,} entries; top-100: {counts[kind]['top100']:,}; " \
              f"rarest 10,000: {counts[kind]['rare10k']:,})\n\n" + md_table(hdr, rows) + "\n\n"
    md += ("ratio = uncompressed size / compressed size, so higher is better.  The top-100 vs rarest-10,000 "
           "ratio for each method is in the summary below.\n\n")
    summ = []
    for meth in ["gaps", "vb", "gamma"]:
        r_top = acc["docids"]["top100"]["uncompressed"] / acc["docids"]["top100"][meth]
        r_rare = acc["docids"]["rare10k"]["uncompressed"] / acc["docids"]["rare10k"][meth]
        summ.append([names[meth], f"{r_top:.2f}×", f"{r_rare:.2f}×", f"{r_top / r_rare:.2f}"])
    md += md_table(["docID postings", "ratio top-100", "ratio rarest-10k", "top-100 ÷ rarest-10k"], summ)

    # ---------------------------------------------------------------- dictionary
    terms = m.terms
    nt = len(terms)
    string_len = sum(len(t.encode()) for t in terms)
    k = 4
    meta = m.meta["bytes"]["dict_parts"]
    fc_terms = meta["terms"]
    rows = [
        ["Fixed-width array (20-byte term, 4-byte df, 4-byte postings ptr)", fmt_bytes(28 * nt), "28.0"],
        ["Dictionary as a string (+ 3-byte term ptr, 4 df, 4 ptr)", fmt_bytes(string_len + 11 * nt),
         f"{(string_len + 11 * nt) / nt:.1f}"],
        [f"Blocked string, k={k} (+1 length byte/term, 3-byte ptr/block)",
         fmt_bytes(string_len + nt + 3 * math.ceil(nt / k) + 8 * nt),
         f"{(string_len + nt + 3 * math.ceil(nt / k) + 8 * nt) / nt:.1f}"],
        [f"Blocked k={k} + front coding (Lexis term section, measured) + 4 df + 4 ptr",
         fmt_bytes(fc_terms + 3 * math.ceil(nt / k) + 8 * nt), f"{(fc_terms + 3 * math.ceil(nt / k) + 8 * nt) / nt:.1f}"],
        ["**Lexis dict.bin as stored** (front-coded blocks + VB df, cf and postings lengths for 2 files)",
         fmt_bytes(meta["terms"] + meta["block_pointers"] + meta["metadata"]),
         f"{(meta['terms'] + meta['block_pointers'] + meta['metadata']) / nt:.1f}"],
    ]
    md_dict = md_table(["Dictionary representation", "Size", "bytes/term"], rows, ["---", "---:", "---:"])
    md_dict += (f"\n\n{nt:,} terms, average length {string_len / nt:.2f} characters; the front-coded "
                f"term section is {fc_terms:,} bytes against {string_len + nt:,} for the blocked string "
                f"({100 * (1 - fc_terms / (string_len + nt)):.0f}% smaller). Lexis stores per term: df, cf, "
                f"byte length of its positional list and byte length of its zone-tf list, all VB-coded; "
                f"offsets are prefix sums of lengths.")

    files = {f: os.path.getsize(os.path.join(INDEX, f)) for f in
             ("dict.bin", "postings_pos.bin", "postings_doc.bin", "docs.tsv")}
    corpus_bytes = sum(os.path.getsize(os.path.join(CORPUS, f)) for f in os.listdir(CORPUS))
    md_files = md_table(["File", "Size"], [[f, fmt_bytes(b)] for f, b in files.items()] +
                        [["**total index**", fmt_bytes(sum(files.values()))],
                         ["corpus text (for reference)", fmt_bytes(corpus_bytes)]], ["---", "---:"])
    save_json("compression", {"postings": acc, "counts": counts, "files": files, "corpus_bytes": corpus_bytes})
    save_md("compression_postings", md)
    save_md("compression_dictionary", md_dict)
    save_md("compression_files", md_files)


if __name__ == "__main__":
    main()
