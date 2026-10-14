"""Tolerant retrieval: k-gram index for wildcards and spelling correction (IIR Ch. 3).

Wildcards
    The pattern, with boundary markers ($econom*$), is split at '*'.  The
    k-grams of every fragment are intersected in the k-gram index, and the
    candidates are then post-filtered with an anchored regex.  The post-filter
    is required because k-grams lose order: red* would also match "retired".
    Any number of '*' works with the same procedure.  A k-gram index costs
    about one posting per character of vocabulary, against roughly one rotation
    per character for a permuterm index, and multi-* patterns would need an
    extra filter in permuterm anyway.

Spelling correction (isolated-term)
    Candidates are vocabulary terms whose k-gram Jaccard overlap with the query
    term is at least ``jaccard``.  They are ranked by Damerau-Levenshtein (OSA)
    distance, then by collection frequency (the more common word wins a tie,
    IIR Sec. 3.3.2), then alphabetically.
"""

import re
from collections import Counter


def kgrams(term, k):
    t = f"${term}$"
    return {t[i : i + k] for i in range(len(t) - k + 1)}


class KGramIndex:
    def __init__(self, vocabulary, k=2, freq=None):
        self.k = k
        self.vocab = list(vocabulary)
        self.vocab_set = set(self.vocab)
        self.freq = freq or (lambda t: 0)
        idx = {}
        for i, t in enumerate(self.vocab):
            for g in kgrams(t, k):
                idx.setdefault(g, []).append(i)
        self.index = idx
        self.ngrams = [len(kgrams(t, k)) for t in self.vocab]

    def size(self):
        return {"grams": len(self.index), "postings": sum(map(len, self.index.values()))}

    # ------------------------------------------------------------------ wildcard
    def wildcard(self, pattern):
        """All vocabulary terms matching pattern (one or more '*')."""
        if "*" not in pattern:
            return [pattern] if pattern in self.vocab_set else []
        k = self.k
        frags = f"${pattern}$".split("*")
        grams = set()
        for f in frags:
            for i in range(len(f) - k + 1):
                grams.add(f[i : i + k])
        if grams:
            lists = sorted((self.index.get(g, []) for g in grams), key=len)
            cand = set(lists[0])
            for l in lists[1:]:
                cand.intersection_update(l)
                if not cand:
                    break
            cand = [self.vocab[i] for i in sorted(cand)]
        else:
            cand = self.vocab
        rx = re.compile("^" + ".*".join(re.escape(p) for p in pattern.split("*")) + "$")
        return [t for t in cand if rx.match(t)]

    # ---------------------------------------------------------------- spelling
    def candidates(self, term, jaccard=0.2):
        g = kgrams(term, self.k)
        overlap = Counter()
        for gram in g:
            for i in self.index.get(gram, ()):
                overlap[i] += 1
        out = []
        ng = len(g)
        for i, shared in overlap.items():
            j = shared / (ng + self.ngrams[i] - shared)
            if j >= jaccard:
                out.append(self.vocab[i])
        return out

    def suggest(self, term, n=1, max_dist=3, jaccard=0.2):
        """Best corrections for a term not in the vocabulary ([] if it is in it or nothing is close)."""
        if term in self.vocab_set:
            return []
        cands = self.candidates(term, jaccard)
        scored = []
        for c in cands:
            if abs(len(c) - len(term)) > max_dist:
                continue
            d = osa_distance(term, c, max_dist)
            if d <= max_dist:
                scored.append((d, -self.freq(c), c))
        scored.sort()
        return [c for _, _, c in scored[:n]]


def osa_distance(a, b, limit=None):
    """Optimal-string-alignment (restricted Damerau-Levenshtein) distance with an early exit."""
    if a == b:
        return 0
    la, lb = len(a), len(b)
    prev2 = None
    prev = list(range(lb + 1))
    for i in range(1, la + 1):
        cur = [i] + [0] * lb
        ai = a[i - 1]
        row_min = cur[0]
        for j in range(1, lb + 1):
            cost = 0 if ai == b[j - 1] else 1
            v = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + cost)
            if i > 1 and j > 1 and ai == b[j - 2] and a[i - 2] == b[j - 1]:
                v = min(v, prev2[j - 2] + 1)
            cur[j] = v
            if v < row_min:
                row_min = v
        if limit is not None and row_min > limit:
            return limit + 1
        prev2, prev = prev, cur
    return prev[lb]
