"""Ranked retrieval: cosine similarity over tf-idf vectors with title and body zones.

Scoring model
-------------
Every document has two vectors, one per zone (title, body), each weighted with
the SMART scheme ``ddd`` and normalized on its own.  The query vector uses
``qqq``.  The score is a weighted-zone combination of two cosines (IIR Sec. 6.1, 6.3):

    score(q, d) = g * cos(q, d_title) + (1 - g) * cos(q, d_body)

Because this is linear in the document weights, it equals q . W[d] with the
precomputed combined matrix W = g*W_title + (1-g)*W_body.  W is held both
term-major (CSC: the postings lists, for term-at-a-time scoring) and
doc-major (CSR: for scoring candidate sets after pruning).

    tf   n: tf     l: 1 + log10 tf     b: 1 if tf > 0
    idf  n: 1      t: log10 N/df       p: max(0, log10 (N-df)/df)
    norm n: none   c: cosine           pivoted cosine: (1-s)*avg_norm + s*norm (IIR Sec. 6.4.4)

Top-k retrieval never scores every document.  Term-at-a-time accumulation
touches only documents in the postings of query terms, and the top 10 are
selected with a partial sort (argpartition, O(n)) instead of a full sort.
The pruning methods in ``Scorer`` cut the scored set further.
"""

import time
from dataclasses import dataclass, replace

import numpy as np
import scipy.sparse as sp

from .text import Analyzer


@dataclass(frozen=True)
class RankConfig:
    stem: bool = True
    stop: bool = True
    doc_tf: str = "l"
    doc_idf: str = "n"
    doc_norm: str = "c"
    slope: float = 1.0          # pivoted normalization slope; 1.0 = plain cosine
    q_tf: str = "l"
    q_idf: str = "t"
    zones: bool = True
    g: float = 0.3              # title-zone weight (only when zones=True)

    def smart(self):
        n = self.doc_norm if self.slope == 1.0 or self.doc_norm == "n" else f"u{self.slope:g}"
        return f"{self.doc_tf}{self.doc_idf}{n}.{self.q_tf}{self.q_idf}c"

    def describe(self):
        z = f"zones g={self.g:g}" if self.zones else "single zone"
        return (f"{self.smart()}, {z}, stem={'on' if self.stem else 'off'}, "
                f"stop={'on' if self.stop else 'off'}")

    def with_(self, **kw):
        return replace(self, **kw)


def _tf(mat, scheme):
    m = mat.copy().astype(np.float64)
    if scheme == "l":
        m.data = 1.0 + np.log10(m.data)
    elif scheme == "b":
        m.data[:] = 1.0
    elif scheme != "n":
        raise ValueError(scheme)
    return m


def _tf_vec(v, scheme):
    v = v.astype(np.float64)
    if scheme == "l":
        return 1.0 + np.log10(v)
    if scheme == "b":
        return np.ones_like(v)
    return v


class Collection:
    """Zone term-frequency matrices for one analyzer setting (stem/stop)."""

    def __init__(self, index, stem=True, stop=True):
        self.analyzer = Analyzer(stem, stop)
        terms, ti, docs, tt, tb = index.zone_postings()
        norm_terms = [self.analyzer.normalize(t) for t in terms]
        self.vocab = sorted({t for t in norm_terms if t is not None})
        self.tid = {t: i for i, t in enumerate(self.vocab)}
        col_of = np.array([self.tid.get(t, -1) if t is not None else -1 for t in norm_terms], dtype=np.int64)
        self.doc_ids = np.array(index.doc_ids, dtype=np.int64)
        self.N = len(self.doc_ids)
        col = col_of[ti]
        keep = (col >= 0) & np.isin(docs, self.doc_ids)
        rows = np.searchsorted(self.doc_ids, docs[keep])
        col = col[keep]
        shape = (self.N, len(self.vocab))
        self.tf_title = sp.csr_matrix((tt[keep].astype(np.float64), (rows, col)), shape=shape)
        self.tf_body = sp.csr_matrix((tb[keep].astype(np.float64), (rows, col)), shape=shape)
        self.tf_title.eliminate_zeros()
        self.tf_body.eliminate_zeros()
        self.tf_all = (self.tf_title + self.tf_body).tocsr()
        self.df = np.diff(self.tf_all.tocsc().indptr).astype(np.float64)
        self._cache = {}

    def idf(self, scheme):
        df = np.maximum(self.df, 1)
        if scheme == "n":
            return np.ones_like(df)
        if scheme == "t":
            return np.log10(self.N / df)
        if scheme == "p":
            return np.maximum(0.0, np.log10(np.maximum(self.N - df, 1e-9) / df))
        raise ValueError(scheme)

    def zone_weights(self, zone, tf, idf, norm, slope):
        key = (zone, tf, idf, norm, slope)
        if key in self._cache:
            return self._cache[key]
        base = {"title": self.tf_title, "body": self.tf_body, "all": self.tf_all}[zone]
        w = _tf(base, tf) @ sp.diags(self.idf(idf))
        w = w.tocsr()
        if norm == "c":
            lens = np.sqrt(np.asarray(w.multiply(w).sum(axis=1)).ravel())
            if slope != 1.0:
                avg = lens[lens > 0].mean()
                lens = (1.0 - slope) * avg + slope * lens
            inv = np.divide(1.0, lens, out=np.zeros_like(lens), where=lens > 0)
            w = sp.diags(inv) @ w
        w = w.tocsr()
        self._cache[key] = w
        return w

    def doc_matrix(self, cfg):
        """Combined N x V document weight matrix for a config."""
        if cfg.zones:
            wt = self.zone_weights("title", cfg.doc_tf, cfg.doc_idf, cfg.doc_norm, cfg.slope)
            wb = self.zone_weights("body", cfg.doc_tf, cfg.doc_idf, cfg.doc_norm, cfg.slope)
            return (cfg.g * wt + (1.0 - cfg.g) * wb).tocsr()
        return self.zone_weights("all", cfg.doc_tf, cfg.doc_idf, cfg.doc_norm, cfg.slope)

    def query_vector(self, text, cfg):
        """-> (term ids, weights), cosine-normalized."""
        counts = {}
        for t in self.analyzer.analyze(text):
            i = self.tid.get(t)
            if i is not None:
                counts[i] = counts.get(i, 0) + 1
        if not counts:
            return np.zeros(0, dtype=np.int64), np.zeros(0)
        ids = np.fromiter(counts, dtype=np.int64)
        w = _tf_vec(np.array([counts[i] for i in ids]), cfg.q_tf) * self.idf(cfg.q_idf)[ids]
        n = np.linalg.norm(w)
        if n > 0:
            w = w / n
        keep = w > 0
        return ids[keep], w[keep]

    def query_matrix(self, texts, cfg):
        """V x nq sparse matrix of query vectors (for batch evaluation)."""
        rows, cols, vals = [], [], []
        for j, t in enumerate(texts):
            ids, w = self.query_vector(t, cfg)
            rows.extend(ids)
            cols.extend([j] * len(ids))
            vals.extend(w)
        return sp.csc_matrix((vals, (rows, cols)), shape=(len(self.vocab), len(texts)))


def rank_from_scores(scores, doc_ids, depth=1000):
    """Docs with positive score, by score desc then docID asc."""
    scores = np.round(scores, 10)   # make ties exact regardless of summation order
    pos = np.flatnonzero(scores > 0)
    if pos.size > depth:
        part = np.argpartition(-scores[pos], depth - 1)[:depth]
        thr = scores[pos][part].min()
        pos = pos[scores[pos] >= thr]
    order = np.lexsort((doc_ids[pos], -scores[pos]))
    return [int(doc_ids[i]) for i in pos[order][:depth]]


def batch_rankings(coll, cfg, texts, depth=1000, W=None):
    W = coll.doc_matrix(cfg) if W is None else W
    S = (W @ coll.query_matrix(texts, cfg)).toarray()
    return [rank_from_scores(S[:, j], coll.doc_ids, depth) for j in range(len(texts))]


class Scorer:
    """Online top-k scoring for one config, with exhaustive TAAT and three Ch. 7 pruning methods."""

    def __init__(self, coll, cfg, champion_r=50, n_leaders=None, seed=13):
        self.coll = coll
        self.cfg = cfg
        self.W = coll.doc_matrix(cfg)          # CSR, doc-major
        self.Wt = self.W.tocsc()               # CSC, term-major = postings with weights
        self.idf = coll.idf("t")
        self.N = coll.N
        self._champ_r = None
        self.build_champions(champion_r)
        self._leaders = None
        self.n_leaders = n_leaders or int(round(np.sqrt(self.N)))
        self.seed = seed

    # ------------------------------------------------------------------ helpers
    def _top(self, cand, scores, k):
        if cand.size == 0:
            return []
        scores = np.round(scores, 10)
        if cand.size > k:
            thr = scores[np.argpartition(-scores, k - 1)[:k]].min()
            sel = scores >= thr              # keep everything tied with the k-th score
            cand, scores = cand[sel], scores[sel]
        order = np.lexsort((self.coll.doc_ids[cand], -scores))
        return [(int(self.coll.doc_ids[cand[i]]), float(scores[i])) for i in order[:k]]

    def _score_candidates(self, cand, ids, w):
        """Full cosine scores for an explicit (sorted) candidate set.  For each query term
        the candidates are located in its postings by binary search, O(|cand| log df),
        so the postings are never scanned in full."""
        ip, ix, dat = self.Wt.indptr, self.Wt.indices, self.Wt.data
        scores = np.zeros(cand.size)
        for t, wq in zip(ids, w):
            rows = ix[ip[t]:ip[t + 1]]
            if rows.size == 0:
                continue
            pos = np.searchsorted(rows, cand)
            pos[pos == rows.size] = 0
            hit = rows[pos] == cand
            scores[hit] += wq * dat[ip[t] + pos[hit]]
        return scores

    # --------------------------------------------------------------- exhaustive
    def exhaustive(self, ids, w, k=10):
        """Term-at-a-time over the full postings of every query term."""
        acc = np.zeros(self.N)
        touched = []
        ip, ix, dat = self.Wt.indptr, self.Wt.indices, self.Wt.data
        for t, wq in zip(ids, w):
            s, e = ip[t], ip[t + 1]
            rows = ix[s:e]
            acc[rows] += wq * dat[s:e]
            touched.append(rows)
        cand = np.unique(np.concatenate(touched)) if touched else np.zeros(0, dtype=np.int64)
        return self._top(cand, acc[cand], k), int(cand.size)

    # ------------------------------------------------------- index elimination
    def index_elimination(self, ids, w, k=10, idf_min=0.5, min_frac=0.5):
        """IIR Sec. 7.1.2: (a) only query terms with idf >= idf_min, (b) only docs containing
        at least ceil(min_frac * |kept terms|) of them."""
        keep = self.idf[ids] >= idf_min
        if not keep.any():
            keep = self.idf[ids] == self.idf[ids].max()
        kids, kw = ids[keep], w[keep]
        ip, ix = self.Wt.indptr, self.Wt.indices
        lists = [ix[ip[t]:ip[t + 1]] for t in kids]
        need = max(1, int(np.ceil(min_frac * len(kids))))
        allrows = np.concatenate(lists) if lists else np.zeros(0, dtype=np.int64)
        if need > 1:
            cnt = np.bincount(allrows, minlength=self.N)
            cand = np.flatnonzero(cnt >= need)
            if cand.size < k:          # too strict: relax to any kept term
                cand = np.unique(allrows)
        else:
            cand = np.unique(allrows)
        if cand.size == 0:
            return [], 0
        return self._top(cand, self._score_candidates(cand, kids, kw), k), int(cand.size)

    # ----------------------------------------------------------- champion lists
    def build_champions(self, r):
        """IIR Sec. 7.1.3: for each term, the r docs with the highest weight in W."""
        if r == self._champ_r:
            return
        ip, ix, dat = self.Wt.indptr, self.Wt.indices, self.Wt.data
        champs = []
        for t in range(self.Wt.shape[1]):
            s, e = ip[t], ip[t + 1]
            if e - s <= r:
                champs.append(ix[s:e])
            else:
                part = np.argpartition(-dat[s:e], r - 1)[:r]
                champs.append(np.sort(ix[s:e][part]))
        self.champions = champs
        self._champ_r = r

    def champion(self, ids, w, k=10):
        lists = [self.champions[t] for t in ids]
        if not lists:
            return [], 0
        cand = np.unique(np.concatenate(lists))
        if cand.size == 0:
            return [], 0
        return self._top(cand, self._score_candidates(cand, ids, w), k), int(cand.size)

    # ----------------------------------------------------------- cluster pruning
    def build_clusters(self, b1=1):
        """IIR Sec. 7.1.6: sqrt(N) random leaders; each doc attaches to its b1 nearest leaders."""
        rng = np.random.default_rng(self.seed)
        lens = np.sqrt(np.asarray(self.W.multiply(self.W).sum(axis=1)).ravel())
        inv = np.divide(1.0, lens, out=np.zeros_like(lens), where=lens > 0)
        U = (sp.diags(inv) @ self.W).tocsr()
        leaders = np.sort(rng.choice(self.N, size=self.n_leaders, replace=False))
        L = U[leaders]
        sims = (U @ L.T).toarray()
        sims[leaders, np.arange(len(leaders))] = np.inf      # a leader belongs to its own cluster
        near = np.argsort(-sims, axis=1)[:, :b1]
        docs = np.repeat(np.arange(self.N), b1)
        lead = near.ravel()
        order = np.lexsort((docs, lead))
        bounds = np.searchsorted(lead[order], np.arange(len(leaders) + 1))
        self._leaders = leaders
        self._L = L
        self._b1 = b1
        self._followers = [docs[order[bounds[i]:bounds[i + 1]]] for i in range(len(leaders))]

    def cluster(self, ids, w, k=10, b2=1, b1=1):
        if self._leaders is None or self._b1 != b1:
            self.build_clusters(b1)
        q = np.zeros(self.W.shape[1])
        q[ids] = w
        ls = self._L @ q
        best = np.argsort(-ls)[:b2]
        cand = np.unique(np.concatenate([self._followers[i] for i in best]))
        scored = len(self._leaders) + cand.size
        return self._top(cand, self._score_candidates(cand, ids, w), k), int(scored)

    # ------------------------------------------------------------------- facade
    def search(self, text, k=10, method="exhaustive", **kw):
        ids, w = self.coll.query_vector(text, self.cfg)
        if ids.size == 0:
            return [], 0
        return getattr(self, method)(ids, w, k=k, **kw)

    def timed(self, text, k=10, method="exhaustive", repeat=3, **kw):
        best = float("inf")
        for _ in range(repeat):
            t0 = time.perf_counter()
            res, n = self.search(text, k, method, **kw)
            best = min(best, time.perf_counter() - t0)
        return res, n, best
