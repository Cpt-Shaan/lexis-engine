"""Index-free reference evaluator: scans the lowercased text of every document.

It shares only the query parser and the tokenizer with the real engine.  It
does not touch the index, k-grams or any list algorithm, so agreement between
the two is a meaningful correctness check.
"""

import re

from .boolean import parse
from .corpus import stream_documents
from .text import tokenize


class BruteForce:
    def __init__(self, corpus_dir=None, docs=None):
        """docs: optional iterable of (doc_id, title, body) to scan instead of a corpus directory."""
        if docs is None:
            docs = ((d.doc_id, d.title, d.body) for d in stream_documents(corpus_dir))
        self.docs = []
        for doc_id, title, body in docs:
            toks = tokenize(title) + tokenize(body)
            self.docs.append((doc_id, toks, set(toks)))

    def _match(self, node, toks, tset):
        kind = node[0]
        if kind == "term":
            return node[1] in tset
        if kind == "wild":
            rx = re.compile("^" + ".*".join(re.escape(p) for p in node[1].split("*")) + "$")
            return any(rx.match(t) for t in tset)
        if kind in ("phrase", "prox"):
            return bool(self._positions(node, toks))
        if kind == "and":
            return all(self._match(k, toks, tset) for k in node[1])
        if kind == "or":
            return any(self._match(k, toks, tset) for k in node[1])
        if kind == "not":
            return not self._match(node[1], toks, tset)
        raise ValueError(kind)

    def _positions(self, node, toks):
        kind = node[0]
        if kind == "term":
            return [i for i, t in enumerate(toks) if t == node[1]]
        if kind == "wild":
            rx = re.compile("^" + ".*".join(re.escape(p) for p in node[1].split("*")) + "$")
            return [i for i, t in enumerate(toks) if rx.match(t)]
        if kind == "phrase":
            w = node[1]
            n = len(w)
            return [i for i in range(len(toks) - n + 1) if toks[i : i + n] == w]
        if kind == "prox":
            a = self._positions(node[1], toks)
            b = self._positions(node[2], toks)
            k = node[3]
            return [i for i in a if any(abs(i - j) <= k for j in b)]
        raise ValueError(kind)

    def search(self, query):
        node = parse(query) if isinstance(query, str) else query
        return [d for d, toks, tset in self.docs if self._match(node, toks, tset)]
