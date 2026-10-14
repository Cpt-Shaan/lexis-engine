"""Lexis: one object that answers every query type over a (dynamic) index."""

import json
import os

from .boolean import BooleanEngine, is_boolean_query, leaves, parse
from .index import Index
from .ranking import Collection, RankConfig, Scorer
from .text import tokenize
from .tolerant import KGramIndex

DEFAULT_CONFIG_FILE = "rank_config.json"


class Lexis:
    def __init__(self, index_dir, rank_config=None, kgram_k=2):
        self.index = Index(index_dir)
        self.kgram_k = kgram_k
        if rank_config is None:
            path = os.path.join(index_dir, DEFAULT_CONFIG_FILE)
            rank_config = RankConfig(**json.load(open(path))) if os.path.exists(path) else RankConfig()
        self.rank_config = rank_config
        self._version = None
        self._sync()

    def _sync(self):
        """Rebuild derived structures after the index changes (add/delete/merge)."""
        if self._version == self.index.version:
            return
        self.kgram = KGramIndex(self.index.vocabulary, self.kgram_k, self.index.collection_frequency)
        self.boolean = BooleanEngine(self.index, self.kgram)
        self._scorer = None
        self._version = self.index.version

    @property
    def scorer(self):
        self._sync()
        if self._scorer is None:
            self._scorer = Scorer(Collection(self.index, self.rank_config.stem, self.rank_config.stop),
                                  self.rank_config)
        return self._scorer

    # ------------------------------------------------------------------- queries
    def suggestions(self, query, boolean=None):
        """{misspelled term: best correction} for every query term not in the vocabulary."""
        self._sync()
        if boolean is None:
            boolean = is_boolean_query(query)
        if boolean:
            terms = []
            for leaf in leaves(parse(query)):
                if leaf[0] == "term":
                    terms.append(leaf[1])
                elif leaf[0] == "phrase":
                    terms.extend(leaf[1])
        else:
            terms = tokenize(query)
        out = {}
        for t in terms:
            if t not in self.index and t not in out:
                s = self.kgram.suggest(t)
                out[t] = s[0] if s else None
        return out

    def search(self, query, k=10, mode="auto", method="exhaustive", **kw):
        """mode: 'boolean', 'ranked' or 'auto' (Boolean syntax -> exact, otherwise ranked)."""
        self._sync()
        boolean = is_boolean_query(query) if mode == "auto" else mode == "boolean"
        result = {"query": query, "mode": "boolean" if boolean else "ranked",
                  "suggestions": self.suggestions(query, boolean)}
        if boolean:
            docs = self.boolean.search(query)
            result["total"] = len(docs)
            result["hits"] = [(d, None) for d in docs]
        else:
            hits, scored = self.scorer.search(query, k, method, **kw)
            result["hits"] = hits
            result["docs_scored"] = scored
        result["hits"] = [(d, self.index.doc_name(d), s) for d, s in result["hits"]]
        return result

    def wildcard(self, pattern):
        self._sync()
        return self.kgram.wildcard(pattern.lower())

    # ------------------------------------------------------------------- updates
    def add_document(self, title, body, name=None):
        return self.index.add_document(title, body, name)

    def delete_document(self, doc_id):
        self.index.delete_document(doc_id)

    def merge(self):
        self.index.merge()
