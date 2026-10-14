"""The one tokenizer / normalizer used for documents *and* queries.

Pipeline (each stage is applied identically on both sides):

1. Case folding      ``text.lower()``
2. Tokenization      maximal runs of ``[a-z0-9]``; everything else is a separator.
                     "U.S.-Japan" -> u, s, japan;  "1,000" -> 1, 000;  "don't" -> don, t
3. Stop words        optional; the 25 non-selective Reuters-RCV1 words of IIR Sec. 2.2.2
4. Stemming          optional; Porter (1980), see ``porter.py``

Stages 1-2 define the *exact* term layer.  The positional index stores these
terms, so Boolean/phrase/proximity/wildcard answers can match a brute-force scan
of the lowercased text.  Stages 3-4 are a term-to-term mapping on top of the
exact layer and are used only for ranked retrieval.  Because they map terms to
terms, the ranked index can be derived from the positional index without
re-reading the corpus, and stemming or stop words can be switched off for
ablations.
"""

import re

from .porter import stem as porter_stem

TOKEN_RE = re.compile(r"[a-z0-9]+")

# IIR Section 2.2.2 / Fig. 2.5: "a stop list of 25 semantically non-selective
# words which are common in Reuters-RCV1".
STOP_WORDS = frozenset(
    "a an and are as at be by for from has he in is it its of on that the to was were will with".split()
)


def tokenize(text):
    """Case-fold and split into exact terms."""
    return TOKEN_RE.findall(text.lower())


class Analyzer:
    """Maps exact terms to index terms for ranked retrieval."""

    def __init__(self, stem=True, stop=True):
        self.stem = stem
        self.stop = stop
        self._cache = {}

    def normalize(self, term):
        """Exact (case-folded) term -> index term, or None if it is a stop word."""
        try:
            return self._cache[term]
        except KeyError:
            pass
        out = term
        if self.stop and term in STOP_WORDS:
            out = None
        elif self.stem:
            out = porter_stem(term)
        self._cache[term] = out
        return out

    def analyze(self, text):
        out = []
        for t in tokenize(text):
            n = self.normalize(t)
            if n is not None:
                out.append(n)
        return out

    def __repr__(self):
        return f"Analyzer(stem={self.stem}, stop={self.stop})"
