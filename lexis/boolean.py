"""Exact retrieval: Boolean operators, phrases, proximity and wildcards over the positional index.

Query language
--------------
    expr    := or
    or      := and ( "OR" and )*
    and     := unary ( ["AND"] unary )*          adjacent operands are ANDed
    unary   := "NOT" unary | prox
    prox    := atom ( "/k" atom )*               |pos(a) - pos(b)| <= k, either order
    atom    := TERM | WILDCARD | "\"phrase\"" | "(" expr ")"

Operators must be upper case; lower-case "and"/"or"/"not" are ordinary terms.
A term goes through the document tokenizer (case folding only, no stemming).
If it splits into several tokens, e.g. U.S. -> u s, it becomes a phrase, which
is exactly how the same characters were indexed.
"""

import heapq
import math
import re

from .text import tokenize

_TOKEN_RE = re.compile(r'\s*(?:(\()|(\))|"([^"]*)"|/(\d+)|([^\s()"]+))')


class QueryError(ValueError):
    pass


# ------------------------------------------------------------------------ parse
def lex(query):
    out = []
    pos = 0
    query = query.strip()
    while pos < len(query):
        m = _TOKEN_RE.match(query, pos)
        if not m or m.end() == pos:
            raise QueryError(f"cannot parse query near: {query[pos:]!r}")
        pos = m.end()
        lp, rp, phrase, prox, word = m.groups()
        if lp:
            out.append(("(",))
        elif rp:
            out.append((")",))
        elif phrase is not None:
            out.append(("phrase", phrase))
        elif prox is not None:
            out.append(("prox", int(prox)))
        elif word in ("AND", "OR", "NOT"):
            out.append((word,))
        else:
            out.append(("word", word))
    return out


def _leaf(word):
    if "*" in word:
        pat = re.sub(r"[^a-z0-9*]", "", word.lower())
        pat = re.sub(r"\*+", "*", pat)
        if not pat.replace("*", ""):
            raise QueryError(f"wildcard {word!r} has no letters")
        return ("wild", pat)
    toks = tokenize(word)
    if not toks:
        raise QueryError(f"term {word!r} contains no indexable characters")
    return ("term", toks[0]) if len(toks) == 1 else ("phrase", toks)


def parse(query):
    toks = lex(query)
    i = 0

    def peek():
        return toks[i][0] if i < len(toks) else None

    def take():
        nonlocal i
        i += 1
        return toks[i - 1]

    def p_or():
        kids = [p_and()]
        while peek() == "OR":
            take()
            kids.append(p_and())
        return kids[0] if len(kids) == 1 else ("or", kids)

    def p_and():
        kids = [p_unary()]
        while peek() in ("AND", "NOT", "(", "word", "phrase"):
            if peek() == "AND":
                take()
            kids.append(p_unary())
        return kids[0] if len(kids) == 1 else ("and", kids)

    def p_unary():
        if peek() == "NOT":
            take()
            return ("not", p_unary())
        return p_prox()

    def p_prox():
        node = p_atom()
        while peek() == "prox":
            k = take()[1]
            node = ("prox", node, p_atom(), k)
        return node

    def p_atom():
        kind = peek()
        if kind == "(":
            take()
            node = p_or()
            if peek() != ")":
                raise QueryError("missing ')'")
            take()
            return node
        if kind == "word":
            return _leaf(take()[1])
        if kind == "phrase":
            words = tokenize(take()[1])
            if not words:
                raise QueryError("empty phrase")
            return ("term", words[0]) if len(words) == 1 else ("phrase", words)
        raise QueryError(f"unexpected token {toks[i] if i < len(toks) else 'end of query'}")

    if not toks:
        raise QueryError("empty query")
    node = p_or()
    if i != len(toks):
        raise QueryError(f"unexpected token {toks[i]}")
    return node


def is_boolean_query(q):
    """Heuristic used by the CLI to route a query to exact or ranked retrieval."""
    return bool(re.search(r'\b(AND|OR|NOT)\b|["()*]|\s/\d+\s', q))


def leaves(node):
    if node[0] in ("term", "wild", "phrase"):
        yield node
    elif node[0] == "not":
        yield from leaves(node[1])
    elif node[0] == "prox":
        yield from leaves(node[1])
        yield from leaves(node[2])
    else:
        for k in node[1]:
            yield from leaves(k)


# ------------------------------------------------------------- list algorithms
class Counter:
    __slots__ = ("n",)

    def __init__(self):
        self.n = 0


def intersect(p1, p2, skip=None, counter=None):
    """Two-pointer intersection of sorted docID lists (IIR Fig. 2.10).

    skip: spacing of skip pointers.  "sqrt" means floor(sqrt(len)) per list, an
    int means a fixed spacing, None means no skips.  A list position i has a
    skip pointer when i % s == 0 and i + s < len, so pointers are implicit
    strides over the array.  counter.n counts docID comparisons, including the
    comparisons made when testing a skip target.
    """
    n1, n2 = len(p1), len(p2)
    if skip == "sqrt":
        s1, s2 = max(1, int(math.sqrt(n1))), max(1, int(math.sqrt(n2)))
    elif skip:
        s1 = s2 = int(skip)
    else:
        s1 = s2 = 0
    i = j = c = 0
    out = []
    while i < n1 and j < n2:
        a, b = p1[i], p2[j]
        c += 1
        if a == b:
            out.append(a)
            i += 1
            j += 1
            continue
        c += 1
        if a < b:
            if s1 > 1 and i % s1 == 0 and i + s1 < n1:
                c += 1
                if p1[i + s1] <= b:
                    i += s1
                    while i % s1 == 0 and i + s1 < n1:
                        c += 1
                        if p1[i + s1] <= b:
                            i += s1
                        else:
                            break
                    continue
            i += 1
        else:
            if s2 > 1 and j % s2 == 0 and j + s2 < n2:
                c += 1
                if p2[j + s2] <= a:
                    j += s2
                    while j % s2 == 0 and j + s2 < n2:
                        c += 1
                        if p2[j + s2] <= a:
                            j += s2
                        else:
                            break
                    continue
            j += 1
    if counter is not None:
        counter.n += c
    return out


def union(lists):
    out = []
    last = None
    for d in heapq.merge(*lists):
        if d != last:
            out.append(d)
            last = d
    return out


def difference(p1, p2):
    out = []
    j, n2 = 0, len(p2)
    for d in p1:
        while j < n2 and p2[j] < d:
            j += 1
        if j < n2 and p2[j] == d:
            continue
        out.append(d)
    return out


def positional_merge(lists):
    """Union of positional postings lists -> one list with merged positions per doc."""
    acc = {}
    for pl in lists:
        for d, pos in pl:
            acc.setdefault(d, []).append(pos)
    return [(d, sorted(set(p for ps in acc[d] for p in ps)) if len(acc[d]) > 1 else acc[d][0])
            for d in sorted(acc)]


def phrase_step(p1, p2, offset):
    """Keep positions x in p1 such that x + offset occurs in p2 (same doc)."""
    out = []
    i = j = 0
    while i < len(p1) and j < len(p2):
        d1, d2 = p1[i][0], p2[j][0]
        if d1 == d2:
            target = set(p2[j][1])
            keep = [x for x in p1[i][1] if x + offset in target]
            if keep:
                out.append((d1, keep))
            i += 1
            j += 1
        elif d1 < d2:
            i += 1
        else:
            j += 1
    return out


def proximity(p1, p2, k):
    """Positional intersect (IIR Fig. 2.12) with |pos1 - pos2| <= k.  Returns positions from p1."""
    out = []
    i = j = 0
    while i < len(p1) and j < len(p2):
        d1, d2 = p1[i][0], p2[j][0]
        if d1 == d2:
            a, b = p1[i][1], p2[j][1]
            keep = []
            lo = 0
            for x in a:
                while lo < len(b) and b[lo] < x - k:
                    lo += 1
                if lo < len(b) and b[lo] <= x + k:
                    keep.append(x)
            if keep:
                out.append((d1, keep))
            i += 1
            j += 1
        elif d1 < d2:
            i += 1
        else:
            j += 1
    return out


# --------------------------------------------------------------------- evaluate
class BooleanEngine:
    def __init__(self, index, kgram, skip="sqrt"):
        self.index = index
        self.kgram = kgram
        self.skip = skip
        self.comparisons = Counter()

    # positional view of any leaf / positional node
    def _positional(self, node):
        kind = node[0]
        if kind == "term":
            return self.index.positional(node[1])
        if kind == "wild":
            return positional_merge([self.index.positional(t) for t in self.kgram.wildcard(node[1])])
        if kind == "phrase":
            words = node[1]
            res = self.index.positional(words[0])
            for off, w in enumerate(words[1:], 1):
                if not res:
                    break
                res = phrase_step(res, self.index.positional(w), off)
            return res
        if kind == "prox":
            return proximity(self._positional(node[1]), self._positional(node[2]), node[3])
        raise QueryError(f"{kind} cannot be used inside a phrase/proximity operand")

    def _docs(self, node):
        kind = node[0]
        if kind == "term":
            return self.index.docs_for(node[1])
        if kind == "wild":
            return union([self.index.docs_for(t) for t in self.kgram.wildcard(node[1])])
        if kind in ("phrase", "prox"):
            return [d for d, _ in self._positional(node)]
        if kind == "or":
            return union([self._docs(k) for k in node[1]])
        if kind == "not":
            return difference(self.index.doc_ids, self._docs(node[1]))
        if kind == "and":
            pos = [k for k in node[1] if k[0] != "not"]
            neg = [k[1] for k in node[1] if k[0] == "not"]
            if pos:
                lists = sorted((self._docs(k) for k in pos), key=len)  # rarest first
                res = lists[0]
                for other in lists[1:]:
                    if not res:
                        break
                    res = intersect(res, other, self.skip, self.comparisons)
            else:
                res = self.index.doc_ids
            for k in neg:
                if not res:
                    break
                res = difference(res, self._docs(k))
            return res
        raise QueryError(f"unknown node {kind}")

    def search(self, query):
        node = parse(query) if isinstance(query, str) else query
        return self._docs(node)
