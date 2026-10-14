"""Porter stemmer (M.F. Porter, "An algorithm for suffix stripping", 1980).

A from-scratch implementation of the original algorithm (steps 1a-5b), with
no dictionary look-ups.  It is checked against NLTK's ORIGINAL_ALGORITHM mode
in tests/test_text.py when NLTK happens to be installed.
"""

from functools import lru_cache

_VOWELS = frozenset("aeiou")


def _is_cons(w, i):
    c = w[i]
    if c in _VOWELS:
        return False
    if c == "y":
        return i == 0 or not _is_cons(w, i - 1)
    return True


def _measure(stem):
    """m in [C](VC)^m[V]."""
    m = 0
    i, n = 0, len(stem)
    while i < n and _is_cons(stem, i):
        i += 1
    while i < n:
        while i < n and not _is_cons(stem, i):
            i += 1
        if i >= n:
            break
        while i < n and _is_cons(stem, i):
            i += 1
        m += 1
    return m


def _has_vowel(stem):
    return any(not _is_cons(stem, i) for i in range(len(stem)))


def _ends_double_cons(w):
    return len(w) >= 2 and w[-1] == w[-2] and _is_cons(w, len(w) - 1)


def _cvc(w):
    """*o: stem ends cvc where the final c is not w, x or y."""
    if len(w) < 3:
        return False
    if not (_is_cons(w, len(w) - 3) and not _is_cons(w, len(w) - 2) and _is_cons(w, len(w) - 1)):
        return False
    return w[-1] not in "wxy"


def _replace(w, suffix, repl, min_m):
    """If w ends with suffix and m(stem) > min_m, replace. Returns (matched, word)."""
    if not w.endswith(suffix):
        return False, w
    stem = w[: len(w) - len(suffix)]
    if _measure(stem) > min_m:
        return True, stem + repl
    return True, w


_STEP2 = [
    ("ational", "ate"), ("tional", "tion"), ("enci", "ence"), ("anci", "ance"),
    ("izer", "ize"), ("abli", "able"), ("alli", "al"), ("entli", "ent"),
    ("eli", "e"), ("ousli", "ous"), ("ization", "ize"), ("ation", "ate"),
    ("ator", "ate"), ("alism", "al"), ("iveness", "ive"), ("fulness", "ful"),
    ("ousness", "ous"), ("aliti", "al"), ("iviti", "ive"), ("biliti", "ble"),
]
_STEP3 = [
    ("icate", "ic"), ("ative", ""), ("alize", "al"), ("iciti", "ic"),
    ("ical", "ic"), ("ful", ""), ("ness", ""),
]
_STEP4 = [
    "al", "ance", "ence", "er", "ic", "able", "ible", "ant", "ement", "ment",
    "ent", "ion", "ou", "ism", "ate", "iti", "ous", "ive", "ize",
]


@lru_cache(maxsize=200_000)
def stem(word):
    w = word
    if len(w) <= 2:
        return w

    # Step 1a
    if w.endswith("sses"):
        w = w[:-2]
    elif w.endswith("ies"):
        w = w[:-2]
    elif w.endswith("ss"):
        pass
    elif w.endswith("s"):
        w = w[:-1]

    # Step 1b
    extra = False
    if w.endswith("eed"):
        if _measure(w[:-3]) > 0:
            w = w[:-1]
    elif w.endswith("ed") and _has_vowel(w[:-2]):
        w = w[:-2]
        extra = True
    elif w.endswith("ing") and _has_vowel(w[:-3]):
        w = w[:-3]
        extra = True
    if extra:
        if w.endswith(("at", "bl", "iz")):
            w += "e"
        elif _ends_double_cons(w) and w[-1] not in "lsz":
            w = w[:-1]
        elif _measure(w) == 1 and _cvc(w):
            w += "e"

    # Step 1c
    if w.endswith("y") and _has_vowel(w[:-1]):
        w = w[:-1] + "i"

    # Step 2 (longest matching suffix only; the table order resolves overlaps)
    for suf, rep in sorted(_STEP2, key=lambda p: -len(p[0])):
        if w.endswith(suf):
            _, w = _replace(w, suf, rep, 0)
            break

    # Step 3
    for suf, rep in sorted(_STEP3, key=lambda p: -len(p[0])):
        if w.endswith(suf):
            _, w = _replace(w, suf, rep, 0)
            break

    # Step 4
    for suf in sorted(_STEP4, key=len, reverse=True):
        if w.endswith(suf):
            stem_ = w[: -len(suf)]
            if suf == "ion":
                if _measure(stem_) > 1 and stem_ and stem_[-1] in "st":
                    w = stem_
            elif _measure(stem_) > 1:
                w = stem_
            break

    # Step 5a
    if w.endswith("e"):
        s = w[:-1]
        m = _measure(s)
        if m > 1 or (m == 1 and not _cvc(s)):
            w = s

    # Step 5b
    if _measure(w) > 1 and _ends_double_cons(w) and w.endswith("l"):
        w = w[:-1]
    return w
