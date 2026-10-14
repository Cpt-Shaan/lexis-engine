"""Lexis test suite.   python -m unittest discover -s tests -v

Builds small indexes under tests/_work and uses the full index in ./index when
present (it is built if missing).
"""

import filecmp
import os
import random
import shutil
import sys
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from lexis.boolean import BooleanEngine, intersect, parse
from lexis.bruteforce import BruteForce
from lexis.codec import gamma_decode, gamma_encode, vb_decode, vb_decode_np, vb_encode
from lexis.construct import build_bsbi, build_parallel, build_spimi
from lexis.corpus import (list_corpus, load_boolean_tests, load_misspellings, load_queries,
                          stream_documents)
from lexis.index import Index
from lexis.porter import stem
from lexis.ranking import Collection, RankConfig, Scorer, batch_rankings
from lexis.text import Analyzer, tokenize
from lexis.tolerant import KGramIndex

DATA = os.path.join(ROOT, "lexis_dataset")
CORPUS = os.path.join(DATA, "corpus")
WORK = os.path.join(ROOT, "tests", "_work")
FULL = os.path.join(ROOT, "index")
FILES = list_corpus(CORPUS)


def full_index():
    if not os.path.exists(os.path.join(FULL, "dict.bin")):
        build_spimi(CORPUS, FULL, 4_000_000)
    return Index(FULL)


def same_index(a, b):
    return all(filecmp.cmp(os.path.join(a, f), os.path.join(b, f), shallow=False)
               for f in ("postings_pos.bin", "postings_doc.bin", "dict.bin", "docs.tsv"))


class TestText(unittest.TestCase):
    def test_tokenizer(self):
        self.assertEqual(tokenize("U.S.-Japan RIFT, 1,000 dlrs; don't"),
                         ["u", "s", "japan", "rift", "1", "000", "dlrs", "don", "t"])

    def test_analyzer_same_for_docs_and_queries(self):
        a = Analyzer(stem=True, stop=True)
        self.assertEqual(a.analyze("The Exports of WHEAT"), ["export", "wheat"])
        self.assertEqual(Analyzer(stem=False, stop=False).analyze("The Exports"), ["the", "exports"])

    def test_porter_examples(self):
        cases = {"caresses": "caress", "ponies": "poni", "relational": "relat", "conditional": "condit",
                 "generalizations": "gener", "hopping": "hop", "filing": "file", "happy": "happi",
                 "agreed": "agre", "electrical": "electr", "adjustable": "adjust", "controll": "control"}
        for w, s in cases.items():
            self.assertEqual(stem(w), s, w)

    def test_porter_matches_nltk(self):
        try:
            from nltk.stem.porter import PorterStemmer
        except ImportError:
            self.skipTest("nltk not installed")
        ps = PorterStemmer(mode=PorterStemmer.ORIGINAL_ALGORITHM)
        vocab = full_index().vocabulary
        # Words of length <= 2 are deliberately left alone (as in Porter's reference C code).
        bad = [w for w in vocab if len(w) > 2 and stem(w) != ps.stem(w)]
        self.assertEqual(bad, [])


class TestCodecs(unittest.TestCase):
    def test_roundtrips(self):
        rng = random.Random(1)
        xs = [rng.randint(0, 10 ** rng.randint(1, 9)) for _ in range(20000)]
        enc = bytes(vb_encode(xs))
        self.assertEqual(vb_decode(enc), xs)
        self.assertEqual(vb_decode_np(enc).tolist(), xs)
        ys = [x + 1 for x in xs]
        data, nbits = gamma_encode(ys)
        self.assertEqual(gamma_decode(data, nbits), ys)

    def test_textbook_examples(self):
        self.assertEqual(bytes(vb_encode([824, 5, 214577])).hex(), "06b8850d0cb1")   # IIR Table 5.4
        data, n = gamma_encode([13])
        self.assertEqual(n, 7)                                                     # 1110,101


class TestConstruction(unittest.TestCase):
    """SPIMI, BSBI and the 4-shard parallel build must produce byte-identical indexes."""

    @classmethod
    def setUpClass(cls):
        shutil.rmtree(WORK, ignore_errors=True)
        cls.files = FILES[:1200]

    def test_builders_agree(self):
        dirs = {}
        for name, fn in (("spimi", build_spimi), ("bsbi", build_bsbi)):
            st, _ = fn(CORPUS, os.path.join(WORK, name), 200_000, files=self.files)
            self.assertGreater(st["blocks"], 3)
            dirs[name] = os.path.join(WORK, name)
        self.assertTrue(same_index(dirs["spimi"], dirs["bsbi"]))
        big, _ = build_spimi(CORPUS, os.path.join(WORK, "oneblock"), 10 ** 12, files=self.files)
        self.assertEqual(big["blocks"], 1)
        self.assertTrue(same_index(dirs["spimi"], os.path.join(WORK, "oneblock")))

    def test_parallel_matches_full(self):
        full_index()
        tmp = os.path.join(WORK, "par")
        build_parallel(CORPUS, tmp, 4_000_000)
        self.assertTrue(same_index(FULL, tmp))

    def test_dynamic_updates_equal_rebuild(self):
        """Index first 800 docs, add the next 200 dynamically, delete 50, merge,
        and compare with a from-scratch build of the same live documents."""
        base = os.path.join(WORK, "dyn")
        build_spimi(CORPUS, base, 500_000, files=self.files[:800])
        ix = Index(base, aux_limit=10 ** 9)
        added = []
        for d in stream_documents(CORPUS, self.files[800:1000]):
            added.append(ix.add_document(d.title, d.body, d.name))
        self.assertEqual(added, list(range(801, 1001)))
        rng = random.Random(7)
        dead = sorted(rng.sample(range(1, 1001), 50))
        for d in dead:
            ix.delete_document(d)
        # before merging: queries see aux docs and hide deleted ones
        bf = BruteForce(docs=[(d.doc_id, d.title, d.body) for d in stream_documents(CORPUS, self.files[:1000])
                              if d.doc_id not in dead])
        eng = BooleanEngine(ix, KGramIndex(ix.vocabulary))
        for q in ("oil AND NOT opec", '"crude oil"', "trade /3 deficit", "export*", "NOT wheat"):
            self.assertEqual(eng.search(q), bf.search(q), q)
        # survives a reload
        ix = Index(base, aux_limit=10 ** 9)
        self.assertEqual(ix.N, 950)
        ix.merge()
        ref = os.path.join(WORK, "dyn_ref")
        live = [f for f in self.files[:1000] if int(f[4:9]) not in dead]
        build_spimi(CORPUS, ref, 500_000, files=live)
        self.assertTrue(same_index(base, ref))


class TestExactRetrieval(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.ix = full_index()
        cls.kg = KGramIndex(cls.ix.vocabulary, 2, cls.ix.collection_frequency)
        cls.eng = BooleanEngine(cls.ix, cls.kg)
        cls.bf = BruteForce(CORPUS)

    def test_boolean_tests_file(self):
        for t in load_boolean_tests(os.path.join(DATA, "boolean_tests.tsv")):
            got = self.eng.search(t["query"])
            self.assertEqual(len(got), t["n"], t["query"])
            self.assertEqual(got, t["expected"], t["query"])

    def test_random_queries_match_bruteforce(self):
        rng = random.Random(42)
        vocab = [t for t in self.ix.vocabulary if t.isalpha()]
        common = [t for t in vocab if self.ix.collection_frequency(t) >= 20]
        docs = self.bf.docs

        def term():
            return rng.choice(common if rng.random() < .8 else vocab)

        def phrase():
            _, toks, _ = rng.choice(docs)
            n = rng.randint(2, 4)
            i = rng.randrange(max(1, len(toks) - n))
            return '"' + " ".join(toks[i:i + n]) + '"'

        def wild():
            t = rng.choice(common)
            c = rng.randint(1, len(t) - 1)
            return rng.choice([t[:c] + "*", "*" + t[c:], t[:c] + "*" + t[-1:], t[:1] + "*" + t[c:c + 2] + "*"])

        def atom(depth):
            r = rng.random()
            if depth > 0 and r < .3:
                op = rng.choice([" AND ", " OR ", " AND NOT ", " "])
                return "(" + atom(depth - 1) + op + atom(depth - 1) + ")"
            if r < .55:
                return term()
            if r < .7:
                return phrase()
            if r < .85:
                return wild()
            return f"{term()} /{rng.randint(1, 8)} {term()}"

        for _ in range(250):
            q = atom(2) + rng.choice([" AND ", " OR ", " AND NOT ", " "]) + atom(2)
            self.assertEqual(self.eng.search(q), self.bf.search(q), q)

    def test_wildcards_match_vocabulary_scan(self):
        import fnmatch
        for pat in ["econom*", "*port", "pet*chem*", "a*b*c", "*ization", "s*s", "*x*", "q*"]:
            exp = [t for t in self.ix.vocabulary if fnmatch.fnmatchcase(t, pat)]
            self.assertEqual(self.kg.wildcard(pat), exp, pat)

    def test_skips_do_not_change_results(self):
        a, b = self.ix.docs_for("bank"), self.ix.docs_for("rate")
        ref = intersect(a, b)
        for s in ("sqrt", 2, 4, 16, 64):
            self.assertEqual(intersect(a, b, s), ref)

    def test_query_parser(self):
        self.assertEqual(parse("a b OR NOT c"), ("or", [("and", [("term", "a"), ("term", "b")]),
                                                        ("not", ("term", "c"))]))
        self.assertEqual(parse("U.S. /2 trade"), ("prox", ("phrase", ["u", "s"]), ("term", "trade"), 2))


class TestTolerant(unittest.TestCase):
    def test_misspellings(self):
        ix = full_index()
        kg = KGramIndex(ix.vocabulary, 2, ix.collection_frequency)
        rows = load_misspellings(os.path.join(DATA, "misspellings.tsv"))
        hits = sum(kg.suggest(bad)[:1] == good[:1] for bad, good in rows)
        self.assertGreaterEqual(hits / len(rows), 0.95)


class TestRanking(unittest.TestCase):
    def test_taat_equals_batch_scoring(self):
        ix = full_index()
        coll = Collection(ix)
        qs = [q for _, q in load_queries(os.path.join(DATA, "queries_train.tsv"))]
        for cfg in (RankConfig(), RankConfig(zones=False, slope=0.3), RankConfig(doc_idf="t", g=0.6)):
            sc = Scorer(coll, cfg)
            for q, ranking in zip(qs, batch_rankings(coll, cfg, qs)):
                hits, scored = sc.search(q)
                self.assertEqual([d for d, _ in hits], ranking[:10], q)
                self.assertLess(scored, coll.N)


if __name__ == "__main__":
    unittest.main()
