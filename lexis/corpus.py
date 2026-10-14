"""Streaming access to the corpus and loaders for the evaluation files."""

import os
from collections import defaultdict
from dataclasses import dataclass

from .text import tokenize


@dataclass
class Document:
    doc_id: int          # 1-based, from the file name doc_00001.txt -> 1
    name: str            # "doc_00001"
    title: str
    body: str

    def zones(self):
        """(title tokens, body tokens).  Positions run over title then body as one stream."""
        return tokenize(self.title), tokenize(self.body)


def name_to_id(name):
    return int(name.split("_")[1])


def id_to_name(doc_id):
    return f"doc_{doc_id:05d}"


def parse_document(name, raw):
    title, _, body = raw.partition("\n")
    return Document(name_to_id(name), name, title.strip(), body.strip())


def list_corpus(corpus_dir):
    return sorted(f for f in os.listdir(corpus_dir) if f.endswith(".txt"))


def stream_documents(corpus_dir, files=None):
    """Yield documents one at a time; only one article is in memory at once."""
    if files is None:
        files = list_corpus(corpus_dir)
    for f in files:
        with open(os.path.join(corpus_dir, f), encoding="utf-8") as fh:
            yield parse_document(f[:-4], fh.read())


def _rows(path):
    with open(path, encoding="utf-8") as fh:
        header = fh.readline()
        for line in fh:
            line = line.rstrip("\n")
            if line:
                yield line.split("\t")


def load_queries(path):
    return [(qid, q) for qid, q in _rows(path)]


def load_qrels(path):
    """query_id -> {doc_id(int): grade}"""
    out = defaultdict(dict)
    for qid, doc, grade in _rows(path):
        out[qid][name_to_id(doc)] = int(grade)
    return dict(out)


def load_boolean_tests(path):
    out = []
    for row in _rows(path):
        qid, q, typ, n = row[:4]
        exp = row[4] if len(row) > 4 else ""
        ids = sorted(name_to_id(x) for x in exp.split(",") if x)
        out.append({"id": qid, "query": q, "type": typ, "n": int(n), "expected": ids})
    return out


def load_misspellings(path):
    return [(bad, good.split("|")) for bad, good in _rows(path)]
