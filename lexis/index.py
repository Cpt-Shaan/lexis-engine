"""Query-time view of the index: the main on-disk segment plus dynamic updates.

Dynamic indexing (IIR Sec. 4.5)
-------------------------------
* Added documents are inverted into a small in-memory auxiliary index, persisted
  as one run file (aux/aux.run).  Queries read main and auxiliary together.
* Deleted documents go into an invalidation set (aux/deleted.txt).  Results are
  filtered through it, and ranking statistics (N, df) use only live documents.
* merge() streams main (minus deletions) and auxiliary through the same k-way
  merge used at build time and writes a fresh main segment.  The corpus is never
  re-read or re-tokenized, so this is not a full rebuild.  It runs automatically
  once the auxiliary index holds more than ``aux_limit`` documents.
"""

import json
import os
import shutil

import numpy as np

from .codec import vb_decode_np
from .construct import _doc_terms, merge_streams
from .corpus import Document, id_to_name
from .storage import (IndexWriter, RunWriter, decode_positional, read_dictionary,
                      read_doc_table, read_run)


class DiskIndex:
    """Read-only main segment.  Postings are decoded lazily per term."""

    def __init__(self, path):
        self.path = path
        self.terms, arr = read_dictionary(os.path.join(path, "dict.bin"))
        self.df = arr["df"]
        self.cf = arr["cf"]
        self.pos_len = arr["pos_len"]
        self.doc_len = arr["doc_len"]
        self.pos_off = np.concatenate(([0], np.cumsum(self.pos_len)))
        self.doc_off = np.concatenate(([0], np.cumsum(self.doc_len)))
        self.tid = {t: i for i, t in enumerate(self.terms)}
        with open(os.path.join(path, "postings_pos.bin"), "rb") as f:
            self.pos_data = f.read()
        with open(os.path.join(path, "postings_doc.bin"), "rb") as f:
            self.doc_data = f.read()
        self.docs = read_doc_table(os.path.join(path, "docs.tsv"))
        with open(os.path.join(path, "meta.json")) as f:
            self.meta = json.load(f)

    def positional(self, term):
        i = self.tid.get(term)
        if i is None:
            return []
        return decode_positional(self.pos_data, int(self.pos_off[i]), int(self.df[i]))

    def stream(self):
        for i, t in enumerate(self.terms):
            yield t, decode_positional(self.pos_data, int(self.pos_off[i]), int(self.df[i]))

    def zone_postings(self):
        """Decode the whole zone-tf file at once -> (term_index, doc, tf_title, tf_body) arrays."""
        v = vb_decode_np(self.doc_data).reshape(-1, 3)
        df = self.df
        term_idx = np.repeat(np.arange(len(df)), df)
        gaps = v[:, 0]
        c = np.cumsum(gaps)
        starts = np.concatenate(([0], np.cumsum(df)[:-1]))
        base = np.where(starts > 0, c[np.maximum(starts - 1, 0)], 0)
        docs = c - np.repeat(base, df)
        return term_idx, docs, v[:, 1], v[:, 2]


class Index:
    """Main segment + auxiliary index + invalidation set."""

    def __init__(self, path, aux_limit=1000):
        self.path = path
        self.aux_limit = aux_limit
        self.version = 0
        self._load()

    # ------------------------------------------------------------------ loading
    def _load(self):
        self.main = DiskIndex(self.path)
        self.aux = {}            # term -> list[(doc, positions)]
        self.aux_docs = {}       # doc -> (name, title_len, body_len)
        self.deleted = set()
        aux_dir = os.path.join(self.path, "aux")
        if os.path.exists(os.path.join(aux_dir, "aux.run")):
            for t, pl in read_run(os.path.join(aux_dir, "aux.run")):
                self.aux[t] = pl
            self.aux_docs = read_doc_table(os.path.join(aux_dir, "docs.tsv"))
        if os.path.exists(os.path.join(aux_dir, "deleted.txt")):
            with open(os.path.join(aux_dir, "deleted.txt")) as f:
                self.deleted = {int(x) for x in f.read().split()}
        self._refresh()

    def _refresh(self):
        self.docs = {d: v for d, v in self.main.docs.items() if d not in self.deleted}
        self.docs.update({d: v for d, v in self.aux_docs.items() if d not in self.deleted})
        self.doc_ids = sorted(self.docs)
        self.N = len(self.doc_ids)
        self.title_len = {d: v[1] for d, v in self.docs.items()}
        self._vocab = None
        self.version += 1

    # ------------------------------------------------------------------- access
    @property
    def vocabulary(self):
        """Sorted vocabulary.  Terms whose postings are all deleted stay until the next merge."""
        if self._vocab is None:
            if self.aux:
                self._vocab = sorted(set(self.main.terms) | set(self.aux))
            else:
                self._vocab = self.main.terms
        return self._vocab

    def __contains__(self, term):
        return term in self.main.tid or term in self.aux

    def collection_frequency(self, term):
        i = self.main.tid.get(term)
        cf = int(self.main.cf[i]) if i is not None else 0
        return cf + sum(len(p) for _, p in self.aux.get(term, ()))

    def positional(self, term):
        """Live positional postings [(doc, [positions])], sorted by doc."""
        pl = self.main.positional(term)
        if self.deleted:
            pl = [x for x in pl if x[0] not in self.deleted]
        extra = self.aux.get(term)
        if extra:
            pl = pl + [x for x in extra if x[0] not in self.deleted]
        return pl

    def docs_for(self, term):
        return [d for d, _ in self.positional(term)]

    def zone_postings(self):
        """(terms list, term_index, doc, tf_title, tf_body) over live docs, for building the ranker."""
        ti, docs, tt, tb = self.main.zone_postings()
        terms = list(self.main.terms)
        if self.deleted:
            keep = ~np.isin(docs, np.fromiter(self.deleted, dtype=np.int64))
            ti, docs, tt, tb = ti[keep], docs[keep], tt[keep], tb[keep]
        if self.aux:
            tid = dict(self.main.tid)
            ext = [[], [], [], []]
            for t, pl in self.aux.items():
                i = tid.get(t)
                if i is None:
                    i = tid[t] = len(terms)
                    terms.append(t)
                for d, pos in pl:
                    if d in self.deleted:
                        continue
                    nt = sum(1 for p in pos if p < self.title_len[d])
                    ext[0].append(i); ext[1].append(d); ext[2].append(nt); ext[3].append(len(pos) - nt)
            ti, docs, tt, tb = (np.concatenate((a, np.array(b, dtype=np.int64)))
                                for a, b in zip((ti, docs, tt, tb), ext))
        return terms, ti, docs, tt, tb

    def doc_name(self, doc_id):
        return self.docs[doc_id][0] if doc_id in self.docs else id_to_name(doc_id)

    def stats(self):
        return {"live_docs": self.N, "main_docs": len(self.main.docs), "aux_docs": len(self.aux_docs),
                "deleted": len(self.deleted), "vocabulary": len(self.vocabulary)}

    # ---------------------------------------------------------------- updates
    def add_document(self, title, body, name=None):
        doc_id = max(max(self.main.docs, default=0), max(self.aux_docs, default=0)) + 1
        doc = Document(doc_id, name or id_to_name(doc_id), title, body)
        local, row = _doc_terms(doc)
        for t, pos in local.items():
            self.aux.setdefault(t, []).append((doc_id, pos))
        self.aux_docs[doc_id] = row[1:]
        self._refresh()
        self._save_aux()
        if len(self.aux_docs) > self.aux_limit:
            self.merge()
        return doc_id

    def delete_document(self, doc_id):
        if doc_id not in self.docs:
            raise KeyError(f"no live document {doc_id}")
        self.deleted.add(doc_id)
        self._refresh()
        self._save_aux()

    def _save_aux(self):
        aux_dir = os.path.join(self.path, "aux")
        os.makedirs(aux_dir, exist_ok=True)
        w = RunWriter(os.path.join(aux_dir, "aux.run"))
        for t in sorted(self.aux):
            w.add(t, self.aux[t])
        w.close()
        with open(os.path.join(aux_dir, "docs.tsv"), "w", encoding="utf-8") as f:
            for d in sorted(self.aux_docs):
                f.write("\t".join(map(str, (d,) + tuple(self.aux_docs[d]))) + "\n")
        with open(os.path.join(aux_dir, "deleted.txt"), "w") as f:
            f.write("\n".join(map(str, sorted(self.deleted))))

    def merge(self):
        """Fold the auxiliary index and deletions into a new main segment (no corpus re-read)."""
        dead = self.deleted

        def live(stream):
            for t, pl in stream:
                if dead:
                    pl = [x for x in pl if x[0] not in dead]
                if pl:
                    yield t, pl

        tmp = self.path.rstrip("/\\") + ".merging"
        shutil.rmtree(tmp, ignore_errors=True)
        rows = [(d,) + tuple(v) for d, v in self.docs.items()]
        w = IndexWriter(tmp, {d: v[1] for d, v in self.docs.items()})
        merge_streams([live(self.main.stream()), live(sorted(self.aux.items()))], w)
        meta = dict(self.main.meta)
        w.close(rows, {k: meta[k] for k in ("built_with", "budget") if k in meta} | {"merged": True})
        # keep side files that are not part of the segment (e.g. the tuned rank_config.json)
        segment = {"dict.bin", "docs.tsv", "meta.json", "postings_pos.bin", "postings_doc.bin"}
        for f in os.listdir(self.path):
            src = os.path.join(self.path, f)
            if f not in segment and os.path.isfile(src):
                shutil.copy2(src, os.path.join(tmp, f))
        # swap in the new segment
        self.main = None
        old = self.path.rstrip("/\\") + ".old"
        shutil.rmtree(old, ignore_errors=True)
        os.replace(self.path, old)
        os.replace(tmp, self.path)
        shutil.rmtree(old, ignore_errors=True)
        self._load()
