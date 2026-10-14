"""On-disk formats: intermediate runs, the final index files, and the compressed dictionary.

Run file (SPIMI blocks, shard outputs, the auxiliary index)
    a sequence of records ``VB(len) payload``, sorted by term, where
    payload = VB(len(term)) term-bytes VB(df) { VB(docgap) VB(tf) VB(posgap)*tf }*df
    The length prefix lets a reader pull one record at a time, so a k-way
    merge holds only one record per run in memory.

Final index directory
    postings_pos.bin  per term: { VB(docgap) VB(tf) VB(posgap)*tf }*df     positional index
    postings_doc.bin  per term: { VB(docgap) VB(tf_title) VB(tf_body) }*df zone tf index (ranking)
    dict.bin          blocked (k=4), front-coded dictionary + VB metadata arrays
    docs.tsv          doc_id, name, title_len, body_len
    meta.json         collection statistics

Positions are token offsets in the stream title + body, so position < title_len
means the token is in the title zone.
"""

import json
import os
import struct
from bisect import bisect_left

import numpy as np

from .codec import VBReader, vb_decode, vb_encode, vb_encode_number

DICT_MAGIC = b"LXD1"


# ----------------------------------------------------------------- postings I/O
def encode_positional(postings, out):
    """postings: list of (doc_id, [positions]) sorted by doc -> appended VB bytes."""
    prev = 0
    for doc, pos in postings:
        vb_encode_number(doc - prev, out)
        prev = doc
        vb_encode_number(len(pos), out)
        p0 = 0
        for p in pos:
            vb_encode_number(p - p0, out)
            p0 = p
    return out


def decode_positional(data, start, df):
    r = VBReader(data, start)
    out = []
    doc = 0
    for _ in range(df):
        doc += r.next()
        tf = r.next()
        pos = []
        p = 0
        for _ in range(tf):
            p += r.next()
            pos.append(p)
        out.append((doc, pos))
    return out


# --------------------------------------------------------------------- run files
SPARSE_EVERY = 64


class RunWriter:
    def __init__(self, path, toc=None):
        self.path = path
        self.f = open(path, "wb")
        self.bytes = 0
        self.records = 0
        self.toc = toc          # optional dict term -> bytes, used to balance term-range reducers
        self.sparse = []        # (term, offset) of every SPARSE_EVERY-th record, for seeking to a term range

    def add(self, term, postings):
        if self.records % SPARSE_EVERY == 0:
            self.sparse.append((term, self.bytes))
        tb = term.encode("utf-8")
        payload = bytearray()
        vb_encode_number(len(tb), payload)
        payload += tb
        vb_encode_number(len(postings), payload)
        encode_positional(postings, payload)
        head = vb_encode([len(payload)])
        self.f.write(head)
        self.f.write(payload)
        self.bytes += len(head) + len(payload)
        self.records += 1
        if self.toc is not None:
            self.toc[term] = self.toc.get(term, 0) + len(head) + len(payload)

    def close(self):
        self.f.close()
        return self.bytes


def _read_vb_from_file(f):
    n = 0
    while True:
        b = f.read(1)
        if not b:
            return None
        b = b[0]
        if b < 128:
            n = (n << 7) | b
        else:
            return (n << 7) | (b & 127)


def read_run(path, lo=None, hi=None, sparse=None):
    """Stream (term, postings) records from a run file, optionally only terms in [lo, hi).
    Records outside the range are skipped without decoding their postings; with the
    writer's sparse (term, offset) table the reader seeks close to lo first."""
    with open(path, "rb", buffering=1 << 16) as f:
        if lo is not None and sparse:
            start = 0
            for t, off in sparse:
                if t >= lo:
                    break
                start = off
            f.seek(start)
        while True:
            n = _read_vb_from_file(f)
            if n is None:
                return
            payload = f.read(n)
            r = VBReader(payload)
            tl = r.next()
            term = payload[r.pos : r.pos + tl].decode("utf-8")
            if lo is not None and term < lo:
                continue
            if hi is not None and term >= hi:
                return
            r.pos += tl
            df = r.next()
            yield term, decode_positional(payload, r.pos, df)


# ------------------------------------------------------------------- dictionary
def write_dictionary(path, terms, arrays, k=4):
    """Blocked storage with front coding (IIR Sec. 5.2.2).

    Within each block of k terms, the first term is stored in full and each
    later term as (shared-prefix length with previous term, suffix).  One block
    pointer is stored per block.  Metadata arrays (df, cf, byte lengths of
    postings) are VB-coded.
    """
    term_sec = bytearray()
    block_ptrs = []
    prev = b""
    for i, t in enumerate(terms):
        tb = t.encode("utf-8")
        if i % k == 0:
            block_ptrs.append(len(term_sec))
            vb_encode_number(len(tb), term_sec)
            term_sec += tb
        else:
            lcp = 0
            m = min(len(prev), len(tb))
            while lcp < m and prev[lcp] == tb[lcp]:
                lcp += 1
            vb_encode_number(lcp, term_sec)
            vb_encode_number(len(tb) - lcp, term_sec)
            term_sec += tb[lcp:]
        prev = tb
    ptr_gaps = [block_ptrs[0]] + [b - a for a, b in zip(block_ptrs, block_ptrs[1:])] if block_ptrs else []
    sections = [bytes(term_sec), bytes(vb_encode(ptr_gaps))]
    names = sorted(arrays)
    for name in names:
        sections.append(bytes(vb_encode(int(x) for x in arrays[name])))
    with open(path, "wb") as f:
        hdr = json.dumps({"n": len(terms), "k": k, "arrays": names}).encode()
        f.write(DICT_MAGIC + struct.pack("<I", len(hdr)) + hdr)
        for s in sections:
            f.write(struct.pack("<I", len(s)))
            f.write(s)
    return {"terms": len(term_sec), "block_pointers": len(sections[1]),
            "metadata": sum(len(s) for s in sections[2:])}


def read_dictionary(path):
    with open(path, "rb") as f:
        data = f.read()
    assert data[:4] == DICT_MAGIC
    (hl,) = struct.unpack_from("<I", data, 4)
    hdr = json.loads(data[8 : 8 + hl])
    p = 8 + hl
    sections = []
    while p < len(data):
        (sl,) = struct.unpack_from("<I", data, p)
        sections.append(data[p + 4 : p + 4 + sl])
        p += 4 + sl
    n, k = hdr["n"], hdr["k"]
    sec = sections[0]
    r = VBReader(sec)
    terms = []
    prev = b""
    for i in range(n):
        if i % k == 0:
            L = r.next()
            tb = sec[r.pos : r.pos + L]
        else:
            lcp = r.next()
            L = r.next()
            tb = prev[:lcp] + sec[r.pos : r.pos + L]
        r.pos += L
        terms.append(tb.decode("utf-8"))
        prev = tb
    arrays = {}
    for name, s in zip(hdr["arrays"], sections[2:]):
        arrays[name] = np.array(vb_decode(s), dtype=np.int64)
    return terms, arrays


# ----------------------------------------------------------------- index writer
class IndexWriter:
    """Writes the final index.  Terms must arrive in lexicographic order."""

    def __init__(self, out_dir, title_len):
        os.makedirs(out_dir, exist_ok=True)
        self.dir = out_dir
        self.title_len = title_len
        self.pos_f = open(os.path.join(out_dir, "postings_pos.bin"), "wb")
        self.doc_f = open(os.path.join(out_dir, "postings_doc.bin"), "wb")
        self.terms, self.df, self.cf, self.pos_len, self.doc_len = [], [], [], [], []
        self.bytes = 0
        self._last = None

    def add(self, term, postings):
        if self._last is not None and term <= self._last:
            raise ValueError(f"terms out of order: {self._last!r} then {term!r}")
        self._last = term
        pb = encode_positional(postings, bytearray())
        db = bytearray()
        prev = 0
        cf = 0
        tl = self.title_len
        for doc, pos in postings:
            vb_encode_number(doc - prev, db)
            prev = doc
            nt = bisect_left(pos, tl[doc])
            vb_encode_number(nt, db)
            vb_encode_number(len(pos) - nt, db)
            cf += len(pos)
        self.pos_f.write(pb)
        self.doc_f.write(db)
        self.bytes += len(pb) + len(db)
        self.terms.append(term)
        self.df.append(len(postings))
        self.cf.append(cf)
        self.pos_len.append(len(pb))
        self.doc_len.append(len(db))

    def close_postings(self):
        """Close the postings files; return the per-term lists (used for partial, term-range outputs)."""
        self.pos_f.close()
        self.doc_f.close()
        return {"terms": self.terms, "df": self.df, "cf": self.cf, "pos_len": self.pos_len,
                "doc_len": self.doc_len}

    def close(self, doc_rows, extra_meta=None):
        """doc_rows: iterable of (doc_id, name, title_len, body_len)."""
        meta, nbytes = finalize_index(self.dir, self.close_postings(), doc_rows, extra_meta)
        self.bytes += nbytes
        return meta


def finalize_index(out_dir, lists, doc_rows, extra_meta=None):
    """Write dict.bin, docs.tsv and meta.json next to already-written postings files."""
    dict_path = os.path.join(out_dir, "dict.bin")
    parts = write_dictionary(dict_path, lists["terms"],
                             {k: lists[k] for k in ("df", "cf", "pos_len", "doc_len")})
    rows = sorted(doc_rows)
    with open(os.path.join(out_dir, "docs.tsv"), "w", encoding="utf-8") as f:
        for r in rows:
            f.write("\t".join(map(str, r)) + "\n")
    meta = {
        "num_docs": len(rows),
        "num_terms": len(lists["terms"]),
        "num_tokens": int(sum(r[2] + r[3] for r in rows)),
        "num_postings": int(sum(lists["df"])),
        "bytes": {
            "postings_pos": os.path.getsize(os.path.join(out_dir, "postings_pos.bin")),
            "postings_doc": os.path.getsize(os.path.join(out_dir, "postings_doc.bin")),
            "dict": os.path.getsize(dict_path),
            "dict_parts": parts,
        },
    }
    meta.update(extra_meta or {})
    with open(os.path.join(out_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    nbytes = sum(os.path.getsize(os.path.join(out_dir, x)) for x in ("dict.bin", "docs.tsv", "meta.json"))
    return meta, nbytes


def read_doc_table(path):
    docs = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            d, name, tl, bl = line.rstrip("\n").split("\t")
            docs[int(d)] = (name, int(tl), int(bl))
    return docs
