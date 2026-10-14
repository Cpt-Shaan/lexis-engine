"""Index construction: SPIMI and BSBI under a simulated memory budget, k-way merge, 4-shard parallel build.

Memory budget model (the "dummy memory constraint")
---------------------------------------------------
We charge the in-memory inversion structure with a fixed cost model rather
than measuring Python object sizes, so a budget means the same thing on every
machine and for both algorithms:

* SPIMI  new dictionary entry: 48 bytes + len(term); new posting: 16 bytes; each position: 4 bytes
* BSBI   each (termID, docID, position) triple: 12 bytes; the termID map is treated as
         resident, as in IIR Sec. 4.2

When the charge reaches the budget, the block is sorted and written to disk.
Blocks are flushed at document boundaries, so a posting never straddles two
blocks.  Real peak memory is measured separately with tracemalloc (see
experiments/construction.py).
"""

import heapq
import os
import shutil
import time
from array import array
from concurrent.futures import ProcessPoolExecutor

from .corpus import list_corpus, stream_documents
from .storage import IndexWriter, RunWriter, finalize_index, read_run

TERM_COST = 48
POSTING_COST = 16
POSITION_COST = 4
TRIPLE_COST = 12


class BuildStats(dict):
    def __getattr__(self, k):
        return self[k]


def _doc_terms(doc):
    """Invert one document: term -> positions, plus the doc-table row."""
    title, body = doc.zones()
    local = {}
    for i, t in enumerate(title + body):
        pl = local.get(t)
        if pl is None:
            local[t] = [i]
        else:
            pl.append(i)
    return local, (doc.doc_id, doc.name, len(title), len(body))


# ------------------------------------------------------------------------- SPIMI
def spimi_invert(docs, run_dir, budget, prefix="spimi", toc=None, sparse=None):
    """Single-pass in-memory indexing (IIR Fig. 4.4).  Returns (run paths, doc rows, stats)."""
    os.makedirs(run_dir, exist_ok=True)
    runs, rows = [], []
    dictionary = {}
    used = peak = 0
    bytes_written = tokens = 0

    def flush():
        nonlocal dictionary, used, bytes_written
        if not dictionary:
            return
        path = os.path.join(run_dir, f"{prefix}_{len(runs):04d}.run")
        w = RunWriter(path, toc)
        for term in sorted(dictionary):
            w.add(term, dictionary[term])
        bytes_written += w.close()
        runs.append(path)
        if sparse is not None:
            sparse[path] = w.sparse
        dictionary = {}
        used = 0

    for doc in docs:
        local, row = _doc_terms(doc)
        rows.append(row)
        tokens += row[2] + row[3]
        for t, pos in local.items():
            pl = dictionary.get(t)
            if pl is None:
                pl = dictionary[t] = []
                used += TERM_COST + len(t)
            pl.append((doc.doc_id, pos))
            used += POSTING_COST + POSITION_COST * len(pos)
        peak = max(peak, used)
        if used >= budget:
            flush()
    flush()
    return runs, rows, BuildStats(blocks=len(runs), bytes_written=bytes_written,
                                  sim_peak=peak, tokens=tokens)


def merge_streams(streams, writer):
    """k-way merge of (term, postings) streams sorted by term.

    Streams must be given in increasing docID order, i.e. stream i holds only
    docs smaller than those of stream i+1 for any shared term.  That holds for
    successive SPIMI blocks, contiguous shards, and main-then-auxiliary.
    """
    heap = []
    iters = [iter(s) for s in streams]
    for i, it in enumerate(iters):
        rec = next(it, None)
        if rec is not None:
            heap.append((rec[0], i, rec[1]))
    heapq.heapify(heap)
    while heap:
        term, i, postings = heapq.heappop(heap)
        merged = postings
        nxt = next(iters[i], None)
        if nxt is not None:
            heapq.heappush(heap, (nxt[0], i, nxt[1]))
        while heap and heap[0][0] == term:
            _, j, more = heapq.heappop(heap)
            if merged is postings:
                merged = list(postings)
            merged.extend(more)
            nxt = next(iters[j], None)
            if nxt is not None:
                heapq.heappush(heap, (nxt[0], j, nxt[1]))
        writer.add(term, merged)


def _merge_runs_to_index(runs, out_dir, rows, extra_meta=None):
    title_len = {r[0]: r[2] for r in rows}
    w = IndexWriter(out_dir, title_len)
    merge_streams([read_run(p) for p in runs], w)
    meta = w.close(rows, extra_meta)
    return w.bytes, meta


def build_spimi(corpus_dir, out_dir, budget, files=None, keep_runs=False):
    t0 = time.perf_counter()
    run_dir = os.path.join(out_dir, "_runs")
    runs, rows, st = spimi_invert(stream_documents(corpus_dir, files), run_dir, budget)
    t1 = time.perf_counter()
    final_bytes, meta = _merge_runs_to_index(runs, out_dir, rows,
                                             {"built_with": "spimi", "budget": budget})
    t2 = time.perf_counter()
    if not keep_runs:
        shutil.rmtree(run_dir, ignore_errors=True)
    st.update(algorithm="SPIMI", budget=budget, invert_s=t1 - t0, merge_s=t2 - t1, total_s=t2 - t0,
              bytes_written=st["bytes_written"] + final_bytes, run_bytes=st["bytes_written"],
              final_bytes=final_bytes)
    return st, meta


# -------------------------------------------------------------------------- BSBI
_DOC_SHIFT = 24
_TID_SHIFT = 48
_MASK = (1 << 24) - 1


def bsbi_invert(docs, run_dir, budget, prefix="bsbi"):
    """Blocked sort-based indexing (IIR Fig. 4.2).

    Each token becomes a (termID, docID, position) triple packed in a 64-bit
    key.  When the budget fills, the block is sorted and written uncompressed,
    8 bytes per triple.
    """
    os.makedirs(run_dir, exist_ok=True)
    term_ids = {}
    runs, rows = [], []
    buf = []
    bytes_written = tokens = 0
    peak = 0
    limit = max(1, budget // TRIPLE_COST)

    def flush():
        nonlocal buf, bytes_written
        if not buf:
            return
        buf.sort()
        path = os.path.join(run_dir, f"{prefix}_{len(runs):04d}.blk")
        with open(path, "wb") as f:
            array("Q", buf).tofile(f)
        bytes_written += 8 * len(buf)
        runs.append(path)
        buf = []

    for doc in docs:
        title, body = doc.zones()
        rows.append((doc.doc_id, doc.name, len(title), len(body)))
        base = doc.doc_id << _DOC_SHIFT
        for i, t in enumerate(title + body):
            tid = term_ids.get(t)
            if tid is None:
                tid = term_ids[t] = len(term_ids)
            buf.append((tid << _TID_SHIFT) | base | i)
        tokens += len(title) + len(body)
        peak = max(peak, TRIPLE_COST * len(buf))
        if len(buf) >= limit:
            flush()
    flush()
    return runs, rows, term_ids, BuildStats(blocks=len(runs), bytes_written=bytes_written,
                                            sim_peak=peak, tokens=tokens)


def _read_block(path, chunk=1 << 15):
    with open(path, "rb") as f:
        while True:
            a = array("Q")
            try:
                a.fromfile(f, chunk)
            except EOFError:
                pass
            if not a:
                return
            yield from a


def build_bsbi(corpus_dir, out_dir, budget, files=None, keep_runs=False):
    """BSBI.  The merge yields termID order, which is first-occurrence order and
    not lexicographic, so the merged postings go to a temporary run and are then
    copied out in term order.  That extra pass is part of BSBI's cost."""
    t0 = time.perf_counter()
    run_dir = os.path.join(out_dir, "_runs")
    runs, rows, term_ids, st = bsbi_invert(stream_documents(corpus_dir, files), run_dir, budget)
    t1 = time.perf_counter()
    id_to_term = [None] * len(term_ids)
    for t, i in term_ids.items():
        id_to_term[i] = t

    # 1) k-way merge of sorted blocks -> per-termID postings, written in termID order
    tmp_path = os.path.join(run_dir, "merged_by_tid.run")
    tw = RunWriter(tmp_path)
    offsets = {}
    cur_tid = -1
    postings = []
    cur_doc = -1
    pos = None

    def emit():
        if cur_tid >= 0:
            offsets[id_to_term[cur_tid]] = tw.bytes
            tw.add(id_to_term[cur_tid], postings)

    for key in heapq.merge(*[_read_block(p) for p in runs]):
        tid = key >> _TID_SHIFT
        doc = (key >> _DOC_SHIFT) & _MASK
        p = key & _MASK
        if tid != cur_tid:
            emit()
            cur_tid, postings, cur_doc = tid, [], -1
        if doc != cur_doc:
            pos = []
            postings.append((doc, pos))
            cur_doc = doc
        pos.append(p)
    emit()
    tmp_bytes = tw.close()

    # 2) copy out in lexicographic term order
    title_len = {r[0]: r[2] for r in rows}
    w = IndexWriter(out_dir, title_len)
    with open(tmp_path, "rb") as f:
        def lex_stream():
            for term in sorted(offsets):
                f.seek(offsets[term])
                yield next(_read_one(f))
        for term, pl in lex_stream():
            w.add(term, pl)
    meta = w.close(rows, {"built_with": "bsbi", "budget": budget})
    t2 = time.perf_counter()
    if not keep_runs:
        shutil.rmtree(run_dir, ignore_errors=True)
    st.update(algorithm="BSBI", budget=budget, invert_s=t1 - t0, merge_s=t2 - t1, total_s=t2 - t0,
              run_bytes=st["bytes_written"],
              bytes_written=st["bytes_written"] + tmp_bytes + w.bytes, final_bytes=w.bytes)
    return st, meta


def _read_one(f):
    """Read a single run record at the current file position."""
    from .storage import _read_vb_from_file, decode_positional
    from .codec import VBReader
    n = _read_vb_from_file(f)
    payload = f.read(n)
    r = VBReader(payload)
    tl = r.next()
    term = payload[r.pos : r.pos + tl].decode("utf-8")
    r.pos += tl
    df = r.next()
    yield term, decode_positional(payload, r.pos, df)


# ---------------------------------------------------------------- parallel build
def _map_worker(args):
    """Parser (map): SPIMI over one contiguous docID range -> sorted runs + per-term byte counts."""
    corpus_dir, files, run_dir, budget, shard_no = args
    t0 = time.perf_counter()
    toc, sparse = {}, {}
    runs, rows, st = spimi_invert(stream_documents(corpus_dir, files), run_dir, budget,
                                  prefix=f"s{shard_no}", toc=toc, sparse=sparse)
    st.update(shard=shard_no, docs=len(rows), seconds=time.perf_counter() - t0)
    return runs, rows, dict(st), (toc, sparse)


def _reduce_worker(args):
    """Inverter (reduce): k-way merge of every run, restricted to terms in [lo, hi)."""
    runs, lo, hi, part_dir, title_len, sparse = args
    t0 = time.perf_counter()
    w = IndexWriter(part_dir, title_len)
    merge_streams([read_run(p, lo, hi, sparse[p]) for p in runs], w)
    lists = w.close_postings()
    return part_dir, lists, w.bytes, time.perf_counter() - t0


def _term_partitions(toc, n):
    """Split points that give each reducer about the same number of run bytes."""
    terms = sorted(toc)
    total = sum(toc.values())
    cuts, acc, k = [], 0, 1
    for t in terms:
        if acc >= k * total / n and k < n:
            cuts.append(t)
            k += 1
        acc += toc[t]
    bounds = [None] + cuts + [None]
    return list(zip(bounds[:-1], bounds[1:]))


def build_parallel(corpus_dir, out_dir, budget, n_shards=4, workers=None):
    """MapReduce-style build (IIR Sec. 4.4, Fig. 4.5) on one machine with n processes.

    map     n parsers each run SPIMI over a contiguous docID range (document partitioning)
            and write sorted runs, each under the same memory limit.
    reduce  n inverters each own a term range (balanced by run bytes).  Each one
            k-way merges its range out of every run and writes a partial postings file.
    finish  the partial files are concatenated in term order and the dictionary is
            written.  Docs are contiguous per shard and terms contiguous per
            partition, so no postings list is re-sorted and the result is
            byte-identical to the sequential build.
    """
    t0 = time.perf_counter()
    files = list_corpus(corpus_dir)
    n = len(files)
    bounds = [round(i * n / n_shards) for i in range(n_shards + 1)]
    run_dir = os.path.join(out_dir, "_runs")
    os.makedirs(run_dir, exist_ok=True)
    jobs = [(corpus_dir, files[bounds[i]:bounds[i + 1]], run_dir, budget, i) for i in range(n_shards)]
    with ProcessPoolExecutor(max_workers=workers or n_shards) as ex:
        list(ex.map(int, range(n_shards)))     # start the worker processes
        t_up = time.perf_counter()
        maps = list(ex.map(_map_worker, jobs))
        t1 = time.perf_counter()
        runs = [p for r, _, _, _ in maps for p in r]        # shard order = docID order
        rows = [row for _, rs, _, _ in maps for row in rs]
        toc, sparse = {}, {}
        for _, _, _, (tc, sp) in maps:
            sparse.update(sp)
            for t, b in tc.items():
                toc[t] = toc.get(t, 0) + b
        title_len = {r[0]: r[2] for r in rows}
        parts = _term_partitions(toc, n_shards)
        rjobs = [(runs, lo, hi, os.path.join(run_dir, f"part{i}"), title_len, sparse)
                 for i, (lo, hi) in enumerate(parts)]
        reds = list(ex.map(_reduce_worker, rjobs))
    t2 = time.perf_counter()
    # concatenate partial outputs
    os.makedirs(out_dir, exist_ok=True)
    lists = {"terms": [], "df": [], "cf": [], "pos_len": [], "doc_len": []}
    for name in ("postings_pos.bin", "postings_doc.bin"):
        with open(os.path.join(out_dir, name), "wb") as out:
            for part_dir, _, _, _ in reds:
                with open(os.path.join(part_dir, name), "rb") as f:
                    shutil.copyfileobj(f, out)
    for _, l, _, _ in reds:
        for k in lists:
            lists[k].extend(l[k])
    meta, meta_bytes = finalize_index(out_dir, lists, rows,
                                      {"built_with": f"spimi-mapreduce-{n_shards}", "budget": budget})
    t3 = time.perf_counter()
    shutil.rmtree(run_dir, ignore_errors=True)
    shard_stats = [s for _, _, s, _ in maps]
    run_bytes = sum(s["bytes_written"] for s in shard_stats)
    part_bytes = sum(b for _, _, b, _ in reds)
    st = BuildStats(algorithm=f"SPIMI x{n_shards} shards", budget=budget,
                    blocks=sum(s["blocks"] for s in shard_stats), run_bytes=run_bytes,
                    bytes_written=run_bytes + 2 * part_bytes + meta_bytes,
                    startup_s=t_up - t0, invert_s=t1 - t0, merge_s=t3 - t1, reduce_s=t2 - t1, concat_s=t3 - t2,
                    total_s=t3 - t0, shards=shard_stats, reducers=[s for _, _, _, s in reds],
                    sim_peak=max(s["sim_peak"] for s in shard_stats))
    return st, meta
