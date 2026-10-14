"""Command-line interface for Lexis.

    python lexis_cli.py build   [--corpus DIR] [--index DIR] [--algo spimi|bsbi|parallel] [--budget BYTES]
    python lexis_cli.py search  "oil AND opec"              # Boolean / phrase / wildcard / proximity
    python lexis_cli.py search  "crude oil price and OPEC"  # free text -> ranked top 10
    python lexis_cli.py search  "..." --method champion     # exhaustive | champion | index_elimination | cluster
    python lexis_cli.py shell                               # interactive prompt
    python lexis_cli.py add     --title T --body-file F     # dynamic insert
    python lexis_cli.py delete  DOC_ID                      # dynamic delete
    python lexis_cli.py merge                               # fold the auxiliary index into the main index
    python lexis_cli.py stats
"""

import argparse
import sys
import time

DEFAULT_CORPUS = "lexis_dataset/corpus"
DEFAULT_INDEX = "index"


def cmd_build(a):
    from lexis.construct import build_bsbi, build_parallel, build_spimi
    fn = {"spimi": build_spimi, "bsbi": build_bsbi, "parallel": build_parallel}[a.algo]
    st, meta = fn(a.corpus, a.index, a.budget)
    print(f"{st['algorithm']}: {st['blocks']} blocks, {st['bytes_written']:,} bytes written, {st['total_s']:.1f}s")
    print(f"{meta['num_docs']:,} docs, {meta['num_terms']:,} terms, {meta['num_tokens']:,} tokens, "
          f"{meta['num_postings']:,} postings")


def _print_result(res, show=10):
    if res["suggestions"]:
        for bad, good in res["suggestions"].items():
            print(f"  '{bad}' is not in the vocabulary" + (f" - did you mean '{good}'?" if good else ""))
    if res["mode"] == "boolean":
        print(f"  {res['total']} matching documents" + (f" (first {show}):" if res["total"] > show else ":"))
        for d, name, _ in res["hits"][:show]:
            print(f"    {name}")
    else:
        print(f"  top {len(res['hits'])} (scored {res['docs_scored']} documents):")
        for d, name, s in res["hits"]:
            print(f"    {s:.4f}  {name}")


def _engine(a):
    from lexis.engine import Lexis
    return Lexis(a.index)


def cmd_search(a):
    eng = _engine(a)
    t0 = time.perf_counter()
    res = eng.search(a.query, mode=a.mode, method=a.method)
    _print_result(res, a.show)
    print(f"  ({(time.perf_counter() - t0) * 1000:.1f} ms)")


def cmd_shell(a):
    eng = _engine(a)
    print("Lexis shell - Boolean (AND OR NOT ( ) \"phrase\" a /k b wild*) or free text. Ctrl-D to quit.")
    while True:
        try:
            q = input("lexis> ").strip()
        except EOFError:
            break
        if not q:
            continue
        try:
            t0 = time.perf_counter()
            _print_result(eng.search(q, mode=a.mode, method=a.method), a.show)
            print(f"  ({(time.perf_counter() - t0) * 1000:.1f} ms)")
        except ValueError as e:
            print("  error:", e)


def cmd_add(a):
    from lexis.index import Index
    ix = Index(a.index)
    body = open(a.body_file, encoding="utf-8").read() if a.body_file else a.body
    print("added doc", ix.add_document(a.title, body, a.name))


def cmd_delete(a):
    from lexis.index import Index
    ix = Index(a.index)
    ix.delete_document(a.doc_id)
    print("deleted doc", a.doc_id)


def cmd_merge(a):
    from lexis.index import Index
    ix = Index(a.index)
    ix.merge()
    print("merged;", ix.stats())


def cmd_stats(a):
    from lexis.index import Index
    import json
    ix = Index(a.index)
    print(json.dumps(ix.stats() | {"meta": ix.main.meta}, indent=2))


def main(argv=None):
    p = argparse.ArgumentParser(description="Lexis search engine")
    p.add_argument("--index", default=DEFAULT_INDEX)
    sub = p.add_subparsers(dest="cmd", required=True)

    b = sub.add_parser("build")
    b.add_argument("--corpus", default=DEFAULT_CORPUS)
    b.add_argument("--algo", choices=["spimi", "bsbi", "parallel"], default="parallel")
    b.add_argument("--budget", type=int, default=4_000_000, help="simulated memory budget in bytes")
    b.set_defaults(fn=cmd_build)

    for name, fn in (("search", cmd_search), ("shell", cmd_shell)):
        s = sub.add_parser(name)
        if name == "search":
            s.add_argument("query")
        s.add_argument("--mode", choices=["auto", "boolean", "ranked"], default="auto")
        s.add_argument("--method", default="exhaustive",
                       choices=["exhaustive", "champion", "index_elimination", "cluster"])
        s.add_argument("--show", type=int, default=10)
        s.set_defaults(fn=fn)

    ad = sub.add_parser("add")
    ad.add_argument("--title", required=True)
    ad.add_argument("--body", default="")
    ad.add_argument("--body-file")
    ad.add_argument("--name")
    ad.set_defaults(fn=cmd_add)

    de = sub.add_parser("delete")
    de.add_argument("doc_id", type=int)
    de.set_defaults(fn=cmd_delete)

    sub.add_parser("merge").set_defaults(fn=cmd_merge)
    sub.add_parser("stats").set_defaults(fn=cmd_stats)

    a = p.parse_args(argv)
    a.fn(a)


if __name__ == "__main__":
    sys.exit(main())
