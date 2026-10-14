# Lexis — a search engine over Reuters-21578

CSL 436 Information Retrieval, Winter 2026 programming assignment.
Everything is pure Python 3 with NumPy and SciPy.  matplotlib is needed only for the report figures.
The design and all measurements are in **[REPORT.md](REPORT.md)**.

```
pip install numpy scipy matplotlib          # nltk is optional (only for a stemmer cross-check test)
python lexis_cli.py build                   # 4-shard parallel SPIMI build into ./index  (~4 s)
python lexis_cli.py search "oil AND opec"
python lexis_cli.py search "(gold OR silver) AND price"
python lexis_cli.py search '"bank of japan"'
python lexis_cli.py search "economy /5 growth"
python lexis_cli.py search "pet*chem*"
python lexis_cli.py search "crude oil prise and opek output"     # ranked + "did you mean"
python lexis_cli.py search "crude oil price" --method champion   # pruned top-10
python lexis_cli.py shell                                        # interactive
python lexis_cli.py add --title "NEW ARTICLE" --body "text ..."  # dynamic insert
python lexis_cli.py delete 42                                    # dynamic delete
python lexis_cli.py merge                                        # fold auxiliary index into main
python -m unittest discover -s tests -v                          # test suite (~1 min)
```

Build options: `--algo spimi|bsbi|parallel` and `--budget BYTES` (the simulated memory limit; default 4 MB).

Reproducing the report (writes `results/*.md|json` and `report/figures/*.png`):

```
cd experiments
python vocabulary.py      # Heaps / Zipf
python exact.py           # skip pointers, positional vs plain size, Boolean suite, spelling, wildcards
python construction.py    # SPIMI vs BSBI at 7 memory limits, 1/2/4-shard speedup   (~10 min)
python compression_sizes.py  # postings + dictionary compression tables
python ranking_eval.py    # grid search on train, single test run, ablations; writes index/rank_config.json
python efficiency.py      # champion lists / index elimination / cluster pruning vs exhaustive
```

## Layout

| Path | Purpose |
|---|---|
| `lexis/text.py`, `lexis/porter.py` | the one tokenizer/normalizer (case folding, `[a-z0-9]+`, 25 stop words, Porter stemmer) |
| `lexis/corpus.py` | streaming document reader, loaders for queries / qrels / tests |
| `lexis/codec.py` | variable-byte (incl. vectorized decoder), Elias-γ, fixed-width packing |
| `lexis/storage.py` | run-file format, final index files, front-coded blocked dictionary |
| `lexis/construct.py` | SPIMI, BSBI, k-way merge, 4-shard parallel build |
| `lexis/index.py` | query-time index: main segment + auxiliary index + deletion set, merge |
| `lexis/boolean.py` | query parser, AND/OR/NOT, skip-pointer intersection, phrases, proximity |
| `lexis/tolerant.py` | k-gram index: wildcards (any number of `*`), spelling correction |
| `lexis/ranking.py` | zone-weighted tf-idf cosine, TAAT top-k, champion lists, index elimination, cluster pruning |
| `lexis/evaluation.py` | P@10, MAP, NDCG@10, 11-point P/R, paired t / Wilcoxon tests |
| `lexis/bruteforce.py` | index-free reference evaluator used to verify exact retrieval |
| `lexis/engine.py`, `lexis_cli.py` | the `Lexis` facade and command-line interface |
| `experiments/` | scripts that produce every number and figure in the report |
| `tests/test_lexis.py` | unit and end-to-end tests |
