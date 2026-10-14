| Method | overlap@10 | docs scored / query | vs exhaustive | latency ms | speedup | P@10 | NDCG@10 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Exhaustive TAAT (all postings of all query terms) | 1.000 | 1,982 | 100.0% | 0.410 | 1.00× | 0.754 | 0.714 |
| Index elimination: idf ≥ 0.5, ≥ 50% of terms | 0.975 | 298 | 15.0% | 0.318 | 1.29× | 0.753 | 0.717 |
| Index elimination: idf ≥ 1.0, ≥ 50% of terms | 0.809 | 208 | 10.5% | 0.283 | 1.45× | 0.719 | 0.661 |
| Index elimination: idf ≥ 0.5, all terms | 0.931 | 1,100 | 55.5% | 0.594 | 0.69× | 0.739 | 0.707 |
| Champion lists r = 20 | 0.733 | 76 | 3.8% | 0.260 | 1.58× | 0.713 | 0.662 |
| Champion lists r = 50 | 0.930 | 177 | 8.9% | 0.290 | 1.42× | 0.747 | 0.705 |
| Champion lists r = 100 | 0.975 | 321 | 16.2% | 0.307 | 1.34× | 0.749 | 0.709 |
| Cluster pruning √N leaders, b₁ = 1, b₂ = 1 | 0.216 | 191 | 9.6% | 0.286 | 1.44× | 0.344 | 0.318 |
| Cluster pruning √N leaders, b₁ = 1, b₂ = 5 | 0.446 | 554 | 28.0% | 0.382 | 1.07× | 0.546 | 0.531 |
| Cluster pruning √N leaders, b₁ = 3, b₂ = 3 | 0.512 | 865 | 43.6% | 0.466 | 0.88× | 0.590 | 0.560 |
| Cluster pruning √N leaders, b₁ = 3, b₂ = 10 | 0.774 | 2,626 | 132.5% | 1.058 | 0.39× | 0.709 | 0.661 |

All 80 queries (train + test), configuration: lnu0.3.ltc, zones g=0.1, stem=on, stop=on; collection N = 10,788. Latency is the best of 20 runs per query, then averaged over queries. Offline cost: champion lists (r=50) 0.10s, clustering (104 leaders) 0.10s.
