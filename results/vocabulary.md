| Setting | T (tokens) | M (terms) | Heaps k | Heaps b | R² | Zipf α (log-binned) | R² | Zipf α (ranks 1–1000) | Zipf α (every rank) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| case-folded | 1,468,154 | 30,952 | 26.7 | 0.499 | 0.9964 | 1.102 | 0.976 | 0.947 | 1.582 |
| case-folded + stemmed | 1,468,154 | 23,402 | 21.6 | 0.493 | 0.9983 | 1.143 | 0.966 | 0.971 | 1.671 |
| case-folded + stemmed + 25 stop words | 1,116,797 | 23,390 | 24.2 | 0.495 | 0.9984 | 1.042 | 0.945 | 0.859 | 1.651 |

|  | k | b |
| --- | ---: | ---: |
| Reuters-RCV1 (IIR Fig. 5.1) | 44 | 0.49 |

With our stemmed fit, a collection the size of the RCV1 sample (T = 1,000,020) would have M ≈ 19,763 terms; the textbook reports 38,323 actual and 38,365 predicted.
