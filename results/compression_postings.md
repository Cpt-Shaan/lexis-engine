**docID postings**  (all: 822,333 entries; top-100: 258,727; rarest 10,000: 10,000)

| Method | All terms | bits/entry | ratio | Top-100 terms | bits/entry | ratio | Rarest 10,000 | bits/entry | ratio |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Uncompressed (32-bit docIDs) | 3.14 MB | 32.00 | 1.00× | 1,010.65 KB | 32.00 | 1.00× | 39.06 KB | 32.00 | 1.00× |
| Gaps, fixed width per list (bit-packed) | 875.86 KB | 8.73 | 3.67× | 182.52 KB | 5.78 | 5.54× | 29.03 KB | 23.78 | 1.35× |
| Gaps, variable byte | 959.70 KB | 9.56 | 3.35× | 252.67 KB | 8.00 | 4.00× | 19.40 KB | 15.89 | 2.01× |
| Gaps, Elias γ | 880.61 KB | 8.77 | 3.65× | 105.80 KB | 3.35 | 9.55× | 34.94 KB | 28.62 | 1.12× |

**position postings**  (all: 1,468,154 entries; top-100: 681,027; rarest 10,000: 13,406)

| Method | All terms | bits/entry | ratio | Top-100 terms | bits/entry | ratio | Rarest 10,000 | bits/entry | ratio |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Uncompressed (32-bit positions) | 5.60 MB | 32.00 | 1.00× | 2.60 MB | 32.00 | 1.00× | 52.37 KB | 32.00 | 1.00× |
| Gaps, fixed width per list (bit-packed) | 1.81 MB | 10.35 | 3.09× | 876.11 KB | 10.54 | 3.04× | 23.93 KB | 14.63 | 2.19× |
| Gaps, variable byte | 1.67 MB | 9.53 | 3.36× | 728.28 KB | 8.76 | 3.65× | 16.49 KB | 10.08 | 3.18× |
| Gaps, Elias γ | 1.87 MB | 10.66 | 3.00× | 795.24 KB | 9.57 | 3.35× | 22.09 KB | 13.50 | 2.37× |

ratio = uncompressed size / compressed size, so higher is better.  The top-100 vs rarest-10,000 ratio for each method is in the summary below.

| docID postings | ratio top-100 | ratio rarest-10k | top-100 ÷ rarest-10k |
| --- | ---: | ---: | ---: |
| Gaps, fixed width per list (bit-packed) | 5.54× | 1.35× | 4.12 |
| Gaps, variable byte | 4.00× | 2.01× | 1.99 |
| Gaps, Elias γ | 9.55× | 1.12× | 8.54 |
