| Dictionary representation | Size | bytes/term |
| --- | ---: | ---: |
| Fixed-width array (20-byte term, 4-byte df, 4-byte postings ptr) | 846.34 KB | 28.0 |
| Dictionary as a string (+ 3-byte term ptr, 4 df, 4 ptr) | 537.38 KB | 17.8 |
| Blocked string, k=4 (+1 length byte/term, 3-byte ptr/block) | 499.60 KB | 16.5 |
| Blocked k=4 + front coding (Lexis term section, measured) + 4 df + 4 ptr | 426.03 KB | 14.1 |
| **Lexis dict.bin as stored** (front-coded blocks + VB df, cf and postings lengths for 2 files) | 298.15 KB | 9.9 |

30,952 terms, average length 6.78 characters; the front-coded term section is 165,429 bytes against 240,758 for the blocked string (31% smaller). Lexis stores per term: df, cf, byte length of its positional list and byte length of its zone-tf list, all VB-coded; offsets are prefix sums of lengths.
