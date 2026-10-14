| Postings | uncompressed (32-bit ints) | VB-coded gaps | VB size vs docIDs only |
| --- | ---: | ---: | ---: |
| docIDs only (Boolean index) | 3.14 MB | 959.70 KB | 1.00× |
| docIDs + tf (plain index) | 6.27 MB | 1.72 MB | 1.84× |
| docIDs + tf_title + tf_body (Lexis zone index) | 9.41 MB | 2.51 MB | 2.67× |
| docIDs + tf + positions (Lexis positional index) | 11.87 MB | 3.39 MB | 3.62× |

Positional / plain (docID+tf), VB-coded: **1.97×**; uncompressed: **1.89×**. Average tokens per document = 136, average tf per posting = 1.79.
