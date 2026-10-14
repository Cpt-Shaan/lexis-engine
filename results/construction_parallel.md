| Memory limit | Build | Process start s | Invert (map) s | Merge (reduce + concat) s | Total s | Speedup (total) | Speedup map | Speedup merge |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 976.56 KB | sequential SPIMI (1 process) | — | 3.14 | 2.21 | 5.39 | 1.00× | — | — |
| 976.56 KB | 2 shards (map) + 2 term ranges (reduce) | 0.35 | 1.87 | 1.68 + 0.11 | 3.98 | **1.35×** | 1.68× | 1.31× |
| 976.56 KB | 4 shards (map) + 4 term ranges (reduce) | 0.49 | 1.79 | 1.54 + 0.18 | 4.00 | **1.35×** | 1.75× | 1.44× |
| 3.81 MB | sequential SPIMI (1 process) | — | 3.43 | 1.96 | 5.37 | 1.00× | — | — |
| 3.81 MB | 2 shards (map) + 2 term ranges (reduce) | 0.38 | 2.13 | 1.33 + 0.11 | 3.99 | **1.34×** | 1.61× | 1.47× |
| 3.81 MB | 4 shards (map) + 4 term ranges (reduce) | 0.47 | 1.61 | 1.03 + 0.17 | 3.33 | **1.61×** | 2.12× | 1.89× |
