| id | query | type | expected | Lexis | match | Lexis ms | brute-force ms |
| --- | --- | --- | ---: | ---: | :---: | ---: | ---: |
| b01 | `oil AND opec` | boolean | 120 | 120 | ✓ | 1.33 | 10 |
| b02 | `wheat AND corn AND NOT soviet` | boolean | 66 | 66 | ✓ | 0.86 | 9 |
| b03 | `(gold OR silver) AND price` | boolean | 44 | 44 | ✓ | 1.28 | 18 |
| b04 | `bank AND NOT (dollar OR yen)` | boolean | 1031 | 1031 | ✓ | 2.26 | 12 |
| b05 | `tanker AND (gulf OR iran)` | boolean | 26 | 26 | ✓ | 0.40 | 9 |
| b06 | `coffee AND (brazil OR colombia) AND quota` | boolean | 29 | 29 | ✓ | 0.62 | 10 |
| b07 | `sugar AND NOT export` | boolean | 149 | 149 | ✓ | 0.67 | 10 |
| b08 | `merger OR takeover` | boolean | 589 | 589 | ✓ | 0.65 | 12 |
| b09 | `dividend AND NOT quarterly` | boolean | 514 | 514 | ✓ | 0.67 | 10 |
| b10 | `steel AND (japan OR korea) AND import` | boolean | 5 | 5 | ✓ | 1.14 | 10 |
| b11 | `"interest rates"` | phrase | 239 | 239 | ✓ | 1.47 | 148 |
| b12 | `"trade deficit"` | phrase | 173 | 173 | ✓ | 1.56 | 149 |
| b13 | `"federal reserve"` | phrase | 215 | 215 | ✓ | 1.16 | 156 |
| b14 | `"central bank"` | phrase | 326 | 326 | ✓ | 1.92 | 159 |
| b15 | `"crude oil"` | phrase | 231 | 231 | ✓ | 1.57 | 159 |
| b16 | `"bank of japan"` | phrase | 126 | 126 | ✓ | 21.21 | 158 |
| b17 | `"stock split"` | phrase | 234 | 234 | ✓ | 1.53 | 159 |
| b18 | `econom*` | wildcard | 872 | 872 | ✓ | 1.61 | 162 |
| b19 | `agricultur*` | wildcard | 568 | 568 | ✓ | 0.60 | 177 |
| b20 | `*port` | wildcard | 1639 | 1639 | ✓ | 2.23 | 170 |
| b21 | `pet*chem*` | wildcard | 23 | 23 | ✓ | 0.17 | 181 |
| b22 | `economy /5 growth` | proximity | 16 | 16 | ✓ | 0.98 | 109 |
| b23 | `oil /3 price` | proximity | 76 | 76 | ✓ | 2.28 | 112 |
| b24 | `dollar /4 yen` | proximity | 93 | 93 | ✓ | 1.10 | 112 |
| b25 | `interest /2 rates` | proximity | 245 | 245 | ✓ | 1.40 | 109 |

**25/25** result sets are identical to the expected sets and to the brute-force scan. Mean latency: Lexis 2.03 ms, brute force 93 ms.
