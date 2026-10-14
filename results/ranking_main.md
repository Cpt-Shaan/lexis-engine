| System | train P@10 | train MAP | train NDCG@10 | test P@10 | test MAP | test NDCG@10 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Boolean (AND) | 0.3250 | 0.0636 | 0.3764 | 0.2400 | 0.0654 | 0.2885 |
| Boolean (OR) | 0.0900 | 0.0775 | 0.0631 | 0.0500 | 0.0559 | 0.0290 |
| raw-tf (nnc.nnc, one zone) | 0.7175 | 0.4560 | 0.6295 | 0.4775 | 0.3379 | 0.4389 |
| lnc.ltc (one zone) | 0.7225 | 0.5635 | 0.6049 | 0.6400 | 0.5339 | 0.5406 |
| **Lexis best**: lnu0.3.ltc, zones g=0.1, stem=on, stop=on | 0.8125 | 0.6559 | 0.7673 | 0.6950 | 0.6095 | 0.6608 |

| Test-set comparison (per-query AP) | ΔMAP | p (paired t) | p (Wilcoxon) |
| --- | ---: | ---: | ---: |
| best vs Boolean (AND) | +0.5441 | 1.1e-17 | 1.8e-12 |
| best vs Boolean (OR) | +0.5535 | 7e-17 | 1.8e-12 |
| best vs raw-tf (nnc.nnc, one zone) | +0.2716 | 1.3e-10 | 4.5e-11 |
| best vs lnc.ltc (one zone) | +0.0756 | 0.00032 | 0.00013 |
