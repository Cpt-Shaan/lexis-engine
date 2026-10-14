| Variant | train MAP | ΔMAP | p (t) | p (Wilcoxon) | test MAP | ΔMAP | p (t) | p (Wilcoxon) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| **Lexis best** (lnu0.3.ltc, zones g=0.1, stem=on, stop=on) | 0.6559 |  |  |  | 0.6095 |  |  |  |
| stemming off | 0.6275 | -0.0283 | 0.027 | 0.017 | 0.5734 | -0.0360 | 0.0085 | 0.026 |
| stop words off | 0.6544 | -0.0015 | 0.32 | 0.36 | 0.6073 | -0.0021 | 0.3 | 0.52 |
| idf off (doc and query) | 0.6162 | -0.0397 | 0.025 | 0.037 | 0.5054 | -0.1041 | 5.3e-06 | 5.7e-08 |
| length normalization off | 0.6048 | -0.0510 | 4.3e-05 | 4.5e-06 | 0.5351 | -0.0743 | 0.00011 | 3.2e-06 |
| zone weighting off (title+body as one zone) | 0.6573 | +0.0015 | 0.67 | 0.79 | 0.6115 | +0.0020 | 0.69 | 0.26 |
