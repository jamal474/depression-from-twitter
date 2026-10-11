# analysis/ — rebuilt pipeline (Oct 2026)

Replaces the notebook pipeline for *analysis* purposes. The original notebooks are left untouched.
See `../INFERENCES.md` for what this found and why each step exists.

| Script | Input | Output |
|---|---|---|
| `build_corpus.py <data_dir> <out_dir>` | `neg_rawdata.csv`, `pos_rawdata.csv`, `worldcities.csv` | `out/corpus.parquet` |
| `evaluate_models.py <out_dir>` | corpus | `model_eval.json`, `top_features.csv` |
| `analyze.py <data_dir> <out_dir>` | corpus (+ legacy `FinalResult.csv` if present) | `a_*`…`g_*.csv`, `summary.json` |
| `keyword_usages.py <out_dir>` | corpus | `h_keyword_usages.csv` |
| `blog_csvs.py <out_dir>` | the tables above | `out/blog/*.csv`, the chart data for the blog post |

`out/corpus.parquet` (~37 MB) is not committed; regenerate it with `build_corpus.py` (~1 min).
