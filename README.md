# Depression from Twitter

About 150,000 English tweets scraped in May 2023 with ten "negative" keywords (sadness, loneliness, despair, anguish, grief, anxiety, stress, depression, isolation, helplessness) and ten "positive" ones (happiness, joy, love, gratitude, hope, success, motivation, inspiration, excitement, optimism).

The project started as a student paper comparing "depression" between developed and developing countries. A 2026 re-analysis found that those results came from how the data was collected and labelled. What the data does show is how people use the language of distress online. **Read [`INFERENCES.md`](INFERENCES.md) for the full write-up.**

## Layout

```
data/
  neg_rawdata.csv     raw scrape, negative keywords (snscrape, 6–24 May 2023, UTC timestamps)
  pos_rawdata.csv     raw scrape, positive keywords (22–24 May 2023)
  worldcities.csv     SimpleMaps World Cities (basic), used for geocoding
analysis/             rebuilt pipeline: corpus, model evaluation, analyses
  out/                result tables and JSON summaries quoted in INFERENCES.md
INFERENCES.md         findings, audit of the original method, next steps
```

## Reproduce

```bash
pip install -r analysis/requirements.txt
python analysis/build_corpus.py data analysis/out
python analysis/evaluate_models.py analysis/out
python analysis/analyze.py data analysis/out
python analysis/keyword_usages.py analysis/out
```

The tweets are public posts collected in 2023. Please don't use them to identify or contact individual users.
