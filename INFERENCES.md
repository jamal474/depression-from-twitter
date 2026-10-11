# Depression from Twitter: a second look at the data

*Review and re-analysis of the May 2023 Twitter scrape (Jamal, Aditya, Mishra), October 2026.*
*Every number here is produced by the scripts in `analysis/` (see the reproducibility section at the end).*

---

## TL;DR

1. **The paper's headline result is a sampling artefact.** Developed countries looked "more depressed" because their tweets came from the negative-keyword scrape far more often (US 24%, India 0.8%). That share alone explains the country ranking (r = 0.85). **When you compare like with like, the gap disappears.** Developed 66.3% vs developing 67.3% "negative", odds ratio 0.99 [0.91–1.09], p = 0.89.
2. **Three of the ten "countries" were mostly the US.** Re-geocoding the paper's own Dataset2: "Mali" is 87% US and 2% Mali (San Francisco/San Diego; *San* is a town in Mali). "Japan" is 64% US and 6% Japan ("Texas, USA"; *Usa* is a city in Japan). "Puerto Rico" is 90% US (Florida and North Carolina; *Florida* and *Carolina* are towns in PR). Only 42% of "Canada" is Canadian, and 57% of "UK" is British.
3. **The time-of-day curve was UTC clock time plus scrape timing** (r = 0.93 with the negative-keyword share per hour). It was not a daily mood rhythm.
4. **The classifier imitates VADER, not depression.** All of the model's top-100 words are VADER lexicon entries. Knowing *which keyword list* a tweet came from already gives 75% accuracy. A single TF-IDF logistic regression (87.3%) beats the whole six-model ensemble.
5. **What the data *can* say is more interesting:**
   - People write about these topics **in the first person much more late at night** (+6–7 percentage points at 21:00–03:00 local time, robust to keyword, country and scrape time). The measured *negativity* barely moves.
   - **Loneliness is the one keyword over-represented late at night** (~21% more than its baseline). Stress is under-represented (~18% less).
   - Developed and developing countries differ in ***how* they talk about distress, not *how much***. First-person use is 44% vs 33% (India 24%, Kenya 22%, US 46%).
   - **Most "negative" keyword tweets are not about the writer's own feelings.** 31% of "grief" tweets are the idiom *"good grief"*. 15% of "depression" tweets are about the Great Depression or the economy. Only ~13% are first-person, and only 1.6% of the negative set contains anything like an explicit self-disclosure, often ironic ("my depression era /j").

The data is a good corpus for studying **how people use the language of distress online**. It is not a measurement of depression prevalence. The blog can be built around that distinction.

---

## 1. What the original pipeline did

```
snscrape  ──►  10 negative + 10 positive keywords, English, up to 10k tweets/keyword
          ──►  location string → country (token lookup in worldcities.csv)
          ──►  VADER compound score  →  label: score ≤ 0 ⇒ "depressive"
          ──►  D1 (32,762 tweets, 191 countries): train 6 classifiers, pick best
               ensemble subset by test accuracy (85.4%)
          ──►  D2 (19,656 tweets, 10 countries): predict "depressive", compare
               developed vs developing, and plot ratio by time of day
```

## 2. Audit: what the original results actually measured

| # | Issue | Evidence | Effect on the paper |
|---|---|---|---|
| 1 | **Keyword mix differs by country** | D2 share from the negative scrape: India 0.8%, Nigeria 0.3%, Philippines 5%, US 24%, UK 23% | Country "depression rate" correlates **r = 0.85** with that share. Fig. 3 measures the scrape, not the countries. |
| 2 | **Geocoder false positives** | Single tokens matched against 44k city names. Top raw locations behind each label: *Mali* → "San Diego, CA", "San Francisco, CA". *Japan* → "USA", "Texas, USA". *Puerto Rico* → "Florida, USA", "North Carolina, USA". *Canada* → "London, England" (London, Ontario). *UK* → "Boston, MA" (Boston, Lincolnshire). | Re-geocoded with the new geocoder, the share of each label that really is that country: Australia 91%, Nigeria 93%, India 89%, US 72%, UK 57%, Philippines 45%, **Canada 42%, Japan 6%, Mali 2%, Puerto Rico 0.4%**. Japan, Mali and Puerto Rico are 64–90% US tweets. |
| 3 | **Time is UTC, not local** | `timeTweetClean` subtracts a `+00:00` offset, which does nothing. Every timestamp from snscrape is UTC. | Fig. 4 mixes time zones. It also tracks *when each scrape ran*: hours 11–12 UTC hold ~7,000 positive-keyword tweets from the 24 May run. r = 0.93 with negative-keyword share. |
| 4 | **Label = VADER** | Label is `VADER ≤ 0`. The final ensemble output agrees with plain VADER **87%** of the time. | "Accuracy" is agreement with a lexicon. VADER alone would score 100%. |
| 5 | **Label partly *is* the search word** | VADER scores "grief", "anguish" and "depression" as negative and "love", "hope" and "joy" as positive. Masking the keyword flips **20%** of labels. Positive-scrape tweets go from 16% to 39% "depressive". | Much of what the model learns is "which list was this tweet scraped with". A source-only baseline gets **75%**. |
| 6 | **Ensemble chosen on the test set** | All 63 subsets were scored on the test set and the best was reported. | Optimistic selection. Re-run properly (train/val/test, user-grouped): no ensemble beats the best single model. |
| 7 | **Leaky random split** | 26% of test-set authors also appear in training. 5% of test texts are exact duplicates of training texts. | Small (~1 pt) inflation. A user-grouped split fixes it. |
| 8 | **Spelling "correction" damages tweet language** | Norvig corrector trained on Gutenberg text, applied to tokens that were never lowercased: *covid→could, PTSD→its, ADHD→add, meds→beds, burnout→burnt, vibes→vices, selfcare→welfare, idk→ink, lol→ll*, and **"I"→"a"**. | It wipes out first-person pronouns (a well-known linguistic marker of depression) and the mental-health vocabulary itself. |
| 9 | **D2 inference bug (A-BiLSTM)** | D2 is hashed with `one_hot(text, len(ct)+1)`, a different vocabulary size from training (`len(count)+1`). The hash indices no longer match the trained embedding. The cell that produced `p1, p2, p3` is also missing. | BiLSTM predictions on D2 are effectively noise. `FinalResult.csv` can't be regenerated from the notebook. |
| 10 | **Data hygiene** | Tweet IDs were destroyed by Excel (`1.66E+18`). Dates were dropped. Paper says `< 0` but code uses `≤ 0` (4.3% of tweets score exactly 0 and became "depressive"). Location integer-encoded and fed to LinearSVC as a number. | Reproducibility and minor validity issues. |
| 11 | **Search matches more than the text** | 9.9% of negative-scrape tweets contain none of the keywords. A third of those were matched on the *username* (e.g. @FreeManDespair, @nflanguish). | Noise that the label treats as signal. |

**Bottom line of the audit:** the modelling was done competently for a student project. The three headline claims (developed > developing, more depression as the day goes on, 85% accuracy at detecting depression) all come from how the data was collected and labelled.

## 3. What was rebuilt (`analysis/`)

| Step | Old | New |
|---|---|---|
| Source | `Dataset1/2.csv` (Excel-mangled) | Raw `neg_rawdata.csv` + `pos_rawdata.csv`. **149,117 unique tweets** (86,090 neg, 62,759 pos, 268 matched both), full IDs and UTC timestamps |
| Provenance | lost | Every tweet keeps `src` (neg/pos list) and the `keyword` found in its text |
| Geocoding | token → any city | Segment-level. Explicit country > US/CA/AU state (full name or ", TX") > city with population ≥ 50k that is unambiguous (≥ 5× the next same-named city). Blocklist for "San", "USA", "Bay", "Carolina", etc. About 95% precise on a manual check of 80 random unique strings. Geocodes 35% of tweets (52.7k). |
| Time | UTC clock | **Local time** via the IANA zone of the resolved city/state (timezonefinder). Country-only hits in multi-zone countries are left out. 49k tweets have local time. |
| Text | spell-correct + lemmatise | URLs and mentions stripped, nothing else. Raw text keeps pronouns, slang and clinical terms. |
| Features | — | VADER on the raw text; VADER with the keyword *masked*; first-person flag; explicit self-disclosure flag; duplicate-text flag |
| Evaluation | random split, test-set model selection | Dedupe; **user-grouped** split; train/val/test for ensemble selection; bootstrap CIs; baselines |
| Comparisons | across the keyword design | **Within** the negative-keyword set, **standardised to a common keyword mix**, user-clustered bootstrap CIs |

### Model re-evaluation (target is still "VADER ≤ 0", to stay comparable with the paper)

| Setting | Accuracy |
|---|---|
| Majority class | 54.0% |
| **Which keyword list it came from (no text at all)** | **74.9%** |
| Naive Bayes | 80.3% |
| Random Forest | 85.1% |
| Linear SVC | 86.7% |
| **Logistic regression (TF-IDF 1–2-grams)** | **87.3%** (±0.4) |
| Best ensemble, chosen on validation | 87.3% (same as LogReg) |
| Paper's reported best ensemble (chosen on test, smaller data) | 85.4% |

The most important features for "depressive" are *anguish, depression, grief, hell, hate, worst, shit, pain, suicide, death*. For "non-depressive" they are *love, inspiration, gratitude, hope, joy, great, success*. Half of each list is a search keyword, and **100%** of the top-100 unigrams are in VADER's lexicon. More model capacity would not help, because the label itself is the limit.

## 4. What the data does say

### 4.1 Developed vs developing: no difference in negativity, a real difference in voice

Negative-keyword tweets only, deduplicated, with the "depressive" rate standardised to the overall keyword mix and 95% CIs from a user-clustered bootstrap:

| Country | Group | n tweets | "Depressive" rate (std.) | 95% CI | First-person rate (std.) | 95% CI |
|---|---|---|---|---|---|---|
| United Kingdom | developed | 3,995 | 61.3% | 59.7–63.7 | 41.5% | 39.9–43.7 |
| Nigeria | developing | 893 | 64.2% | 58.8–71.6 | 34.9% | 29.8–42.4 |
| India | developing | 1,930 | 64.5% | 62.1–67.6 | **23.5%** | 21.0–27.5 |
| Australia | developed | 879 | 64.8% | 61.5–69.2 | 41.0% | 37.4–45.6 |
| Kenya | developing | 275 | 66.1% | 61.2–71.8 | **21.6%** | 17.3–28.7 |
| Canada | developed | 1,431 | 66.6% | 63.5–69.2 | 43.3% | 39.9–46.8 |
| Pakistan | developing | 413 | 66.6% | 62.1–72.8 | 29.7% | 23.9–34.3 |
| South Africa | developing | 599 | 67.5% | 62.0–76.6 | 40.3% | 34.3–45.3 |
| United States | developed | 13,784 | 67.6% | 66.6–68.6 | **45.6%** | 44.7–46.6 |
| **Developed (pooled)** | | 21,349 | **66.3%** | 65.4–67.2 | **44.3%** | |
| **Developing (pooled)** | | 7,156 | **67.3%** | 66.0–68.9 | **32.9%** | |

Logistic regression (`negative ~ developed + keyword`, SEs clustered by user): **OR = 0.99 [0.91, 1.09], p = 0.89.**

**Inference.** When people in these countries tweet about sadness, anxiety, grief and so on, they do it with about the same lexical negativity. The difference is in *register*. Tweets from India, Kenya and Pakistan use these words far less in the first person: more news, religion, politics, customer service and commentary, and less "I feel…". The cause could be culture (stigma around personal disclosure is the obvious hypothesis and fits the literature the paper already cites), second-language English, or who is on English Twitter in each country. This data can't separate those. It is still a more interesting and more defensible finding than the original one.

### 4.2 Time of day: late at night, people talk about themselves

Negative-keyword tweets with known local time (n = 26,576). The model is a linear probability model with fixed effects for keyword, country and **UTC date-hour**. The last one compares tweets posted at the same moment in different time zones, so scrape timing cancels out. Differences are relative to 09:00–12:00 local:

| Local time | Raw first-person | Adjusted Δ first-person (pp) | Adjusted Δ "depressive" (pp) |
|---|---|---|---|
| 00–03 | 45.2% | **+6.4** [3.2, 9.6] | +2.5 [−0.5, 5.5] |
| 03–06 | 42.2% | +2.0 [−1.4, 5.5] | +2.1 [−1.2, 5.3] |
| 06–09 | 36.4% | −1.5 [−4.1, 1.0] | −1.5 [−4.1, 1.1] |
| 09–12 | 37.0% | ref | ref |
| 12–15 | 37.3% | +0.8 [−1.5, 3.1] | −0.1 [−2.4, 2.2] |
| 15–18 | 41.2% | +2.2 [−0.3, 4.7] | +0.7 [−1.8, 3.2] |
| 18–21 | 43.0% | +3.3 [0.8, 5.9] | +0.3 [−2.1, 2.8] |
| 21–24 | 47.7% | **+6.9** [3.9, 9.8] | +0.8 [−1.9, 3.5] |

Which negative keywords lean late-night (22:00–04:00 local), with the same controls:

| Keyword | Late-night share vs baseline | 95% CI (pp) |
|---|---|---|
| **loneliness** | **+21% relative** (+2.2 pp) | [+1.1, +3.3] |
| helplessness | +15% | [−0.2, +1.8] (n.s.) |
| sadness / anguish / depression | +6–7% | n.s. |
| anxiety / grief / despair | ≈ 0 | n.s. |
| isolation | −13% (−1.3 pp) | [−2.4, −0.2] |
| **stress** | **−18% relative** (−1.9 pp) | [−3.0, −0.9] |

**Inference.** The paper's idea that "it gets worse as the day goes on" survives in a different form. The *tone* of distress tweets hardly changes through the day. Their *voice* does: personal, first-person posting rises steadily from morning to midnight, and loneliness concentrates late at night. Stress is a daytime word and loneliness a night-time one, which fits intuition. This is the most blog-worthy result in the data, and it holds up after controlling for the confounds that sank the original chart.

### 4.3 The search words mostly don't mean what the study assumed

| Keyword | Usage | Share of that keyword's tweets |
|---|---|---|
| grief | idiom *"good grief"* (mostly political exasperation; 2,454 distinct users) | **31.1%** |
| depression | Great Depression / economy (largely partisan memes) | **14.8%** |
| depression | first-person ("my depression", "I have depression") | 13.5% |
| depression | clinical / medication / therapy | 6.6% |
| depression | awareness / advocacy | 3.4% |
| anguish | customer-service template ("we understand your anguish… registered contact details"); @zomatocare alone posted 290 | 4.2% |
| isolation | public-health "social isolation" (US Surgeon General's loneliness advisory, May 2023) | 3.4% |
| anxiety | casual ("gives me anxiety") | 2.9% |
| hope | well-wishing ("I hope you…") | 68.4% |
| love | reaction ("love this / love it") | 21.9% |
| motivation | self-help / promotional | 21.0% |

**Inference.** A keyword scrape gives you a *vocabulary* sample, not a *population* sample. The negative words are used for politics, idiom, economics, brand customer care and sport as well as feeling. The positive words are used for politeness and marketing. Any "prevalence" figure built on keyword hits mixes all of these, and the mix differs by country (see 4.1) and by hour.

### 4.4 The label is mostly the keyword

| | "Depressive" rate (VADER ≤ 0) | Same, with the keyword masked |
|---|---|---|
| Negative-scrape tweets | 68.1% | 56.4% |
| Positive-scrape tweets | 15.5% | 39.4% |
| *e.g.* grief | 72.2% | 44.6% |
| *e.g.* love | 6.5% | 43.4% |
| *e.g.* happiness | 3.3% | 20.0% |

Hide the search word and the two sets converge (56% vs 39%, down from 68% vs 16%). The rest of each tweet differs much less than the labels suggested.

### 4.5 Explicit self-disclosure is rare and often ironic

Only **1.6%** of negative-scrape tweets contain a pattern like *"I was diagnosed…", "my depression", "my anxiety attack", "my therapist"*. In the positive set it is 0.04%. Even these are frequently jokes or memes: *"in my depression era"*, *"consumerism heals my depression /j"*, *"My skin is clear… My depression is gone"*. Self-disclosure tweets are somewhat more VADER-negative (78% vs 68%), but this is a small, noisy signal, not a diagnostic one.

## 5. What this data cannot tell us

- **Prevalence of depression in any country.** There are no clinical labels, the sample is a keyword sample, English-only, and English Twitter users are not representative populations (least of all in India, Nigeria, Kenya or Pakistan).
- **Change over time.** The negative scrape spans 7–24 May but is heavily weighted to the last two days (55k of 86k unique tweets). The positive scrape covers only ~42 hours (22–24 May). Daily volumes reflect scraper runs, not behaviour.
- **Individuals.** There is one tweet per user for most users (≈ 1.3 tweets/user), so nothing longitudinal per person. That rules out the user-level approaches (e.g. Coppersmith et al. 2014; De Choudhury et al. 2013) that actually work.
- **Event effects.** May 2023 included US Mental Health Awareness Month, the UK's Mental Health Awareness Week (theme: anxiety), and the US Surgeon General's loneliness advisory (3 May). All are visible in the text, but the sampling can't support before/after claims. Awareness hashtags appear in only 0.4% of negative tweets.

## 6. Directions worth pursuing

**With this dataset (cheap):**
1. **Reframe the paper/blog as "the language of distress", not "depression detection".** The defensible claims are 4.1 (voice differs by region, not tone), 4.2 (first-person and loneliness peak at night) and 4.3 and 4.4 (what keywords really capture).
2. **Replace VADER with a model trained for the task.** An off-the-shelf Twitter emotion/sentiment transformer (e.g. the CardiffNLP TweetEval family) separates *sadness* from *anger* from *fear*, handles "good grief" and sarcasm better, and needs no training. Re-run 4.1 and 4.2 with emotion categories instead of a single polarity.
3. **Hand-label ~500 tweets** for *personal distress / commentary / idiom / spam*. That gives a validated target and a real precision number for every claim above, and it's a weekend of work for three people.
4. **Topic structure inside "loneliness" at night vs day**, to see what people are actually saying at 1 a.m.

**With new data (if it ever becomes possible again):** Twitter/X's API is now paid and snscrape stopped working in mid-2023, so this exact scrape is no longer reproducible. That makes the existing corpus more valuable as a historical snapshot. Reddit and Bluesky are the realistic successors (Bluesky's firehose is open). A user-level design (self-reported diagnosis plus timeline, after Coppersmith 2014) is the standard way to get real depression signal.

**For the blog (planned interactive elements):**
1. *"The chart that fooled us"*: the original country bar chart, then the same bars recoloured by negative-keyword share. Ends on the standardised version, where the gap disappears.
2. *"Where is Mali?"*: a small table or map of raw location strings and where the old geocoder put them. Playful and memorable.
3. *"The 1 a.m. voice"*: an hour-of-day line of first-person share with CI band, plus a toggle for loneliness vs stress.
4. *"Mask the word"*: a runnable VADER demo. Type a tweet, see the score, then see it with the keyword hidden.
5. *"Good grief"*: a stacked bar of what each keyword is really used for.
6. A short, honest methods box: what changed and why.

## 7. Reproducibility

```
analysis/
  build_corpus.py      raw scrapes → out/corpus.parquet (dedupe, keyword, geocode, local time, VADER)
  evaluate_models.py   honest model evaluation → out/model_eval.json, out/top_features.csv
  analyze.py           sections 4.1–4.5 + the artefact checks → out/*.csv, out/summary.json
  keyword_usages.py    section 4.3 → out/h_keyword_usages.csv
  requirements.txt
```

Run from the repo root:

```bash
pip install -r analysis/requirements.txt
python analysis/build_corpus.py . analysis/out
python analysis/evaluate_models.py analysis/out
python analysis/analyze.py . analysis/out
python analysis/keyword_usages.py analysis/out
```

Known limitations of the rebuild: the geocoder is ~95% precise on unique location strings, with remaining errors like "Fremantle, WA" being read as Washington State. Keyword attribution is regex-based. First-person and self-disclosure are pattern-based flags, not validated classifiers. Local time is available for 33% of tweets.
