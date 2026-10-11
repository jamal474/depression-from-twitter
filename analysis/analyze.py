"""
Step 3 — What can the data actually tell us?

All analyses are done WITHIN the keyword design rather than across it, because the
positive/negative keyword mix differs wildly between countries and hours.

Outputs (out/):
  a_original_artifact.csv     original country rates vs negative-keyword share
  b_keyword_circularity.csv   per-keyword "depressive" rate, raw vs keyword-masked
  c_country_adjusted.csv      corrected countries, keyword-standardised rates + CIs
  d_diurnal_neg.csv           local-hour profile inside the negative-keyword set
  d_diurnal_keyword.csv       relative time-of-day volume per negative keyword
  e_depression_senses.csv     what "depression" refers to
  f_self_disclosure.csv       first-person + explicit self-disclosure rates
  g_topics.csv                NMF topics of the negative-keyword tweets
  summary.json                key numbers quoted in INFERENCES.md
"""
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

DATA = Path(sys.argv[1] if len(sys.argv) > 1 else "data")
OUT = Path(sys.argv[2] if len(sys.argv) > 2 else "out")
S = {}
rng = np.random.default_rng(0)

d = pd.read_parquet(OUT / "corpus.parquet")
d = d[d.clean.str.len() > 0]
d["neg_label"] = (d.vader <= 0).astype(int)          # the paper's "depressive" label
d["neg_label_masked"] = (d.vader_masked <= 0).astype(int)
d["neg_strict"] = (d.vader <= -0.05).astype(int)     # VADER's own recommended cut-off
neg = d[(d.src == "neg") & ~d.dup_text].copy()
S["n_unique"] = len(d)
S["n_neg_dedup"] = len(neg)

# ------------------------------------------------------------------ A. the artefact
# Needs the original pipeline's FinalResult.csv, which was removed from the repo in
# Oct 2026. Its outputs are kept in out/a_original_*.csv; this block only reruns
# if the file is put back in the data folder.
if (DATA / "FinalResult.csv").exists():
    F = pd.read_csv(DATA / "FinalResult.csv")
    P = pd.read_csv(DATA / "pos_rawdata.csv"); N = pd.read_csv(DATA / "neg_rawdata.csv")
    P["src"], N["src"] = "pos", "neg"
    R = pd.concat([N, P])
    R["t"] = pd.to_datetime(R["Date Created"], utc=True).dt.strftime("%H:%M:%S")
    F = F.merge(R.drop_duplicates(["User", "t"])[["User", "t", "src", "location"]]
                .rename(columns={"location": "raw_location"}),
                left_on=["User", "Time"], right_on=["User", "t"], how="left")
    a = F.groupby("location").agg(n=("src", "size"),
                                  neg_keyword_share=("src", lambda s: (s == "neg").mean()),
                                  paper_depression_rate=("Modelled Dep. Status", "mean"))
    top_raw = F.groupby("location").raw_location.agg(lambda s: "; ".join(s.value_counts().head(3).index))
    a["top_raw_locations"] = top_raw
    a.sort_values("paper_depression_rate").to_csv(OUT / "a_original_artifact.csv")
    S["artifact_country_corr"] = float(a.neg_keyword_share.corr(a.paper_depression_rate))
    F["h"] = pd.to_datetime(F.Time, format="%H:%M:%S").dt.hour
    ah = F.groupby("h").agg(neg_keyword_share=("src", lambda s: (s == "neg").mean()),
                            paper_rate=("Modelled Dep. Status", "mean"), n=("src", "size"))
    ah.to_csv(OUT / "a_original_hourly.csv")
    S["artifact_hour_corr"] = float(ah.neg_keyword_share.corr(ah.paper_rate))
    S["artifact_model_vs_vader_agreement"] = float(((F.Sentiment <= 0).astype(int) == F["Modelled Dep. Status"]).mean())
    # The paper's rate split by keyword list: if these two numbers are roughly constant,
    # each country's rate is just  share_neg * rate_neg + (1 - share_neg) * rate_pos.
    bys = F.groupby("src")["Modelled Dep. Status"].agg(["mean", "size"])
    bys.to_csv(OUT / "a_original_by_src.csv")
    S["artifact_rate_neg_src"] = float(bys.loc["neg", "mean"])
    S["artifact_rate_pos_src"] = float(bys.loc["pos", "mean"])
    # Re-geocode the paper's own rows with the new geocoder: how much of each label is right?
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from build_corpus import Geocoder
    geo = Geocoder(DATA / "worldcities.csv")
    F["new_country"] = [geo(x)[0] for x in F.raw_location]
    rel = F.groupby("location").apply(lambda x: pd.Series({
        "n": len(x),
        "really_that_country": (x.new_country.str.lower() == x.name).mean(),
        "actually_united_states": (x.new_country == "United States").mean(),
        "unresolved": x.new_country.isna().mean(),
        "top_raw_locations": "; ".join(x.raw_location.value_counts().head(3).index)}))
    rel.sort_values("really_that_country").to_csv(OUT / "a_original_relabel.csv")

else:  # keep the numbers computed while the file still existed
    prev = OUT / "summary.json"
    if prev.exists():
        S.update({k: v for k, v in json.loads(prev.read_text()).items() if k.startswith("artifact_")})

# ------------------------------------------------------------------ B. circularity
b = d[d.keyword != "unmatched"].groupby(["src", "keyword"]).agg(
    n=("neg_label", "size"), depressive_rate=("neg_label", "mean"),
    depressive_rate_keyword_masked=("neg_label_masked", "mean"),
    mean_vader=("vader", "mean"), mean_vader_masked=("vader_masked", "mean")).reset_index()
b = b[b.src != "both"].sort_values("depressive_rate")
b.to_csv(OUT / "b_keyword_circularity.csv", index=False)

# ------------------------------------------------------------------ C. countries
DEVELOPED = {"United States", "United Kingdom", "Canada", "Australia", "Ireland", "Germany",
             "France", "Spain", "Netherlands", "Japan", "New Zealand", "Italy", "Sweden"}
MIN_N = 250
kw_mix = neg.keyword.value_counts(normalize=True)  # global keyword mix (neg set)


def standardised(x: pd.DataFrame, col: str) -> float:
    """Directly standardise a rate to the global keyword mix."""
    r = x.groupby("keyword")[col].mean()
    w = kw_mix.reindex(r.index).fillna(0)
    return float((r * w).sum() / w.sum())


def user_boot(x, col, B=400):
    """Cluster bootstrap over users (prolific accounts can't dominate the CI)."""
    users = x.user.unique()
    g = {u: i for u, i in x.groupby("user").groups.items()}
    vals = []
    for _ in range(B):
        samp = rng.choice(users, len(users))
        idx = np.concatenate([g[u] for u in samp])
        vals.append(standardised(x.loc[idx], col))
    return np.percentile(vals, [2.5, 97.5])


rows = []
geo_neg = neg[neg.country.notna()]
all_geo = d[d.country.notna() & ~d.dup_text]
for c, x in geo_neg.groupby("country"):
    if len(x) < MIN_N:
        continue
    lo, hi = user_boot(x, "neg_label")
    flo, fhi = user_boot(x, "first_person")
    allc = all_geo[all_geo.country == c]
    rows.append(dict(country=c, group="developed" if c in DEVELOPED else "developing",
                     n_neg_tweets=len(x), n_users=x.user.nunique(),
                     share_of_country_tweets_from_neg_keywords=(allc.src == "neg").mean(),
                     naive_rate_all_tweets=allc.neg_label.mean(),
                     neg_rate_std=standardised(x, "neg_label"), neg_lo=lo, neg_hi=hi,
                     neg_rate_masked_std=standardised(x, "neg_label_masked"),
                     first_person_std=standardised(x, "first_person"), fp_lo=flo, fp_hi=fhi,
                     mean_vader=x.vader.mean()))
c = pd.DataFrame(rows).sort_values("neg_rate_std")
c.to_csv(OUT / "c_country_adjusted.csv", index=False)
gx = geo_neg.assign(group=np.where(geo_neg.country.isin(DEVELOPED), "developed", "developing"))
grp = {}
for gname, x in gx.groupby("group"):
    lo, hi = user_boot(x, "neg_label", B=300)
    grp[gname] = dict(n=len(x), rate=standardised(x, "neg_label"), ci=[lo, hi],
                      first_person=standardised(x, "first_person"))
S["group_rates_within_neg_keywords"] = grp

# Logistic regression with keyword fixed effects as a cross-check
try:
    import statsmodels.formula.api as smf
    m = gx.copy()
    m["dev"] = (m.group == "developed").astype(int)
    fit = smf.logit("neg_label ~ dev + C(keyword)", data=m).fit(
        disp=0, cov_type="cluster", cov_kwds={"groups": pd.factorize(m.user)[0]})
    S["logit_developed_OR"] = float(np.exp(fit.params["dev"]))
    S["logit_developed_OR_CI"] = [float(v) for v in np.exp(fit.conf_int().loc["dev"])]
    S["logit_developed_p"] = float(fit.pvalues["dev"])
except Exception as e:
    S["logit_error"] = str(e)

# ------------------------------------------------------------------ D. local time-of-day
# The scrape did not cover every UTC hour equally (it pulled the most recent N tweets per
# keyword, in several runs). So we compare tweets *posted in the same UTC hour* but in
# different local hours (different time zones), with keyword and country held fixed:
#     y ~ C(local 3h bin) + C(keyword) + C(country) + C(UTC date-hour)
# A linear probability model with user-clustered SEs keeps the coefficients readable
# as percentage-point differences vs. the 09-12 local reference bin.
import statsmodels.formula.api as smf
full = neg[neg.local_hour.notna() & neg.country.notna()].copy()
full["hb"] = (full.local_hour // 3 * 3).astype(int).astype(str).str.zfill(2)
full["utc_slot"] = full.created_utc.dt.strftime("%m%d%H")
top_c = full.country.value_counts()
full["cty"] = np.where(full.country.isin(top_c[top_c >= 200].index), full.country, "other")
S["n_full_local"] = len(full)
raw = full.groupby("hb").agg(n=("neg_label", "size"), depressive_rate=("neg_label", "mean"),
                             first_person=("first_person", "mean"), mean_vader=("vader", "mean"))
diur = []
for y in ["first_person", "neg_label"]:
    f = smf.ols(f"{y} ~ C(hb, Treatment('09')) + C(keyword) + C(cty) + C(utc_slot)", data=full.assign(**{y: full[y].astype(float)})
                ).fit(cov_type="cluster", cov_kwds={"groups": pd.factorize(full.user)[0]})
    for hb in sorted(full.hb.unique()):
        k = f"C(hb, Treatment('09'))[T.{hb}]"
        est = f.params.get(k, 0.0)
        lo, hi = (f.conf_int().loc[k].tolist() if k in f.params else [0.0, 0.0])
        diur.append(dict(outcome=y, local_bin=hb, adj_diff_vs_09=est, lo=lo, hi=hi))
dd_ = pd.DataFrame(diur)
raw.join(dd_[dd_.outcome == "first_person"].set_index("local_bin")[["adj_diff_vs_09", "lo", "hi"]]
         .add_prefix("fp_")).join(dd_[dd_.outcome == "neg_label"].set_index("local_bin")[["adj_diff_vs_09", "lo", "hi"]]
         .add_prefix("neg_")).to_csv(OUT / "d_diurnal_neg.csv")
# Which negative keywords lean late-night (22:00-04:00 local)? Same fixed-effects logic.
full["late"] = ((full.local_hour >= 22) | (full.local_hour < 4)).astype(float)
kw = []
for k in sorted(full.keyword.unique()):
    full["is_k"] = (full.keyword == k).astype(float)
    f = smf.ols("is_k ~ late + C(cty) + C(utc_slot)", data=full).fit(
        cov_type="cluster", cov_kwds={"groups": pd.factorize(full.user)[0]})
    base = full.is_k.mean()
    kw.append(dict(keyword=k, n=int(full.is_k.sum()), base_share=base, late_diff_pp=f.params["late"],
                   lo=f.conf_int().loc["late", 0], hi=f.conf_int().loc["late", 1],
                   late_relative=(base + f.params["late"]) / base))
pd.DataFrame(kw).sort_values("late_diff_pp").to_csv(OUT / "d_diurnal_keyword.csv", index=False)

# ------------------------------------------------------------------ E. senses of "depression"
dep = neg[neg.keyword == "depression"].copy()
low = dep.clean.str.lower()
senses = {
    "economic / historical": r"great depression|economic depression|recession|economy|depression[- ]era|1930s|stock market",
    "weather": r"tropical depression|low pressure|cyclone|storm",
    "first-person (my/I + depression)": r"\b(my|i have|i've had|i'm|i am|i was|i get|i got|i suffer|i struggle|i deal)\b[^.!?]{0,40}depress",
    "awareness / advocacy": r"awareness|mental ?health ?(month|awareness)|#mentalhealth|stigma|support|help ?line|resources|reach out",
    "medical / clinical": r"diagnos|medication|antidepress|therap|treatment|clinical|disorder|ssri|psychiatr",
    "postpartum": r"postpartum|post-partum|postnatal",
}
e = pd.DataFrame({k: low.str.contains(v, regex=True) for k, v in senses.items()})
e_out = pd.DataFrame({"share": e.mean(), "n": e.sum()}).sort_values("share", ascending=False)
e_out.loc["none of the above"] = [(~e.any(axis=1)).mean(), (~e.any(axis=1)).sum()]
e_out.to_csv(OUT / "e_depression_senses.csv")

# ------------------------------------------------------------------ F. self-disclosure
DISCLOSE = re.compile(
    r"\b(i('m| am)|i (have|had|was|got|'ve been|have been|was recently)|my)\b[^.!?\n]{0,30}"
    r"\b(diagnos\w*|depression|depressed|anxiety disorder|anxiety attack|panic attack|ptsd|bipolar|"
    r"suicidal|therapist|antidepressant\w*)", re.I)
d["self_disclosure"] = d.clean.str.contains(DISCLOSE)
f = d[~d.dup_text].groupby(["src", "keyword"]).agg(
    n=("self_disclosure", "size"), first_person=("first_person", "mean"),
    self_disclosure=("self_disclosure", "mean"),
    depressive_rate=("neg_label", "mean")).reset_index()
f = f[f.src != "both"]
f.to_csv(OUT / "f_self_disclosure.csv", index=False)
S["self_disclosure_overall_neg"] = float(d[(d.src == "neg") & ~d.dup_text].self_disclosure.mean())
S["self_disclosure_overall_pos"] = float(d[(d.src == "pos") & ~d.dup_text].self_disclosure.mean())
# Is VADER negativity even associated with disclosure?
nn = d[(d.src == "neg") & ~d.dup_text]
S["neg_rate_given_disclosure"] = float(nn[nn.self_disclosure].neg_label.mean())
S["neg_rate_given_no_disclosure"] = float(nn[~nn.self_disclosure].neg_label.mean())
S["disclosure_examples"] = nn[nn.self_disclosure].sample(8, random_state=3).clean.str[:160].tolist()

# ------------------------------------------------------------------ G. topics
from sklearn.decomposition import NMF
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
stems = r"\b\w*(?:sad|lonel|despair|anguish|grie|anxi|stress|depress|isolat|helpless)"
tv = TfidfVectorizer(max_df=0.3, min_df=15, stop_words=list(ENGLISH_STOP_WORDS | {"amp", "just", "like", "don", "people", "know"}),
                     ngram_range=(1, 2), token_pattern=r"(?u)\b[a-zA-Z][a-zA-Z]+\b")
txt = neg.clean.str.lower().str.replace(stems + r"\w*", " ", regex=True)
X = tv.fit_transform(txt)
nmf = NMF(n_components=12, random_state=0, init="nndsvda", max_iter=400)
W = nmf.fit_transform(X)
terms = tv.get_feature_names_out()
neg["topic"] = W.argmax(1)
neg["topic_strength"] = W.max(1)
rows = []
for k, comp in enumerate(nmf.components_):
    sub = neg[neg.topic == k]
    rows.append(dict(topic=k, top_terms=", ".join(terms[comp.argsort()[::-1][:12]]),
                     share=len(sub) / len(neg), depressive_rate=sub.neg_label.mean(),
                     first_person=sub.first_person.mean(),
                     top_keywords=", ".join(sub.keyword.value_counts().head(3).index),
                     example=sub.sort_values("topic_strength").clean.iloc[-1][:160]))
pd.DataFrame(rows).sort_values("share", ascending=False).to_csv(OUT / "g_topics.csv", index=False)
neg[["tweet_id", "topic"]].to_parquet(OUT / "neg_topics.parquet")

# ------------------------------------------------------------------ H. misc numbers
S["share_with_mh_awareness_tag"] = float(neg.clean.str.contains(r"mental ?health ?awareness|#mhaw|#mentalhealthmonth|#mentalhealthawareness", case=False).mean())
S["share_replies_neg"] = float(neg.is_reply.mean())
S["dup_text_share"] = float(d.dup_text.mean())
S["geocoded_share"] = float(d.country.notna().mean())
S["n_with_local_time"] = int(d.local_hour.notna().sum())
(OUT / "summary.json").write_text(json.dumps(S, indent=2, default=str))
print(json.dumps(S, indent=2, default=str))
