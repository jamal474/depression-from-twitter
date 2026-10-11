"""
Step 5 — Chart-ready CSVs for the blog post (breadboard, sabo.sh).

Reads the result tables in out/ and writes small, rounded, human-labelled CSVs to
out/blog/. Every number in the post's charts comes from here.
Usage: python analysis/blog_csvs.py analysis/out
"""
import json
import sys
from pathlib import Path

import pandas as pd

OUT = Path(sys.argv[1] if len(sys.argv) > 1 else "out")
B = OUT / "blog"
B.mkdir(exist_ok=True)
pct = lambda s, nd=1: (100 * s).round(nd)

# 1. The paper's country chart, with the negative-keyword share next to it
a = pd.read_csv(OUT / "a_original_artifact.csv").set_index("location")
order = ["united states", "canada", "united kingdom", "japan", "australia",
         "india", "nigeria", "puerto rico", "philippines", "mali"]
names = {"united states": "USA", "canada": "Canada", "united kingdom": "UK", "japan": "Japan",
         "australia": "Australia", "india": "India", "nigeria": "Nigeria",
         "puerto rico": "Puerto Rico", "philippines": "Philippines", "mali": "Mali"}
pd.DataFrame({"country": [names[c] for c in order],
              "Labelled depressive": pct(a.loc[order, "paper_depression_rate"]).values,
              "From the sad-word search": pct(a.loc[order, "neg_keyword_share"]).values}
             ).to_csv(B / "paper-countries.csv", index=False)

# 2. The paper's time-of-day chart (UTC hours), same overlay
h = pd.read_csv(OUT / "a_original_hourly.csv")
pd.DataFrame({"hour (UTC)": [f"{x:02d}:00" for x in h.h],
              "Labelled depressive": pct(h.paper_rate).values,
              "From the sad-word search": pct(h.neg_keyword_share).values}
             ).to_csv(B / "paper-hours.csv", index=False)

# 3. Re-geocoding the paper's own rows (table in the post)
r = pd.read_csv(OUT / "a_original_relabel.csv").set_index("location").loc[order]
pd.DataFrame({"label": [names[c] for c in order], "tweets": r.n.astype(int).values,
              "really that country %": pct(r.really_that_country, 0).values,
              "actually US %": pct(r.actually_united_states, 0).values,
              "most common location strings": r.top_raw_locations.values}
             ).to_csv(B / "relabel.csv", index=False)

# 4. Label with and without the search keyword
k = pd.read_csv(OUT / "b_keyword_circularity.csv")
for src, fname in [("neg", "mask-negative.csv"), ("pos", "mask-positive.csv")]:
    x = k[k.src == src].sort_values("depressive_rate", ascending=False)
    pd.DataFrame({"keyword": x.keyword.values,
                  "As scored": pct(x.depressive_rate).values,
                  "Keyword hidden": pct(x.depressive_rate_keyword_masked).values}
                 ).to_csv(B / fname, index=False)

# 5. What the keywords are used for
u = pd.read_csv(OUT / "h_keyword_usages.csv")
pick = [('hope', 'well-wishing ("I hope / hope you")', "hope: “I hope you”"),
        ('grief', 'idiom "good grief"', "grief: “good grief”"),
        ('love', 'reaction ("love this / love it")', "love: “love this”"),
        ('motivation', 'self-help / promo', "motivation: promo"),
        ('depression', 'Great Depression / economy', "depression: economy"),
        ('anguish', 'customer-service template', "anguish: support desk"),
        ('isolation', 'social isolation (public health)', "isolation: health news"),
        ('anxiety', 'casual ("gives me anxiety")', "anxiety: casual")]
rows = []
for kw, usage, label in pick:
    m = u[(u.keyword == kw) & (u.usage == usage)].iloc[0]
    rows.append({"usage": label, "Share of the keyword's tweets": round(100 * m.share, 1)})
pd.DataFrame(rows).to_csv(B / "keyword-uses.csv", index=False)

# 6. Countries, like for like (negative-keyword tweets, standardised keyword mix)
c = pd.read_csv(OUT / "c_country_adjusted.csv")
cn = {"United States": "USA", "United Kingdom": "UK"}
c["label"] = c.country.replace(cn)
c = pd.concat([c[c.group == "developed"].sort_values("first_person_std", ascending=False),
               c[c.group == "developing"].sort_values("first_person_std", ascending=False)])
pd.DataFrame({"country": c.label.values,
              "Negative tone": pct(c.neg_rate_std).values,
              "Written in the first person": pct(c.first_person_std).values,
              "tone 95% low": pct(c.neg_lo).values, "tone 95% high": pct(c.neg_hi).values,
              "first person 95% low": pct(c.fp_lo).values, "first person 95% high": pct(c.fp_hi).values,
              "tweets": c.n_neg_tweets.values, "group": c.group.values}
             ).to_csv(B / "countries.csv", index=False)

# 7. Time of day, adjusted (percentage points vs 09:00-12:00 local)
d = pd.read_csv(OUT / "d_diurnal_neg.csv", dtype={"hb": str})
# start the day at 06:00 so the night sits together at the right-hand end
d = d.set_index("hb").loc[["06", "09", "12", "15", "18", "21", "00", "03"]].reset_index()
labels = {"00": "00–03", "03": "03–06", "06": "06–09", "09": "09–12", "12": "12–15",
          "15": "15–18", "18": "18–21", "21": "21–24"}
pd.DataFrame({"local time": [labels[x] for x in d.hb],
              "First person": (100 * d.fp_adj_diff_vs_09).round(1).values,
              "Negative tone": (100 * d.neg_adj_diff_vs_09).round(1).values,
              "first person 95% low": (100 * d.fp_lo).round(1).values,
              "first person 95% high": (100 * d.fp_hi).round(1).values,
              "tone 95% low": (100 * d.neg_lo).round(1).values,
              "tone 95% high": (100 * d.neg_hi).round(1).values}
             ).to_csv(B / "time-of-day.csv", index=False)

# 8. Which keywords lean late-night
kk = pd.read_csv(OUT / "d_diurnal_keyword.csv")
kk = kk[kk.keyword != "unmatched"].sort_values("late_relative", ascending=False)
pd.DataFrame({"keyword": kk.keyword.values,
              "Late-night lean": (100 * (kk.late_relative - 1)).round(0).astype(int).values,
              "95% low (pp)": (100 * kk.lo).round(1).values, "95% high (pp)": (100 * kk.hi).round(1).values}
             ).to_csv(B / "late-night.csv", index=False)

# 9. Model comparison (table)
m = json.loads((OUT / "model_eval.json").read_text())
pd.DataFrame([
    ("Always guess the commoner label", m["baseline_majority"]),
    ("Which keyword list it came from (no text)", m["baseline_src_only"]),
    ("Naive Bayes", m["single_models_test_acc"]["NaiveBayes"]),
    ("Random forest", m["single_models_test_acc"]["RandomForest"]),
    ("Linear SVC", m["single_models_test_acc"]["LinearSVC"]),
    ("Logistic regression", m["single_models_test_acc"]["LogReg"]),
    ("Best ensemble, chosen on validation", m["ensemble"]["its_test_acc"]),
], columns=["model", "accuracy %"]).assign(**{"accuracy %": lambda x: (100 * x["accuracy %"]).round(1)}
).to_csv(B / "models.csv", index=False)

# 10. Inputs for the runnable "mix" program
s = json.loads((OUT / "summary.json").read_text())
pd.DataFrame({"country": [names[c] for c in order],
              "neg_share": a.loc[order, "neg_keyword_share"].round(4).values,
              "paper_rate": a.loc[order, "paper_depression_rate"].round(4).values}
             ).to_csv(B / "mix-inputs.csv", index=False)
print("rate_neg", round(s["artifact_rate_neg_src"], 4), "rate_pos", round(s["artifact_rate_pos_src"], 4))
for f in sorted(B.glob("*.csv")):
    print("==", f.name); print(f.read_text())
