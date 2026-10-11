"""
Step 2 — Re-evaluate the classification set-up honestly.

The original target is   y = 1  if VADER compound <= 0   ("depressive")
(the paper says < 0, the code uses <= 0, so ~5% neutral tweets were labelled depressive).

Questions answered here:
  Q1  Can we reproduce ~85% with a plain TF-IDF linear model?
  Q2  How much does the random split leak (duplicate texts, same users)?
  Q3  How much of the label is explained by *which keyword list* the tweet came from?
  Q4  How much of the ensemble's "gain" survives a proper validation/test separation?
  Q5  What does the model actually learn? (top features)
Outputs: out/model_eval.json, out/top_features.csv
"""
import itertools
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import GroupShuffleSplit, train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier

OUT = Path(sys.argv[1] if len(sys.argv) > 1 else "out")
RNG = 42
res = {}

d = pd.read_parquet(OUT / "corpus.parquet")
d = d[d.clean.str.len() > 0].reset_index(drop=True)
d["y"] = (d.vader <= 0).astype(int)
res["n"] = len(d)
res["label_rate"] = d.y.mean()
res["share_exact_zero"] = float((d.vader == 0).mean())


def boot_ci(y, p, n=1000, seed=0):
    r = np.random.default_rng(seed)
    y, p = np.asarray(y), np.asarray(p)
    accs = [accuracy_score(y[i], p[i]) for i in (r.integers(0, len(y), len(y)) for _ in range(n))]
    return [float(np.percentile(accs, 2.5)), float(np.percentile(accs, 97.5))]


def tfidf():
    return TfidfVectorizer(sublinear_tf=True, min_df=3, max_df=0.5, ngram_range=(1, 2),
                           stop_words="english")


# ---------------------------------------------------------------- Q1 + Q2
# (a) paper-style random split, duplicates kept
tr, te = train_test_split(d.index, test_size=0.2, random_state=RNG)
vec = tfidf().fit(d.clean[tr])
Xtr, Xte = vec.transform(d.clean[tr]), vec.transform(d.clean[te])
svc = LinearSVC(C=0.5).fit(Xtr, d.y[tr])
p = svc.predict(Xte)
res["random_split_svc_acc"] = accuracy_score(d.y[te], p)
res["random_split_svc_f1"] = f1_score(d.y[te], p)
te_texts = set(d.clean[te]); tr_texts = set(d.clean[tr])
res["random_split_test_text_seen_in_train"] = float(np.mean([t in tr_texts for t in d.clean[te]]))
res["random_split_test_users_seen_in_train"] = float(d.user[te].isin(set(d.user[tr])).mean())

# (b) dedupe + group split by user (no author in both train and test)
dd = d[~d.dup_text].reset_index(drop=True)
gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=RNG)
tr2, te2 = next(gss.split(dd, groups=dd.user))
vec2 = tfidf().fit(dd.clean.iloc[tr2])
Xtr2, Xte2 = vec2.transform(dd.clean.iloc[tr2]), vec2.transform(dd.clean.iloc[te2])
ytr2, yte2 = dd.y.iloc[tr2].values, dd.y.iloc[te2].values
svc2 = LinearSVC(C=0.5).fit(Xtr2, ytr2)
p2 = svc2.predict(Xte2)
res["clean_split_svc_acc"] = accuracy_score(yte2, p2)
res["clean_split_svc_acc_ci"] = boot_ci(yte2, p2)

# ---------------------------------------------------------------- Q3 baselines
te_df = dd.iloc[te2]
res["baseline_majority"] = max(yte2.mean(), 1 - yte2.mean())
res["baseline_src_only"] = accuracy_score(yte2, (te_df.src != "pos").astype(int))
res["label_rate_by_src"] = dd.groupby("src").y.mean().to_dict()
res["label_rate_by_keyword"] = dd.groupby("keyword").y.mean().round(3).to_dict()
# How often does the VADER label flip when we hide the search keyword itself?
dd["y_masked"] = (dd.vader_masked <= 0).astype(int)
res["label_flip_rate_when_keyword_masked"] = float((dd.y != dd.y_masked).mean())
res["label_rate_masked_by_src"] = dd.groupby("src").y_masked.mean().to_dict()

# ---------------------------------------------------------------- Q4 ensemble selection
# Proper protocol: train / validation / test. Choose the best subset on VALIDATION,
# report on TEST, and compare with "choose on test" (what the original notebook did).
tr3, va3 = next(GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=1)
                .split(np.arange(len(tr2)), groups=dd.user.iloc[tr2]))
tr3, va3 = tr2[tr3], tr2[va3]
v3 = tfidf().fit(dd.clean.iloc[tr3])
X = {k: v3.transform(dd.clean.iloc[ix]) for k, ix in [("tr", tr3), ("va", va3), ("te", te2)]}
Y = {k: dd.y.iloc[ix].values for k, ix in [("tr", tr3), ("va", va3), ("te", te2)]}
models = {
    "NaiveBayes": MultinomialNB(alpha=0.3),
    "LogReg": LogisticRegression(C=4, max_iter=2000),
    "LinearSVC": LinearSVC(C=0.5),
    "RandomForest": RandomForestClassifier(n_estimators=200, n_jobs=-1, random_state=0,
                                           min_samples_leaf=2),
}
preds = {}
for name, m in models.items():
    m.fit(X["tr"], Y["tr"])
    preds[name] = {k: m.predict(X[k]) for k in ("va", "te")}
res["single_models_test_acc"] = {n: accuracy_score(Y["te"], p["te"]) for n, p in preds.items()}


def wae(names, split):
    """The paper's weighted-average ensemble: weight = 1/errors, vote in {-1,+1}."""
    errs = [np.sum(preds[n]["va"] != Y["va"]) for n in names]  # weights from VALIDATION
    w = np.array([1 / e for e in errs]); w /= w.sum()
    s = sum(wi * (2 * preds[n][split] - 1) for wi, n in zip(w, names))
    return (s >= 0).astype(int)


combos = [c for r in range(1, len(models) + 1) for c in itertools.combinations(models, r)]
va_acc = {c: accuracy_score(Y["va"], wae(c, "va")) for c in combos}
te_acc = {c: accuracy_score(Y["te"], wae(c, "te")) for c in combos}
best_on_va = max(va_acc, key=va_acc.get)
best_on_te = max(te_acc, key=te_acc.get)
res["ensemble"] = {
    "best_subset_chosen_on_validation": list(best_on_va),
    "its_test_acc": te_acc[best_on_va],
    "best_subset_chosen_on_test (leaky)": list(best_on_te),
    "its_test_acc (optimistic)": te_acc[best_on_te],
    "test_acc_range_over_all_subsets": [min(te_acc.values()), max(te_acc.values())],
    "best_single_test_acc": max(res["single_models_test_acc"].values()),
    "test_ci_halfwidth": float(np.diff(boot_ci(Y["te"], wae(best_on_va, "te")))[0] / 2),
}

# ---------------------------------------------------------------- Q5 features
lr = models["LogReg"]
coef = pd.Series(lr.coef_[0], index=v3.get_feature_names_out()).sort_values()
pd.concat([coef.head(25).rename("coef").reset_index().assign(direction="non-depressive"),
           coef.tail(25).rename("coef").reset_index().assign(direction="depressive")]
          ).to_csv(OUT / "top_features.csv", index=False)
try:
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    lex = SentimentIntensityAnalyzer().lexicon
    top = list(coef.head(50).index) + list(coef.tail(50).index)
    uni = [t for t in top if " " not in t]
    res["top100_unigrams_in_vader_lexicon"] = float(np.mean([t in lex for t in uni]))
except Exception as e:  # pragma: no cover
    res["lexicon_check_error"] = str(e)

(OUT / "model_eval.json").write_text(json.dumps(res, indent=2, default=lambda o: o.item() if hasattr(o, "item") else str(o)))
print(json.dumps(res, indent=2, default=str))
