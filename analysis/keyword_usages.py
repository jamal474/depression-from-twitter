"""
Step 4 — What do the search keywords actually mean in context?

Many tweets that contain a "negative" keyword are not about a negative feeling at all
(idioms, economics, customer-service templates, basketball), and many "positive" keyword
tweets are formulaic. Output: out/h_keyword_usages.csv
"""
import sys
from pathlib import Path

import pandas as pd

OUT = Path(sys.argv[1] if len(sys.argv) > 1 else "out")
d = pd.read_parquet(OUT / "corpus.parquet")
d = d[~d.dup_text]
low = d.clean.str.lower()

PATTERNS = {
    ("grief", 'idiom "good grief"'): r"good grief",
    ("anguish", "customer-service template"): r"understand your anguish|registered contact|private message|dm us|inconvenience",
    ("depression", "Great Depression / economy"): r"great depression|recession|econom|depression[- ]era|1930s",
    ("depression", '"depression nap/meal/era" meme'): r"depression (?:era|nap|meal|core|music|playlist)",
    ("isolation", 'basketball "isolation" play'): r"\biso\b|isolation play|isolation basketball|\bnba\b|lebron|jokic|celtics|lakers|nuggets|playoff",
    ("isolation", "social isolation (public health)"): r"social isolation|loneliness and isolation|isolation and loneliness|surgeon general|epidemic",
    ("loneliness", "Surgeon General / epidemic framing"): r"surgeon general|epidemic of loneliness|loneliness epidemic|public health",
    ("stress", '"stress test" (finance/engineering)'): r"stress test|stress-test|stress fracture|stress testing",
    ("anxiety", 'casual ("gives me anxiety")'): r"gives? me anxiety|giving me anxiety|anxiety (?:is|was) (?:through|thru) the roof|my anxiety could never",
    ("hope", 'well-wishing ("I hope / hope you")'): r"\bhope (?:you|u|ur|your|it|this|everyone|they|he|she|we)\b|\bi hope\b",
    ("love", 'reaction ("love this / love it")'): r"\blove (?:this|it|that|u|you|ya)\b",
    ("motivation", "self-help / promo"): r"#motivation|motivational|quote|follow|subscribe|link|mindset|hustle",
}
rows = []
for (kw, label), pat in PATTERNS.items():
    m = d.keyword == kw
    hit = low[m].str.contains(pat, regex=True)
    rows.append(dict(keyword=kw, usage=label, n_keyword_tweets=int(m.sum()), share=hit.mean(),
                     vader_depressive_rate_of_these=(d.loc[m, "vader"][hit.values] <= 0).mean()))
out = pd.DataFrame(rows)

# Tweets returned by the search that do not contain the keyword at all
STEMS = {"neg": "sad|lonel|despair|anguish|grie|anxi|stress|depress|isolat|helpless",
         "pos": "happ|joy|lov|gratitud|grateful|hope|succe|motivat|inspir|excit|optimis"}
for src, st in STEMS.items():
    x = d[d.src == src]
    um = x.keyword == "unmatched"
    out.loc[len(out)] = dict(keyword=f"({src} set)", usage="no keyword in text at all",
                             n_keyword_tweets=len(x), share=um.mean(),
                             vader_depressive_rate_of_these=(x.vader[um] <= 0).mean())
    out.loc[len(out)] = dict(keyword=f"({src} set)", usage="...of which keyword is in the username",
                             n_keyword_tweets=int(um.sum()),
                             share=(um & x.user.str.lower().str.contains(st)).sum() / um.sum(),
                             vader_depressive_rate_of_these=float("nan"))
out.to_csv(OUT / "h_keyword_usages.csv", index=False)
print(out.round(3).to_string())
