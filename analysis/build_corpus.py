"""
Step 1 — Build a clean, analysis-ready corpus from the RAW scrapes.

Why rebuild instead of using Dataset1/Dataset2?
  * Dataset1/2 were round-tripped through Excel: every Tweet ID became 1.66E+18,
    the date was dropped and only a UTC clock time survived.
  * Location was resolved with a token-level city lookup that produced large
    false-positive countries (e.g. "San Francisco" -> Mali via the town "San",
    "Texas, USA" -> Japan via the city "Usa", "Florida" -> Puerto Rico).
  * Which keyword list (positive vs negative) a tweet came from was lost,
    and that turned out to be the main driver of every downstream result.

Outputs  out/corpus.parquet  with one row per unique tweet:
  tweet_id, user, created_utc, text, raw_location,
  src ('neg' | 'pos'), keyword (query keyword found in text, or 'unmatched'),
  country, region, geo_level ('city'|'state'|'country'|None), lat, lng, tz,
  local_time, local_hour, vader, vader_masked, is_retweet_like, first_person
"""
from __future__ import annotations

import re
import sys
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd

DATA = Path(sys.argv[1] if len(sys.argv) > 1 else "data")
OUT = Path(sys.argv[2] if len(sys.argv) > 2 else "out")
OUT.mkdir(exist_ok=True, parents=True)

KW_NEG = ["sadness", "loneliness", "despair", "anguish", "grief",
          "anxiety", "stress", "depression", "isolation", "helplessness"]
KW_POS = ["happiness", "joy", "love", "gratitude", "hope",
          "success", "motivation", "inspiration", "excitement", "optimism"]
# Twitter search matches inflections loosely, so match on stems.
STEM = {"sadness": r"sadness|\bsad\b", "loneliness": r"lonel", "despair": r"despair",
        "anguish": r"anguish", "grief": r"grie[fv]", "anxiety": r"anxi",
        "stress": r"stress", "depression": r"depress", "isolation": r"isolat",
        "helplessness": r"helpless", "happiness": r"happ(?:y|i)", "joy": r"\bjoy",
        "love": r"\blov", "gratitude": r"gratitud|grateful", "hope": r"\bhop(?:e|ing)",
        "success": r"succe", "motivation": r"motivat", "inspiration": r"inspir",
        "excitement": r"excit", "optimism": r"optimis"}


# ----------------------------------------------------------------------------
# 1. Load + dedupe
# ----------------------------------------------------------------------------
def load_raw() -> pd.DataFrame:
    frames = []
    for name, src in [("neg_rawdata.csv", "neg"), ("pos_rawdata.csv", "pos")]:
        d = pd.read_csv(DATA / name, dtype={"Tweet ID": "int64"})
        d["src"] = src
        frames.append(d)
    d = pd.concat(frames, ignore_index=True)
    d = d.rename(columns={"User": "user", "Date Created": "created", "Tweet ID": "tweet_id",
                          "Tweet": "text", "location": "raw_location"})
    d["created_utc"] = pd.to_datetime(d["created"], utc=True)
    # A tweet that matched both a positive and a negative query is genuinely ambiguous;
    # keep it once, flagged as 'both'.
    both = d.groupby("tweet_id")["src"].nunique()
    d = d.drop_duplicates("tweet_id").copy()
    d.loc[d.tweet_id.isin(both[both > 1].index), "src"] = "both"
    d["text"] = d["text"].astype(str)
    return d[["tweet_id", "user", "created_utc", "text", "raw_location", "src"]]


def attribute_keyword(d: pd.DataFrame) -> pd.Series:
    low = d.text.str.lower()
    out = pd.Series("unmatched", index=d.index)
    for src, kws in [("neg", KW_NEG), ("pos", KW_POS), ("both", KW_NEG + KW_POS)]:
        m = d.src == src
        for kw in reversed(kws):  # earlier keywords win ties
            hit = m & low.str.contains(STEM[kw], regex=True)
            out[hit] = kw
    return out


# ----------------------------------------------------------------------------
# 2. Geocoder — conservative, segment-level, ambiguity-aware
# ----------------------------------------------------------------------------
US_STATES = {
    "AL": "Alabama", "AK": "Alaska", "AZ": "Arizona", "AR": "Arkansas", "CA": "California",
    "CO": "Colorado", "CT": "Connecticut", "DE": "Delaware", "FL": "Florida", "GA": "Georgia",
    "HI": "Hawaii", "ID": "Idaho", "IL": "Illinois", "IN": "Indiana", "IA": "Iowa",
    "KS": "Kansas", "KY": "Kentucky", "LA": "Louisiana", "ME": "Maine", "MD": "Maryland",
    "MA": "Massachusetts", "MI": "Michigan", "MN": "Minnesota", "MS": "Mississippi",
    "MO": "Missouri", "MT": "Montana", "NE": "Nebraska", "NV": "Nevada", "NH": "New Hampshire",
    "NJ": "New Jersey", "NM": "New Mexico", "NY": "New York", "NC": "North Carolina",
    "ND": "North Dakota", "OH": "Ohio", "OK": "Oklahoma", "OR": "Oregon", "PA": "Pennsylvania",
    "RI": "Rhode Island", "SC": "South Carolina", "SD": "South Dakota", "TN": "Tennessee",
    "TX": "Texas", "UT": "Utah", "VT": "Vermont", "VA": "Virginia", "WA": "Washington",
    "WV": "West Virginia", "WI": "Wisconsin", "WY": "Wyoming", "DC": "District of Columbia"}
CA_PROV = {"ON": "Ontario", "BC": "British Columbia", "AB": "Alberta", "QC": "Quebec",
           "MB": "Manitoba", "SK": "Saskatchewan", "NS": "Nova Scotia", "NB": "New Brunswick",
           "NL": "Newfoundland and Labrador", "PE": "Prince Edward Island"}
AU_STATES = {"NSW": "New South Wales", "VIC": "Victoria", "QLD": "Queensland",
             "WA": "Western Australia", "SA": "South Australia", "TAS": "Tasmania",
             "ACT": "Australian Capital Territory", "NT": "Northern Territory"}

COUNTRY_ALIASES = {
    "usa": "United States", "us": "United States", "u.s.": "United States", "u.s.a.": "United States",
    "u.s.a": "United States", "united states": "United States", "united states of america": "United States",
    "america": "United States", "uk": "United Kingdom", "u.k.": "United Kingdom",
    "united kingdom": "United Kingdom", "england": "United Kingdom", "scotland": "United Kingdom",
    "wales": "United Kingdom", "northern ireland": "United Kingdom", "great britain": "United Kingdom",
    "britain": "United Kingdom", "gb": "United Kingdom", "naija": "Nigeria",
    "republic of the philippines": "Philippines", "pilipinas": "Philippines", "ph": "Philippines",
    "bharat": "India", "hindustan": "India", "south korea": "Korea, South", "korea": "Korea, South",
    "uae": "United Arab Emirates", "ksa": "Saudi Arabia", "nz": "New Zealand", "aus": "Australia",
    "pr": "Puerto Rico", "deutschland": "Germany", "españa": "Spain", "méxico": "Mexico",
    "brasil": "Brazil", "türkiye": "Turkey", "turkiye": "Turkey", "holland": "Netherlands",
    "the netherlands": "Netherlands", "ivory coast": "Côte d'Ivoire", "drc": "Congo (Kinshasa)",
}
# Single words that are also city names somewhere — never resolve these on their own.
BLOCKLIST = {"san", "usa", "earth", "home", "world", "everywhere", "heaven", "hell", "here",
             "there", "global", "worldwide", "online", "internet", "moon", "mars", "space",
             "she", "her", "he", "him", "they", "them", "love", "god", "bay", "of", "the",
             "city", "north", "south", "east", "west", "nowhere", "somewhere", "planet earth",
             "metaverse", "universe", "carolina", "your heart", "your mind", "lost", "dreamland"}


def _norm(s: str) -> str:
    s = unicodedata.normalize("NFKC", str(s))
    s = re.sub(r"[^\w\s,.'\-/|()]", " ", s)  # drop emoji / symbols
    return re.sub(r"\s+", " ", s).strip()


class Geocoder:
    def __init__(self, cities_csv: Path, min_pop: int = 50_000):
        w = pd.read_csv(cities_csv)
        w["population"] = w["population"].fillna(0)
        self.w = w
        self.countries = {c.lower(): c for c in w.country.unique()}
        self.countries.update({k: v for k, v in COUNTRY_ALIASES.items()})
        # country centroid = population-weighted mean of its cities (for country-level hits)
        g = w[w.population > 0].groupby("country")
        self.country_ll = {c: (np.average(x.lat, weights=x.population),
                               np.average(x.lng, weights=x.population)) for c, x in g}
        # admin regions (states/provinces) -> (country, lat, lng of largest city)
        big = w.sort_values("population", ascending=False)
        self.admin = {}
        for (country, admin), x in big.groupby(["country", "admin_name"], sort=False):
            if not isinstance(admin, str):
                continue
            key = admin.lower()
            if key in self.admin and self.admin[key][3] >= x.population.sum():
                continue
            self.admin[key] = (country, x.lat.iloc[0], x.lng.iloc[0], x.population.sum(), admin)
        for abbr, name in US_STATES.items():
            self.admin[abbr.lower() + "@us"] = self.admin.get(name.lower(), ("United States", 39.8, -98.6, 0, name))
        for abbr, name in CA_PROV.items():
            self.admin[abbr.lower() + "@ca"] = self.admin.get(name.lower(), ("Canada", 56, -96, 0, name))
        for abbr, name in AU_STATES.items():
            self.admin[abbr.lower() + "@au"] = self.admin.get(name.lower(), ("Australia", -25, 133, 0, name))
        # US/CA/AU state full names must win over same-named places elsewhere ("Georgia")
        self.admin_priority = {n.lower() for n in list(US_STATES.values()) + list(CA_PROV.values())
                               + list(AU_STATES.values())}
        # cities: keep only those above min_pop; resolve ambiguous names only if one
        # candidate is >= 5x bigger than the next.
        c = w[w.population >= min_pop].copy()
        c["k"] = c.city_ascii.str.lower()
        self.city = {}
        for k, x in c.sort_values("population", ascending=False).groupby("k", sort=False):
            if k in BLOCKLIST or len(k) < 4:
                continue
            if len(x) == 1 or x.population.iloc[0] >= 5 * x.population.iloc[1]:
                r = x.iloc[0]
                self.city[k] = (r.country, r.lat, r.lng, r.admin_name)
        self.cache = {}

    def __call__(self, raw):
        if not isinstance(raw, str) or not raw.strip():
            return (None, None, None, None, None)
        if raw in self.cache:
            return self.cache[raw]
        res = self._resolve(raw)
        self.cache[raw] = res
        return res

    def _resolve(self, raw):
        s = _norm(raw)
        segs = [x.strip(" .'-()") for x in re.split(r"[,|/]| - |\(|\)", s) if x.strip(" .'-()")]
        low = [x.lower() for x in segs]
        if not low or " ".join(low) in BLOCKLIST:
            return (None, None, None, None, None)
        country = None
        # (a) explicit country anywhere (rightmost wins)
        for x in reversed(low):
            if x in self.countries:
                country = self.countries[x]
                break
        # "Atlanta, Georgia" is the US state, not the country
        if country == "Georgia" and any(self.city.get(x, (None,))[0] == "United States" for x in low):
            country = "United States"
        # (b) state / province (full name or abbreviation following a comma)
        for i in range(len(segs) - 1, -1, -1):
            x, X = low[i], segs[i]
            cand = None
            if X.isupper() and 2 <= len(X) <= 3 and i > 0:
                for suf in ("@us", "@ca", "@au"):
                    if x + suf in self.admin:
                        a = self.admin[x + suf]
                        if country is None or a[0] == country:
                            cand = a
                            break
            elif x in self.admin and (x in self.admin_priority or country):
                a = self.admin[x]
                if country is None or a[0] == country:
                    cand = a
            if cand:
                # a city in front of the state gives a more precise point
                if (i > 0 and low[i - 1] in self.city and self.city[low[i - 1]][0] == cand[0]
                        and self.city[low[i - 1]][3] == cand[4]):
                    ct = self.city[low[i - 1]]
                    return (ct[0], ct[3], "city", ct[1], ct[2])
                return (cand[0], cand[4], "state", cand[1], cand[2])
        # (c) city segment
        for x in low:
            if x in self.city:
                ct = self.city[x]
                if country is None or ct[0] == country:
                    return (ct[0], ct[3], "city", ct[1], ct[2])
        if country:
            ll = self.country_ll.get(country, (None, None))
            return (country, None, "country", ll[0], ll[1])
        return (None, None, None, None, None)


# ----------------------------------------------------------------------------
# 3. Local time
# ----------------------------------------------------------------------------
# Countries spanning several time zones: only trust local time when we have a
# city/state-level point.
MULTI_TZ = {"United States", "Canada", "Australia", "Russia", "Brazil", "Mexico", "Indonesia",
            "Kazakhstan", "Mongolia", "Congo (Kinshasa)", "Chile", "Argentina"}


def add_local_time(d: pd.DataFrame) -> pd.DataFrame:
    from timezonefinder import TimezoneFinder
    tf = TimezoneFinder()
    ok = d.lat.notna() & ~((d.geo_level == "country") & d.country.isin(MULTI_TZ))
    ll = d.loc[ok, ["lat", "lng"]].astype(float).round(2)
    keys = ll.drop_duplicates()
    tzmap = {(r.lat, r.lng): tf.timezone_at(lat=r.lat, lng=r.lng) for r in keys.itertuples()}
    d["tz"] = None
    d.loc[ok, "tz"] = [tzmap[k] for k in zip(ll.lat, ll.lng)]
    local = pd.Series(pd.NaT, index=d.index, dtype="object")
    for tz, idx in d[d.tz.notna()].groupby("tz").groups.items():
        local[idx] = d.loc[idx, "created_utc"].dt.tz_convert(tz).dt.tz_localize(None)
    d["local_time"] = pd.to_datetime(local)
    d["local_hour"] = d.local_time.dt.hour + d.local_time.dt.minute / 60
    d["local_dow"] = d.local_time.dt.dayofweek
    return d


# ----------------------------------------------------------------------------
# 4. Text features
# ----------------------------------------------------------------------------
FIRST_PERSON = re.compile(r"\b(i|i'm|im|i've|ive|i'd|id|i'll|me|my|mine|myself)\b", re.I)
URL = re.compile(r"https?://\S+")
MENTION = re.compile(r"@\w+")


def add_text_features(d: pd.DataFrame) -> pd.DataFrame:
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    sia = SentimentIntensityAnalyzer()
    clean = d.text.str.replace(URL, " ", regex=True).str.replace(MENTION, " ", regex=True)
    d["clean"] = clean.str.replace(r"\s+", " ", regex=True).str.strip()
    d["vader"] = [sia.polarity_scores(t)["compound"] for t in d.clean]
    # Same score with every query keyword (and its stems) masked out. This tells
    # us how much of the "negativity" is just the search word itself.
    all_stems = "|".join(f"(?:{v})\\w*" for v in STEM.values())
    masked = d.clean.str.replace(all_stems, " ", regex=True, flags=re.I)
    d["vader_masked"] = [sia.polarity_scores(t)["compound"] for t in masked]
    d["first_person"] = d.clean.str.contains(FIRST_PERSON)
    d["n_words"] = d.clean.str.split().str.len()
    d["is_reply"] = d.text.str.startswith("@")
    return d


def main():
    d = load_raw()
    print("unique tweets:", len(d), d.src.value_counts().to_dict())
    d["keyword"] = attribute_keyword(d)
    geo = Geocoder(DATA / "worldcities.csv")
    g = pd.DataFrame([geo(x) for x in d.raw_location], index=d.index,
                     columns=["country", "region", "geo_level", "lat", "lng"])
    d = pd.concat([d, g], axis=1)
    d = add_local_time(d)
    d = add_text_features(d)
    # near-duplicate / spam marker: identical cleaned text posted more than once
    d["dup_text"] = d.duplicated("clean", keep="first")
    d.to_parquet(OUT / "corpus.parquet", index=False)
    print(d.geo_level.value_counts(dropna=False).to_dict())
    print(d.country.value_counts().head(20).to_string())


if __name__ == "__main__":
    main()
