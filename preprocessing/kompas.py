from __future__ import annotations
import argparse, re
from pathlib import Path
import pandas as pd
from functools import lru_cache
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from nltk.stem import PorterStemmer

INPUT_PATH  = Path("dataset/kompas.csv")
OUTPUT_PATH = Path("dataset_clean/kompas_clean.csv")

_URL_RE   = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)
_MENTION  = re.compile(r"@[\w_]+"); _HASHTAG = re.compile(r"#")
_NONALPHA = re.compile(r"[^a-z\s]")           # huruf a-z saja (angka & simbol hilang)
_MULTISP  = re.compile(r"\s+"); _REP3 = re.compile(r"(.)\1{2,}")
_VOWEL    = re.compile(r"[aeiou]"); _DIGIT_ONLY = re.compile(r"^\d+$")
_ENWORD   = re.compile(r"^[a-z]+$")           # kandidat kata Inggris (huruf latin saja)

COMMON_TEXT_COLS = ['content','text','isi','artikel','judul','title','body','description']

# Sastrawi + Porter
STOPWORDS = set(StopWordRemoverFactory().get_stop_words())
_id_stemmer = StemmerFactory().create_stemmer()
_en_stemmer = PorterStemmer()

def _normalize(t: str) -> str:
    if not isinstance(t, str): t = "" if t is None else str(t)
    t = t.lower()
    t = _URL_RE.sub(" ", t)
    t = _MENTION.sub(" ", t)
    t = _HASHTAG.sub(" ", t)
    t = _NONALPHA.sub(" ", t)
    return _MULTISP.sub(" ", t).strip()

def _is_noise(w: str, min_len: int = 3) -> bool:
    return (
        len(w) < min_len or
        not _VOWEL.search(w) or
        _REP3.search(w) is not None or
        _DIGIT_ONLY.match(w) is not None
    )

@lru_cache(maxsize=200_000)
def _stem_hybrid(tok: str) -> str:
    # 1) coba Sastrawi dulu
    s_id = _id_stemmer.stem(tok)
    if s_id != tok:
        return s_id
    # 2) jika tidak berubah & terlihat English, pakai Porter
    if _ENWORD.match(tok) and tok.endswith(("ing","ed","tion","s","es","ers","ies")):
        return _en_stemmer.stem(tok)
    return tok

def tokenize(t: str) -> list[str]:
    return _normalize(t).split()

def preprocess_text(t: str) -> list[str]:
    toks = tokenize(t)
    # stopword removal pra-stem
    toks = [w for w in toks if w not in STOPWORDS]
    # stemming hybrid
    toks = [_stem_hybrid(w) for w in toks]
    # stopword removal pasca-stem + noise filter
    toks = [w for w in toks if w not in STOPWORDS and not _is_noise(w)]
    return toks

def _pick_col(df: pd.DataFrame, forced: str | None):
    if forced and forced in df.columns: return forced
    low = {c.lower(): c for c in df.columns}
    for k in COMMON_TEXT_COLS:
        if k in low: return low[k]
    obj = df.select_dtypes(include="object").columns
    return obj[0] if len(obj) else df.columns[0]

def run(text_col: str | None = None):
    df = pd.read_csv(INPUT_PATH)
    col = _pick_col(df, text_col)
    clean_tokens = [preprocess_text(x) for x in df[col].tolist()]
    pd.DataFrame({"clean_tokens": clean_tokens}).to_csv(OUTPUT_PATH, index=False)
    print(f"[✓] Kompas selesai: {len(clean_tokens)} baris. Contoh: 'monitoring' -> '{_stem_hybrid('monitoring')}'")

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--text-col", type=str, default=None)
    run(ap.parse_args().text_col)
