"""
Preprocessing Kompas 30
Fokus: case folding, tokenization, stopword removal (Sastrawi)
Output: hanya 'clean_tokens' (tanpa angka)
"""
from __future__ import annotations
import argparse
import re
from pathlib import Path
import pandas as pd
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory

# ==== Konfigurasi ====
INPUT_PATH  = Path("dataset/kompas.csv")
OUTPUT_PATH = Path("dataset_clean/kompas_clean.csv")

# ==== Regex & helper ====
_URL_RE   = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)
_MENTION  = re.compile(r"@[\w_]+", re.UNICODE)
_HASHTAG  = re.compile(r"#")
_NONALPHA = re.compile(r"[^a-z\s]", re.UNICODE)  # hanya huruf a-z
_MULTISP  = re.compile(r"\s+")
_REP3     = re.compile(r"(.)\1{2,}")
_VOWEL    = re.compile(r"[aeiou]")
_DIGIT_ONLY = re.compile(r"^\d+$")

COMMON_TEXT_COLS = ['content','text','isi','artikel','judul','title','body','description']

# ==== Load stopwords dari Sastrawi ====
factory = StopWordRemoverFactory()
SASTRAWI_STOPWORDS = set(factory.get_stop_words())

def _normalize(text: str) -> str:
    if not isinstance(text, str):
        text = "" if text is None else str(text)
    t = text.lower()
    t = _URL_RE.sub(" ", t)
    t = _MENTION.sub(" ", t)
    t = _HASHTAG.sub(" ", t)
    t = _NONALPHA.sub(" ", t)
    t = _MULTISP.sub(" ", t).strip()
    return t

def _is_noise(token: str, min_len: int = 3) -> bool:
    if len(token) < min_len:
        return True
    if not _VOWEL.search(token):
        return True
    if _REP3.search(token):
        return True
    if _DIGIT_ONLY.match(token):
        return True
    return False

def tokenize(text: str) -> list[str]:
    return _normalize(text).split()

def preprocess_text(text: str) -> list[str]:
    toks = tokenize(text)
    toks = [t for t in toks if t not in SASTRAWI_STOPWORDS]
    toks = [t for t in toks if not _is_noise(t)]
    return toks

def pick_text_column(df: pd.DataFrame, forced: str | None):
    if forced and forced in df.columns:
        return forced
    lower_map = {c.lower(): c for c in df.columns}
    for k in COMMON_TEXT_COLS:
        if k in lower_map:
            return lower_map[k]
    return df.select_dtypes(include='object').columns[0]

def run(text_col: str | None = None):
    df = pd.read_csv(INPUT_PATH)
    col = pick_text_column(df, text_col)
    clean_tokens = [preprocess_text(x) for x in df[col].tolist()]
    out = pd.DataFrame({"clean_tokens": clean_tokens})
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUTPUT_PATH, index=False)
    print(f"[✓] Kompas selesai: {len(out)} baris, tanpa angka, pakai stopword Sastrawi.")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--text-col", type=str, default=None)
    args = ap.parse_args()
    run(args.text_col)
