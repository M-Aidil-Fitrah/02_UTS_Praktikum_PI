"""
Preprocessing Mojok
Fokus: case folding, tokenization, stopword removal (Sastrawi), stemming (Indo + English), dan noise filter
Output: hanya 'clean_tokens'
"""
from __future__ import annotations
import argparse
import re
from pathlib import Path
import pandas as pd
from functools import lru_cache
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from nltk.stem import PorterStemmer

# ==== Konfigurasi ====
INPUT_PATH  = Path("dataset/mojok.csv")
OUTPUT_PATH = Path("dataset_clean/mojok_clean_all.csv")

# ==== Regex & helper ====
_URL_RE   = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)
_MENTION  = re.compile(r"@[\w_]+", re.UNICODE)
_HASHTAG  = re.compile(r"#")
_NUMBER   = re.compile(r"\b\d+[\d.,]*\b", re.UNICODE)
_NONALPHA = re.compile(r"[^a-z\s]", re.UNICODE)
_MULTISP  = re.compile(r"\s+")
_REP3     = re.compile(r"(.)\1{2,}")
_VOWEL    = re.compile(r"[aeiou]")
_DIGIT_ONLY = re.compile(r"^\d+$")
_ENWORD   = re.compile(r"^[a-z]+$")

COMMON_TEXT_COLS = ['content','text','isi','artikel','judul','title','body','description']

# ==== Inisialisasi Sastrawi dan Porter ====
STOPWORDS = set(StopWordRemoverFactory().get_stop_words())
_id_stemmer = StemmerFactory().create_stemmer()
_en_stemmer = PorterStemmer()

# ==== Fungsi utilitas ====
def _normalize(text: str) -> str:
    """Lowercase + bersihkan URL, mention, hashtag, angka, simbol."""
    if not isinstance(text, str):
        text = "" if text is None else str(text)
    t = text.lower()
    t = _URL_RE.sub(" ", t)
    t = _MENTION.sub(" ", t)
    t = _HASHTAG.sub(" ", t)
    t = _NUMBER.sub(" ", t)
    t = _NONALPHA.sub(" ", t)
    t = _MULTISP.sub(" ", t).strip()
    return t

def _is_noise(token: str, min_len: int = 3) -> bool:
    """Buang token terlalu pendek, tanpa vokal, angka saja, atau huruf berulang."""
    return (
        len(token) < min_len
        or not _VOWEL.search(token)
        or bool(_REP3.search(token))
        or _DIGIT_ONLY.match(token)
    )

@lru_cache(maxsize=200_000)
def _stem_hybrid(tok: str) -> str:
    """Stemming Bahasa Indonesia + fallback Porter (Inggris)."""
    s_id = _id_stemmer.stem(tok)
    if s_id != tok:
        return s_id
    if _ENWORD.match(tok) and tok.endswith(("ing", "ed", "tion", "s", "es", "ers", "ies")):
        return _en_stemmer.stem(tok)
    return tok

def tokenize(text: str) -> list[str]:
    return _normalize(text).split()

def preprocess_text(text: str) -> list[str]:
    toks = tokenize(text)
    toks = [t for t in toks if t not in STOPWORDS]
    toks = [_stem_hybrid(t) for t in toks]
    toks = [t for t in toks if t not in STOPWORDS and not _is_noise(t)]
    return toks

def pick_text_column(df: pd.DataFrame, forced: str | None):
    if forced and forced in df.columns:
        return forced
    lower_map = {c.lower(): c for c in df.columns}
    for k in COMMON_TEXT_COLS:
        if k in lower_map:
            return lower_map[k]
    return df.select_dtypes(include='object').columns[0]

# ==== Pipeline utama ====
def run(text_col: str | None = None):
    print(f"[i] Membaca dataset dari: {INPUT_PATH}")
    df = pd.read_csv(INPUT_PATH)
    col = pick_text_column(df, text_col)
    print(f"[i] Kolom teks terdeteksi: '{col}'")

    clean_tokens = [preprocess_text(x) for x in df[col].tolist()]
    out = pd.DataFrame({"clean_tokens": clean_tokens})
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUTPUT_PATH, index=False)
    print(f"[✓] Mojok selesai: {len(out)} baris disimpan ke {OUTPUT_PATH}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--text-col", type=str, default=None)
    args = ap.parse_args()
    run(args.text_col)