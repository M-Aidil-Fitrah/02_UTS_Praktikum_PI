"""
Preprocessing Tempo 30
Fokus: case folding, tokenization, stopword removal + noise filter
Output: hanya 'clean_tokens'
Catatan: angka DIPERTAHANKAN (tahun/tanggal berguna).
"""
from __future__ import annotations
import argparse
import re
from pathlib import Path
import pandas as pd

# ==== Konfigurasi ====
INPUT_PATH  = Path("dataset/tempo.csv")
OUTPUT_PATH = Path("dataset_clean/tempo_clean.csv")
STOPWORDS   = Path("stopwords_indo.txt")

# ==== Regex & helper ====
_URL_RE   = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)
_MENTION  = re.compile(r"@[\w_]+", re.UNICODE)
_HASHTAG  = re.compile(r"#")
_NONALPHA = re.compile(r"[^a-z0-9\s]", re.UNICODE)          # TEMPO: izinkan angka
_MULTISP  = re.compile(r"\s+")
_REP3     = re.compile(r"(.)\1{2,}")
_VOWEL    = re.compile(r"[aeiou]")

COMMON_TEXT_COLS = ['content','text','isi','artikel','judul','title','body','description']

def load_stopwords(path: Path) -> set[str]:
    stops = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        w = line.strip().lower()
        if w and not w.startswith("#"):
            stops.add(w)
    return stops

def _normalize(text: str) -> str:
    if not isinstance(text, str):
        text = "" if text is None else str(text)
    t = text.lower()
    t = _URL_RE.sub(" ", t)
    t = _MENTION.sub(" ", t)
    t = _HASHTAG.sub(" ", t)
    t = _NONALPHA.sub(" ", t)    # buang non alfanumerik
    t = _MULTISP.sub(" ", t).strip()
    return t

def _is_noise(token: str, min_len: int = 3) -> bool:
    # terlalu pendek, tak ada vokal (untuk token alfabetik), atau huruf sama berulang ≥3
    if len(token) < min_len:
        return True
    if token.isalpha() and not _VOWEL.search(token):
        return True
    if _REP3.search(token):
        return True
    return False

def tokenize(text: str) -> list[str]:
    return _normalize(text).split()

def preprocess_text(text: str, stopwords: set[str]) -> list[str]:
    toks = tokenize(text)
    toks = [t for t in toks if t not in stopwords]
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
    stops = load_stopwords(STOPWORDS)
    col = pick_text_column(df, text_col)
    clean_tokens = [preprocess_text(x, stops) for x in df[col].tolist()]
    out = pd.DataFrame({"clean_tokens": clean_tokens})
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUTPUT_PATH, index=False)
    print(f"[✓] Tempo selesai: {len(out)} baris (kolom teks='{col}').")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--text-col", type=str, default=None)
    args = ap.parse_args()
    run(args.text_col)
