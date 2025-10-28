"""
Preprocessing ETD USK (1 tahap):
Preprocessing teks (lowercase, pemisahan ABSTRAK*, stopword, tokenisasi, stemming, dsb)
→ etd_usk_clean.csv
"""
from __future__ import annotations
import argparse
import re
from pathlib import Path
import pandas as pd
from tqdm import tqdm
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory

# ==== Konfigurasi ====
INPUT_PATH   = Path("../dataset/etd_usk.csv")
OUTPUT_PATH  = Path("../dataset_clean/etd_usk_clean.csv")

# ==== Regex & helper ====
_URL_RE     = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)
_MENTION    = re.compile(r"@[\w_]+", re.UNICODE)
_HASHTAG    = re.compile(r"#")
_NONALPHA   = re.compile(r"[^a-z\s]", re.UNICODE)
_MULTISP    = re.compile(r"\s+")
_REP3       = re.compile(r"(.)\1{2,}")
_VOWEL      = re.compile(r"[aeiou]")
_DIGIT_ONLY = re.compile(r"^\d+$")

# Tambahan: mendeteksi ABSTRAK*, Judul*, LatarBelakang yang menempel
_ABSTRACT_JOINED = re.compile(r"(ABSTRAK|Judul|Latar\s*Belakang)(?=[A-Z])")

# Kata yang ingin dihapus total
_REMOVE_KEYWORDS = {"abstrak", "judul", "latar", "belakang"}

COMMON_TEXT_COLS = ['content','text','isi','artikel','judul','title','body','description']


# ==== Fungsi dasar ====
def load_stopwords(extra_stopwords_file: str | None = None) -> set:
    factory = StopWordRemoverFactory()
    stopwords = set(factory.get_stop_words())
    if extra_stopwords_file and Path(extra_stopwords_file).exists():
        for line in Path(extra_stopwords_file).read_text(encoding="utf-8").splitlines():
            word = line.strip().lower()
            if word:
                stopwords.add(word)
    return stopwords


def _normalize(text: str) -> str:
    if not isinstance(text, str):
        text = "" if text is None else str(text)

    # --- Pisahkan kata ABSTRAK*, Judul*, LatarBelakang* yang menempel ---
    text = _ABSTRACT_JOINED.sub(r"\1 ", text)

    t = text.lower()
    t = t.replace('"', ' ').replace('“', ' ').replace('”', ' ')
    t = t.replace('\xa0', ' ').replace('\n', ' ')
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


def preprocess_text(text: str, stopwords: set[str], stemmer) -> list[str]:
    toks = tokenize(text)
    toks = [t for t in toks if t not in stopwords and t not in _REMOVE_KEYWORDS]
    toks = [t for t in toks if not _is_noise(t)]
    stemmed = [stemmer.stem(t) for t in toks]
    return stemmed


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
    stops = load_stopwords()
    col = pick_text_column(df, text_col)

    factory = StemmerFactory()
    stemmer = factory.create_stemmer()

    print(f"[i] Kolom teks yang digunakan: '{col}'")
    print(f"[i] Jumlah baris: {len(df)}")
    print("[…] Mulai preprocessing (normalize + tokenisasi + stopword + stemming)")

    clean_tokens = []
    for text in tqdm(df[col].tolist(), desc="Preprocessing", total=len(df)):
        clean_tokens.append(preprocess_text(text, stops, stemmer))

    out = pd.DataFrame({"clean_tokens": clean_tokens})
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUTPUT_PATH, index=False)

    print(f"[✓] Preprocessing selesai → {OUTPUT_PATH.name}")
    print(f"Jumlah baris: {len(out)}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--text-col", type=str, default=None)
    args = ap.parse_args()
    run(args.text_col)
