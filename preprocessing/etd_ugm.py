"""
Preprocessing ETD UGM (2 tahap):
1. Fix multiline CSV → etd_ugm_linefix.csv
2. Preprocessing teks (lowercase, stopword, tokenisasi, dsb) → etd_ugm_clean.csv
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
INPUT_PATH   = Path("../dataset/etd_ugm.csv")
LINEFIX_PATH = Path("../dataset_clean/etd_ugm_linefix.csv")
OUTPUT_PATH  = Path("../dataset_clean/etd_ugm_clean.csv")

# ==== Regex & helper ====
_URL_RE     = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)
_MENTION    = re.compile(r"@[\w_]+", re.UNICODE)
_HASHTAG    = re.compile(r"#")
_NONALPHA   = re.compile(r"[^a-z\s]", re.UNICODE)   # hanya huruf a-z
_MULTISP    = re.compile(r"\s+")
_REP3       = re.compile(r"(.)\1{2,}")
_VOWEL      = re.compile(r"[aeiou]")
_DIGIT_ONLY = re.compile(r"^\d+$")

COMMON_TEXT_COLS = ['content','text','isi','artikel','judul','title','body','description']


# ==== 1️⃣ Perbaikan baris multiline ====
def fix_multiline_csv(in_path: Path, out_path: Path):
    lines = in_path.read_text(encoding="utf-8").splitlines()
    fixed_lines = []
    buffer = ""

    for line in lines:
        if not buffer:
            buffer = line
        else:
            buffer += " " + line

        # Jika jumlah tanda kutip genap → baris sudah lengkap
        if buffer.count('"') % 2 == 0:
            fixed_lines.append(buffer)
            buffer = ""

    # Jika masih ada buffer tersisa
    if buffer:
        fixed_lines.append(buffer)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(fixed_lines), encoding="utf-8")

    print(f"[✓] Multiline CSV diperbaiki → {out_path.name}")
    print(f"Jumlah baris hasil: {len(fixed_lines)}")


# ==== 2️⃣ Preprocessing ====
def load_stopwords(extra_stopwords_file: str | None = None) -> set:
    factory = StopWordRemoverFactory()
    stopwords = set(factory.get_stop_words())

    # Jika kamu punya file tambahan, tambahkan isinya
    if extra_stopwords_file and Path(extra_stopwords_file).exists():
        for line in Path(extra_stopwords_file).read_text(encoding="utf-8").splitlines():
            word = line.strip().lower()
            if word:
                stopwords.add(word)

    return stopwords

def _normalize(text: str) -> str:
    if not isinstance(text, str):
        text = "" if text is None else str(text)
    t = text.lower()
    t = t.replace('"', ' ')        # hapus tanda kutip
    t = t.replace('“', ' ').replace('”', ' ')
    t = t.replace('\xa0', ' ')     # hapus non-breaking space
    t = t.replace('\n', ' ')       # hapus newline sisa
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
    toks = [t for t in toks if t not in stopwords]
    toks = [t for t in toks if not _is_noise(t)]
    # Stemming per token agar akurat
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
    df = pd.read_csv(LINEFIX_PATH)
    stops = load_stopwords()
    col = pick_text_column(df, text_col)

    # Buat stemmer dari Sastrawi
    factory = StemmerFactory()
    stemmer = factory.create_stemmer()

    print(f"[i] Kolom teks yang digunakan: '{col}'")
    print(f"[i] Jumlah baris: {len(df)}")
    print("[…] Mulai preprocessing (tokenisasi, stopword removal, stemming)")

    # Tambahkan tqdm untuk progress bar
    clean_tokens = []
    for text in tqdm(df[col].tolist(), desc="Preprocessing", total=len(df)):
        clean_tokens.append(preprocess_text(text, stops, stemmer))

    out = pd.DataFrame({"clean_tokens": clean_tokens})
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUTPUT_PATH, index=False)

    print(f"[✓] Preprocessing selesai → {OUTPUT_PATH.name}")
    print(f"Jumlah baris: {len(out)}")


if __name__ == "__main__":
    # 1️⃣ Perbaiki file multiline jadi satu baris
    fix_multiline_csv(INPUT_PATH, Path("../dataset_clean/etd_ugm_linefix.csv"))

    # 2️⃣ Jalankan preprocessing
    INPUT_PATH = Path("../dataset_clean/etd_ugm_linefix.csv")
    OUTPUT_PATH = Path("../dataset_clean/etd_ugm_clean.csv")

    ap = argparse.ArgumentParser()
    ap.add_argument("--text-col", type=str, default=None)
    args = ap.parse_args()
    run(args.text_col)
