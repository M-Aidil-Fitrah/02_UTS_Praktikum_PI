import pandas as pd
import re
import os
import csv
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS


# ======== 0. Fungsi Perbaikan CSV Tidak Rapi ========

def perbaiki_csv_tidak_rapi(input_path):
    """
    Gabungkan baris CSV yang kontennya terpotong oleh newline di dalam tanda kutip.
    Hasil disimpan sementara di memori (tidak buat file _fixed lagi).
    """
    with open(input_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    hasil = []
    buffer = ''
    judul = ''

    for line in lines:
        line = line.strip('\n')
        if not line:
            continue

        # Jika line mengandung kutip pembuka tapi belum kutip penutup
        if line.count('"') == 1 and buffer == '':
            if ',' in line:
                judul, buffer = line.split(',', 1)
            else:
                continue
            continue

        # Jika sedang menggabung isi konten
        if buffer:
            buffer += ' ' + line
            if buffer.count('"') % 2 == 0:
                hasil.append([judul.strip(), buffer.strip('"').strip()])
                buffer = ''
            continue

        # Baris normal (judul + konten sudah lengkap)
        if line.count('"') >= 2 and ',' in line:
            judul, konten = line.split(',', 1)
            hasil.append([judul.strip(), konten.strip('"').strip()])

    df = pd.DataFrame(hasil, columns=['judul', 'konten'])
    return df


# ======== 1. Fungsi Preprocessing ========

def casefolding(text):
    return text.lower()

def normalisasi(text):
    text = re.sub(r'[^a-z0-9\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def tokenisasi(text):
    return text.split()

def filter_token(tokens):
    return [t for t in tokens if t.isalpha() and len(t) > 2]


# ======== 2. Stopword Removal ========

def load_stopwords(file_path="stopwords_indo.txt"):
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            custom_stopwords = set(line.strip() for line in f if line.strip())
            return set(ENGLISH_STOP_WORDS).union(custom_stopwords)
    except FileNotFoundError:
        print(f"⚠️ File stopwords tidak ditemukan: {file_path}. Menggunakan default ENGLISH_STOP_WORDS saja.")
        return set(ENGLISH_STOP_WORDS)

STOPWORDS = load_stopwords()

def hapus_stopword(tokens):
    return [t for t in tokens if t not in STOPWORDS]


# ======== 3. Pipeline Preprocessing ========

def preprocess_text(text):
    text = casefolding(text)
    text = normalisasi(text)
    tokens = tokenisasi(text)
    tokens = filter_token(tokens)
    tokens = hapus_stopword(tokens)
    return tokens


# ======== 4. Proses Dataset ========

def preprocess_csv(file_path, output_path):
    print(f"\n🚀 Memproses dataset: {file_path}")
    df = perbaiki_csv_tidak_rapi(file_path)

    text_col = None
    for col in df.columns:
        if df[col].dtype == 'object':
            text_col = col
            break

    if text_col is None:
        raise ValueError(f"Tidak ada kolom teks dalam {file_path}")

    total = len(df)
    tokens_list = []
    for i, text in enumerate(df[text_col].fillna("")):
        tokens_list.append(preprocess_text(text))
        if (i + 1) % max(1, total // 10) == 0 or i == total - 1:
            progress = int((i + 1) / total * 100)
            print(f"   ➤ Progress: {progress}% ({i+1}/{total})")

    df['tokens'] = tokens_list
    df[['tokens']].to_csv(output_path, index=False)
    print(f"✅ Selesai → {output_path} ({len(df)} data)")


# ======== 5. Jalankan untuk semua dataset ========

os.makedirs("dataset_clean", exist_ok=True)

datasets = [
    'dataset/etd_ugm.csv',
    'dataset/etd_usk.csv',
    'dataset/kompas.csv',
    'dataset/mojok.csv',
    'dataset/tempo.csv'
]

for ds in datasets:
    nama_file = os.path.basename(ds)
    output_path = os.path.join("dataset_clean", nama_file)
    try:
        preprocess_csv(ds, output_path)
    except Exception as e:
        print(f"⚠️ Gagal memproses {ds}: {e}")
