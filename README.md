# Information Retrieval System - UTS Praktikum PI

**UTS Praktikum Penelusuran Informasi**  
Departemen Informatika FMIPA, Universitas Syiah Kuala

## 📋 Deskripsi Proyek

Sistem Information Retrieval (IR) berbasis Command-Line Interface (CLI) yang mampu melakukan pencarian dan ranking dokumen dari berbagai sumber teks nyata menggunakan teknik **Vector Space Model**, **Bag of Words (BoW)**, **Whoosh Indexing**, dan **Cosine Similarity**.

## 🎯 Fitur Utama

✅ **Preprocessing dan Tokenisasi Teks**
- Case folding
- Tokenization
- Stopword removal (Bahasa Indonesia)
- Stemming menggunakan Sastrawi

✅ **Representasi Dokumen (Bag of Words)**
- Menggunakan CountVectorizer dari scikit-learn
- Representasi vektor untuk setiap dokumen

✅ **Pembentukan Index Dokumen (Whoosh)**
- Index terpisah untuk setiap dataset
- Fast full-text search capability

✅ **Pencarian Berbasis Query**
- Search di semua dataset
- Search di dataset spesifik
- Preprocessing otomatis untuk query

✅ **Ranking Hasil (Cosine Similarity)**
- Perhitungan kemiripan dokumen
- Top-N results berdasarkan skor similarity
- Tampilan hasil yang informatif

## 📚 Dataset

Sistem ini menggunakan 5 dataset utama:

1. **etd-usk** - Tesis/Disertasi Universitas Syiah Kuala
2. **etd-ugm** - Tesis/Disertasi Universitas Gadjah Mada
3. **kompas** - Berita Harian Nasional
4. **tempo** - Majalah Berita dan Opini
5. **mojok** - Artikel Populer dan Satir

## 🚀 Instalasi

### 1. Clone Repository

```bash
git clone https://github.com/M-Aidil-Fitrah/UTS-Prak-PI-A.git
cd UTS-Prak-PI-A
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Struktur Folder

Pastikan struktur folder sebagai berikut:

```
UTS-Prak-PI-A/
├── dataset/                    # Dataset asli
│   ├── etd_ugm.csv
│   ├── etd_usk.csv
│   ├── kompas.csv
│   ├── mojok.csv
│   └── tempo.csv
├── dataset_clean/              # Dataset yang sudah di-preprocessing
│   ├── etd_ugm_clean.csv
│   ├── etd_usk_clean.csv
│   ├── kompas_clean.csv
│   ├── mojok_clean_all.csv
│   └── tempo_clean.csv
├── index/                      # Whoosh index files
│   ├── index_etd_ugm_clean/
│   ├── index_etd_usk_clean/
│   ├── index_kompas_clean/
│   ├── index_mojok_clean_all/
│   └── index_tempo_clean/
├── preprocessing/              # Script preprocessing
│   ├── etd_ugm.py
│   ├── etd_usk.py
│   ├── kompas.py
│   ├── mojok.py
│   └── tempo.py
├── bow.ipynb                   # Notebook untuk BoW dan Indexing
├── ui.py                       # Main CLI application
├── requirements.txt            # Dependencies
└── README.md                   # Dokumentasi
```

## 💻 Cara Penggunaan

### Menjalankan Sistem

```bash
python ui.py
```

### Menu Utama

```
=================================================================
               INFORMATION RETRIEVAL SYSTEM
=================================================================
[1] Load & Index Dataset
[2] Search Query - All Datasets
[3] Search Query - Specific Dataset
[4] Show Dataset Statistics
[5] Exit
=================================================================
```

### Langkah-langkah Penggunaan

1. **Load Dataset** (Pilihan 1)
   - Memuat semua dataset dan index
   - Membuat vectorizer untuk setiap dataset
   - Wajib dilakukan sebelum melakukan pencarian

2. **Search All Datasets** (Pilihan 2)
   - Mencari query di semua dataset
   - Menampilkan top-N hasil dengan skor tertinggi
   - Hasil diurutkan berdasarkan cosine similarity

3. **Search Specific Dataset** (Pilihan 3)
   - Memilih dataset tertentu
   - Pencarian lebih fokus dan cepat
   - Cocok untuk domain-specific search

4. **Show Statistics** (Pilihan 4)
   - Menampilkan statistik dataset
   - Jumlah dokumen per dataset
   - Total dokumen keseluruhan

### Contoh Penggunaan

```
👉 Select option [1-5]: 2

🔍 Enter your search query: machine learning
📊 Number of results to display (default 5): 5

🔍 Searching for: 'machine learning'
📝 Preprocessed query: 'machine learn'

======================================================================
📊 Found 5 relevant documents:

======================================================================
Rank #1
Dataset: etd_ugm_clean
Document ID: 42
Similarity Score: 0.8523
Content Preview: ['machine', 'learn', 'algoritma', 'data', 'model']...

======================================================================
```

## 🛠️ Teknologi yang Digunakan

- **Python 3.x** - Programming language
- **Pandas** - Data manipulation
- **scikit-learn** - Machine learning (CountVectorizer, Cosine Similarity)
- **Whoosh** - Full-text indexing and searching
- **Sastrawi** - Indonesian stemming and stopword removal

## 📊 Pipeline Sistem

```
┌─────────────────────┐
│   Raw Documents     │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│   Preprocessing     │
│  - Case folding     │
│  - Tokenization     │
│  - Stopword removal │
│  - Stemming         │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│  BoW Representation │
│  (CountVectorizer)  │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│  Whoosh Indexing    │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│   User Query        │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│ Cosine Similarity   │
│   Calculation       │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│  Ranked Results     │
└─────────────────────┘
```

## 📖 Dokumentasi Kode

### Class: `InformationRetrievalSystem`

#### Methods:

- `__init__()` - Inisialisasi sistem
- `preprocess_query(query)` - Preprocessing query input
- `load_and_index_datasets()` - Load dataset dan index
- `calculate_cosine_similarity(query_text, theme, top_n)` - Hitung cosine similarity
- `search_all_datasets(query_text, top_n)` - Search di semua dataset
- `search_single_dataset(query_text, theme, top_n)` - Search di dataset tertentu
- `display_results(results)` - Tampilkan hasil pencarian
- `show_statistics()` - Tampilkan statistik dataset
- `run()` - Main program loop

## 🔍 Algoritma Cosine Similarity

Cosine similarity mengukur kesamaan antara dua vektor dengan menghitung cosinus dari sudut di antara keduanya:

$$\text{cosine similarity} = \frac{A \cdot B}{\|A\| \|B\|}$$

Dimana:
- $A$ = Query vector
- $B$ = Document vector
- $A \cdot B$ = Dot product
- $\|A\|, \|B\|$ = Euclidean norms

Range: [0, 1]
- 0 = Tidak ada kesamaan
- 1 = Identik sempurna

## 👥 Tim Pengembang

- **Anggota 1** - [Nama]
- **Anggota 2** - [Nama]
- **Anggota 3** - [Nama]

## 📄 Lisensi

Project ini dibuat untuk keperluan akademis UTS Praktikum Penelusuran Informasi.

## 🙏 Acknowledgments

- **Dosen Pengampu:**
  - Prof. Dr. Taufik Fuadi Abidin, S.Si., M.Tech
  - Fathia Sabrina, S.T., M.Inf.Tech
  - Fitria Nilamsari, S.Kom., M.Sc

- **Departemen Informatika FMIPA**
- **Universitas Syiah Kuala**

---

**© 2025 - UTS Praktikum Penelusuran Informasi**