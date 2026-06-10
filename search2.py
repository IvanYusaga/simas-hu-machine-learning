from sentence_transformers import SentenceTransformer, util
from rank_bm25 import BM25Okapi
import ahocorasick
import numpy as np
import re
import os
import pymysql
from dotenv import load_dotenv
from functools import lru_cache

from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory

# ==================================================
# NORMALIZATION MAP
# ==================================================

try:
    from normalization_map import NORMALIZATION_MAP
except Exception:
    NORMALIZATION_MAP = {}

# ==================================================
# LOAD ENVIRONMENT
# ==================================================
load_dotenv()

DB_HOST = os.getenv("DB_HOST", "127.0.0.1")
DB_PORT = int(os.getenv("DB_PORT", 3306))
DB_DATABASE = os.getenv("DB_DATABASE", "skripsi")
DB_USERNAME = os.getenv("DB_USERNAME", "root")
DB_PASSWORD = os.getenv("DB_PASSWORD", "")

# ==================================================
# NLP TOOLS (Load once)
# ==================================================
_stopword_factory = StopWordRemoverFactory()
STOPWORDS = set(_stopword_factory.get_stop_words())

_stemmer_factory = StemmerFactory()
stemmer = _stemmer_factory.create_stemmer()

# ==================================================
# PRECOMPILED NORMALIZATION
# ==================================================

NORMALIZATION_PATTERNS = [
    (
        re.compile(r"\b" + re.escape(k.lower()) + r"\b"),
        v.lower()
    )
    for k, v in NORMALIZATION_MAP.items()
]

# =========================================================
# STEM CACHE
# =========================================================

@lru_cache(maxsize=50000)
def cached_stem(word: str) -> str:
    """
    Melakukan stemming pada satu kata bahasa Indonesia dengan caching.

    Menggunakan LRU cache (kapasitas 50.000 kata) untuk menghindari
    pemanggilan stemmer berulang kali pada kata yang sama, sehingga
    meningkatkan performa secara signifikan saat memproses banyak dokumen.

    Args:
        word (str): Kata tunggal yang akan di-stem.

    Returns:
        str: Bentuk dasar (stem) dari kata tersebut.

    Contoh:
        >>> cached_stem("pembelajaran")
        'ajar'
    """
    return stemmer.stem(word)

# ==================================================
# SBERT MODEL (Load once)
# ==================================================
model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")

# ==================================================
# SYNONYM MAP — Common Indonesian academic synonyms
# Helps with query expansion so users don't need
# to guess the exact wording used in titles.
# ==================================================
SYNONYM_MAP = {
    "ai": ["artificial intelligence", "kecerdasan buatan"],
    "kecerdasan buatan": ["artificial intelligence", "ai"],
    "artificial intelligence": ["kecerdasan buatan", "ai"],
    "ml": ["machine learning", "pembelajaran mesin"],
    "machine learning": ["pembelajaran mesin", "ml"],
    "pembelajaran mesin": ["machine learning", "ml"],
    "dl": ["deep learning", "pembelajaran mendalam"],
    "deep learning": ["pembelajaran mendalam", "dl"],
    "iot": ["internet of things"],
    "internet of things": ["iot"],
    "web": ["website", "aplikasi web"],
    "website": ["web", "situs"],
    "android": ["mobile", "aplikasi mobile"],
    "mobile": ["android", "aplikasi mobile"],
    "analisis": ["analisa"],
    "analisa": ["analisis"],
    "implementasi": ["penerapan"],
    "penerapan": ["implementasi"],
    "perancangan": ["desain", "rancangan"],
    "desain": ["perancangan", "design"],
    "design": ["desain", "perancangan"],
    "sistem": ["system"],
    "system": ["sistem"],
    "klasifikasi": ["classification"],
    "classification": ["klasifikasi"],
    "prediksi": ["prediction", "peramalan"],
    "prediction": ["prediksi"],
    "peramalan": ["prediksi", "forecasting"],
    "forecasting": ["peramalan", "prediksi"],
    "deteksi": ["detection", "pendeteksian"],
    "detection": ["deteksi"],
    "citra": ["image", "gambar"],
    "image": ["citra", "gambar"],
    "gambar": ["citra", "image"],
    "pengolahan": ["processing"],
    "processing": ["pengolahan"],
    "jaringan": ["network"],
    "network": ["jaringan"],
    "data": ["dataset"],
    "basis data": ["database"],
    "database": ["basis data"],
    "keamanan": ["security"],
    "security": ["keamanan"],
    "informasi": ["information"],
    "information": ["informasi"],
    "pengaruh": ["dampak", "efek"],
    "dampak": ["pengaruh", "efek"],
    "efek": ["pengaruh", "dampak"],
    "optimasi": ["optimization", "optimalisasi"],
    "optimization": ["optimasi"],
    "optimalisasi": ["optimasi"],
    "sentimen": ["sentiment"],
    "sentiment": ["sentimen"],
    "pkl": ["magang", "kerja praktek", "praktik kerja lapangan"],
    "magang": ["pkl", "kerja praktek", "praktik kerja lapangan"],
    "skripsi": ["tugas akhir"],
    "tugas akhir": ["skripsi"],
}

# ==================================================
# GLOBAL DATA STORAGE
# ==================================================
documents_magang = []
documents_skripsi = []
bm25_magang = None
bm25_skripsi = None
doc_embeddings_magang = None
doc_embeddings_skripsi = None
# Pre-tokenized docs for quick index-lookup
_tokenized_magang = []
_tokenized_skripsi = []

# ==================================================
# DATABASE CONNECTION
# ==================================================
def get_db_connection():
    """
    Membuat dan mengembalikan koneksi baru ke database MySQL.

    Koneksi menggunakan konfigurasi dari environment variables
    (DB_HOST, DB_PORT, DB_DATABASE, DB_USERNAME, DB_PASSWORD)
    yang dimuat melalui dotenv.

    Returns:
        pymysql.connections.Connection: Objek koneksi MySQL dengan
            charset utf8mb4 dan cursor bertipe DictCursor
            (hasil query berupa dictionary).

    Raises:
        pymysql.err.OperationalError: Jika koneksi ke database gagal
            (misal: server tidak aktif, kredensial salah).
    """
    return pymysql.connect(
        host=DB_HOST,
        port=DB_PORT,
        user=DB_USERNAME,
        password=DB_PASSWORD,
        database=DB_DATABASE,
        charset='utf8mb4',
        cursorclass=pymysql.cursors.DictCursor
    )

# ==================================================
# LOAD DATA FROM DATABASE
# ==================================================
def load_documents_from_db():
    """
    Memuat data dokumen laporan dari database MySQL.

    Fungsi ini mengambil data laporan magang/PKL dan laporan skripsi
    dari database, lalu menyimpannya ke variabel global
    `documents_magang` dan `documents_skripsi`.

    Kriteria filter dokumen:
        1. Hanya laporan yang sudah pernah diajukan (memiliki minimal
           satu record di tabel verifikasi terkait).
        2. Hanya laporan leaf node (tidak memiliki child/revisi turunan),
           sehingga yang diambil adalah versi terbaru.

    Returns:
        tuple: (documents_magang, documents_skripsi) — masing-masing
            berupa list of dict dengan key:
            - 'id'    : ID laporan
            - 'judul' : Judul laporan
            - 'tipe'  : 'magang' atau 'skripsi'

    Side Effects:
        Mengubah variabel global `documents_magang` dan `documents_skripsi`.
    """
    global documents_magang, documents_skripsi

    conn = get_db_connection()
    try:
        with conn.cursor() as cursor:
            # Query untuk Laporan Magang/PKL
            query_magang = """
                SELECT 
                    lm.id_laporan_magang_pkl as id,
                    lm.judul_laporan as judul,
                    'magang' as tipe
                FROM laporan_magang_pkl lm
                WHERE EXISTS (
                        SELECT 1 FROM verifikasi_laporan_magang_pkl v
                        WHERE v.id_laporan_magang_pkl = lm.id_laporan_magang_pkl
                    )
                AND NOT EXISTS (
                    SELECT 1 FROM laporan_magang_pkl child
                    WHERE child.id_parent = lm.id_laporan_magang_pkl
                )
            """
            cursor.execute(query_magang)
            documents_magang = cursor.fetchall()

            # Query untuk Laporan Skripsi
            query_skripsi = """
                SELECT 
                    ls.id_laporan_skripsi as id,
                    ls.judul_skripsi as judul,
                    'skripsi' as tipe
                FROM laporan_skripsi ls
                WHERE EXISTS (
                        SELECT 1 FROM verifikasi_laporan_skripsi v
                        WHERE v.id_laporan_skripsi = ls.id_laporan_skripsi
                    )
                AND NOT EXISTS (
                    SELECT 1 FROM laporan_skripsi child
                    WHERE child.id_parent = ls.id_laporan_skripsi
                )
            """
            cursor.execute(query_skripsi)
            documents_skripsi = cursor.fetchall()

    finally:
        conn.close()

    return documents_magang, documents_skripsi

# ==================================================
# TEXT NORMALIZATION
# ==================================================

def normalize_text(text: str):
    """
    Menormalisasi teks dengan mengubah ke huruf kecil dan menerapkan
    peta normalisasi kata tidak baku.

    Proses normalisasi:
        1. Konversi seluruh teks ke lowercase.
        2. Mengganti kata-kata tidak baku/singkatan dengan bentuk bakunya
           menggunakan NORMALIZATION_PATTERNS (regex yang sudah di-compile).

    Args:
        text (str): Teks mentah yang akan dinormalisasi.

    Returns:
        str: Teks yang sudah dinormalisasi (lowercase + kata baku).

    Contoh:
        >>> normalize_text("Analsis Sistem Infrmasi")
        'analisis sistem informasi'
    """
    text = str(text).lower()

    for pattern, replacement in NORMALIZATION_PATTERNS:
        text = pattern.sub(replacement, text)

    return text

# ==================================================
# PREPROCESSING
# ==================================================
_RE_NON_ALPHANUM = re.compile(r'[^a-z0-9\s]')

def tokenize(text):
    """
    Memecah teks menjadi daftar token (kata) setelah normalisasi.

    Proses:
        1. Normalisasi teks (lowercase + peta normalisasi).
        2. Menghapus semua karakter non-alfanumerik (tanda baca, simbol).
        3. Memecah teks berdasarkan spasi.

    Args:
        text (str): Teks yang akan di-tokenisasi.

    Returns:
        list[str]: Daftar token berupa kata-kata dalam huruf kecil.

    Contoh:
        >>> tokenize("Analisis Sistem Informasi (Studi Kasus)")
        ['analisis', 'sistem', 'informasi', 'studi', 'kasus']
    """
    text = normalize_text(text)

    text = _RE_NON_ALPHANUM.sub(' ', text)

    return text.split()

def tokenize_and_stem(text):
    """
    Tokenisasi teks dengan penghapusan stopword dan stemming bahasa Indonesia.

    Proses:
        1. Tokenisasi teks menggunakan fungsi `tokenize()`.
        2. Menghapus stopword bahasa Indonesia dan token berukuran ≤ 1 karakter.
        3. Melakukan stemming pada setiap token yang tersisa menggunakan
           Sastrawi stemmer (dengan caching).

    Args:
        text (str): Teks yang akan diproses.

    Returns:
        list[str]: Daftar token yang sudah di-stem, tanpa stopword.

    Contoh:
        >>> tokenize_and_stem("Implementasi Pembelajaran Mesin untuk Klasifikasi")
        ['implementasi', 'ajar', 'mesin', 'klasifikasi']
    """
    tokens = tokenize(text)
    result = []
    for t in tokens:
        if t in STOPWORDS or len(t) <= 1:
            continue
        stemmed = cached_stem(t)
        if stemmed:
            result.append(stemmed)
    return result

def tokenize_remove_stopwords(text):
    """
    Tokenisasi teks dengan penghapusan stopword, tanpa stemming.

    Berbeda dengan `tokenize_and_stem()`, fungsi ini mempertahankan
    bentuk asli kata (tidak di-stem). Berguna ketika perlu pencocokan
    kata secara eksak (exact match) tanpa mengubah bentuk kata.

    Args:
        text (str): Teks yang akan diproses.

    Returns:
        list[str]: Daftar token tanpa stopword dan tanpa token pendek (≤ 1 karakter),
            dalam bentuk kata asli (tidak di-stem).

    Contoh:
        >>> tokenize_remove_stopwords("analisis dari sebuah sistem informasi")
        ['analisis', 'sistem', 'informasi']
    """
    tokens = tokenize(text)
    return [t for t in tokens if t not in STOPWORDS and len(t) > 1]

# ==================================================
# QUERY EXPANSION
# ==================================================
def expand_query(query):
    """
    Memperluas query pencarian dengan sinonim untuk meningkatkan recall.

    Fungsi ini mencocokkan kata/frasa pada query dengan SYNONYM_MAP,
    lalu menambahkan semua sinonim yang ditemukan ke akhir query.
    Pencocokan dilakukan dalam dua tahap:
        1. Frasa multi-kata (dicocokkan terlebih dahulu, diurutkan
           dari yang terpanjang agar frasa lebih spesifik diprioritaskan).
        2. Kata tunggal dari hasil tokenisasi query.

    Args:
        query (str): Query pencarian asli dari pengguna.

    Returns:
        str: Query yang sudah diperluas. Format: "<query asli> <sinonim1> <sinonim2> ...".
            Jika tidak ada sinonim ditemukan, mengembalikan query asli tanpa perubahan.

    Contoh:
        >>> expand_query("machine learning")
        'machine learning pembelajaran mesin ml'
    """
    query_lower = normalize_text(query)
    expansions = set()

    # Check multi-word synonyms first (longer phrases take priority)
    sorted_keys = sorted(SYNONYM_MAP.keys(), key=len, reverse=True)
    for key in sorted_keys:
        if key in query_lower:
            for syn in SYNONYM_MAP[key]:
                expansions.add(syn)

    # Check single-word synonyms
    for word in tokenize(query):
        if word in SYNONYM_MAP:
            for syn in SYNONYM_MAP[word]:
                expansions.add(syn)

    # Build expanded query — original query + expansions
    if expansions:
        return query + " " + " ".join(expansions)
    return query

# ==================================================
# BUILD SEARCH INDEX
# ==================================================
def build_search_index(documents):
    """
    Membangun indeks pencarian BM25 dan embedding SBERT untuk kumpulan dokumen.

    Proses:
        1. Untuk setiap dokumen, melakukan tokenisasi + stemming pada judul.
        2. Menyimpan versi bersih (judul_clean) dan versi stem (judul_stemmed)
           ke dalam setiap dict dokumen sebagai field tambahan.
        3. Membangun indeks BM25 dari token yang sudah di-stem.
        4. Menghasilkan embedding SBERT dari judul asli (model menangani
           semantik secara internal).

    Args:
        documents (list[dict]): Daftar dokumen, setiap dokumen berupa dict
            dengan minimal key 'judul'.

    Returns:
        tuple: (bm25, doc_embeddings, tokenized_docs)
            - bm25 (BM25Okapi | None): Objek indeks BM25, atau None jika
              dokumen kosong.
            - doc_embeddings (Tensor | None): Tensor embedding SBERT untuk
              seluruh dokumen, atau None jika dokumen kosong.
            - tokenized_docs (list[list[str]]): Daftar token per dokumen
              (sudah di-stem), digunakan untuk lookup cepat.

    Side Effects:
        Menambahkan key 'judul_clean' dan 'judul_stemmed' pada setiap
        dict dokumen di parameter `documents`.
    """
    if not documents:
        return None, None, []

    # BM25 — use stemmed tokens for better matching
    tokenized_docs = []

    for d in documents:

        stemmed_tokens = tokenize_and_stem(d["judul"])

        d["judul_clean"] = " ".join(
            tokenize(d["judul"])
        )

        d["judul_stemmed"] = " ".join(
            stemmed_tokens
        )

        tokenized_docs.append(
            stemmed_tokens
        )

    bm25 = BM25Okapi(tokenized_docs)

    # SBERT — encode raw titles (SBERT handles semantics itself)
    doc_embeddings = model.encode(
        [d["judul"] for d in documents],
        convert_to_tensor=True,
        show_progress_bar=False
    )

    return bm25, doc_embeddings, tokenized_docs

# ==================================================
# AHO-CORASICK (Word-boundary aware)
# ==================================================
def build_aho_automaton(keywords):
    """
    Membangun automaton Aho-Corasick dari daftar keyword.

    Aho-Corasick adalah algoritma pencocokan multi-pola yang efisien.
    Automaton yang dihasilkan dapat mencari semua keyword secara simultan
    dalam satu kali traversal teks (kompleksitas O(n + m + z), di mana
    n = panjang teks, m = total panjang keyword, z = jumlah match).

    Args:
        keywords (list[str]): Daftar kata kunci yang akan dicari.

    Returns:
        ahocorasick.Automaton: Automaton yang siap digunakan untuk
            pencocokan multi-pola.
    """
    A = ahocorasick.Automaton()
    for i, w in enumerate(keywords):
        A.add_word(w, (i, w))
    A.make_automaton()
    return A

def aho_score_documents(query, documents):
    """
    Menghitung skor relevansi dokumen menggunakan pencocokan multi-pola Aho-Corasick.

    Fungsi ini mencari kemunculan keyword query (baik bentuk asli maupun
    bentuk stem) di dalam judul dokumen, dengan pengecekan batas kata
    (word boundary) untuk menghindari partial match.

    Perbaikan dari versi awal:
        - Menggunakan keyword yang sudah di-stem untuk pencocokan kata dasar.
        - Pengecekan word boundary agar tidak mencocokkan sebagian kata.
        - Skor menggabungkan rasio cakupan keyword unik (coverage) dan
          kepadatan kemunculan (density).

    Rumus skor per dokumen:
        score = 0.7 × coverage + 0.3 × density
        - coverage: fraksi keyword unik yang ditemukan / total keyword
        - density: total hit / total keyword, di-cap pada 1.0

    Args:
        query (str): Query pencarian dari pengguna.
        documents (list[dict]): Daftar dokumen yang memiliki key
            'judul_clean' dan 'judul_stemmed'.

    Returns:
        np.ndarray: Array skor Aho-Corasick untuk setiap dokumen
            (nilai antara 0.0 – 1.0).
    """
    if not documents:
        return np.array([])

    # Get stemmed keywords from query (no stopwords)
    keywords = tokenize_remove_stopwords(query)
    stemmed_keywords = list(set(tokenize_and_stem(query)))

    # Combine original keywords + stemmed for broader matching
    all_keywords = list(set(keywords + stemmed_keywords))

    if not all_keywords:
        return np.zeros(len(documents))

    automaton = build_aho_automaton(all_keywords)
    total_keywords = len(all_keywords)
    scores = []

    for doc in documents:
        combined_text = (
            doc.get("judul_clean", "")
            + " "
            + doc.get("judul_stemmed", "")
        )

        matched_keywords = set()
        total_matches = 0

        for end_idx, (_, matched_word) in automaton.iter(combined_text):
            # Word-boundary check: ensure we match whole words
            start_idx = end_idx - len(matched_word) + 1
            before_ok = (start_idx == 0 or not combined_text[start_idx - 1].isalnum())
            after_ok = (end_idx + 1 >= len(combined_text) or not combined_text[end_idx + 1].isalnum())

            if before_ok and after_ok:
                matched_keywords.add(matched_word)
                total_matches += 1

        # Score = weighted combination of:
        # - coverage: fraction of query keywords found (rewards matching more keywords)
        # - density: total hit count (rewards multiple occurrences, capped)
        coverage = len(matched_keywords) / total_keywords
        density = min(total_matches / total_keywords, 2.0) / 2.0  # cap at 1.0

        score = 0.7 * coverage + 0.3 * density
        scores.append(score)

    return np.array(scores, dtype=float)

# ==================================================
# NORMALIZATION
# ==================================================
def normalize_scores(scores):
    """
    Melakukan normalisasi min-max yang robust pada array skor.

    Menangani edge case:
        - Array kosong: dikembalikan apa adanya.
        - Semua nilai nol: dikembalikan array nol.
        - Semua nilai sama (non-nol): dikembalikan array berisi 0.5.
        - Normal: skor dinormalisasi ke rentang [0.0, 1.0].

    Args:
        scores (np.ndarray): Array skor mentah yang akan dinormalisasi.

    Returns:
        np.ndarray: Array skor yang sudah dinormalisasi ke rentang [0, 1].
    """
    if len(scores) == 0:
        return scores

    min_val = scores.min()
    max_val = scores.max()
    score_range = max_val - min_val

    if score_range < 1e-9:
        # All scores are the same — if they're all zero, keep zero;
        # otherwise set all to 0.5 (neutral)
        if max_val < 1e-9:
            return np.zeros_like(scores)
        return np.full_like(scores, 0.5)

    return (scores - min_val) / score_range

# ==================================================
# RELOAD DOCUMENTS (Called on startup and when needed)
# ==================================================
def reload_documents():
    """
    Memuat ulang seluruh dokumen dari database dan membangun ulang
    semua indeks pencarian.

    Fungsi ini dipanggil saat:
        - Inisialisasi modul (module load).
        - Endpoint /reload pada API dipanggil (misal setelah data berubah).

    Proses:
        1. Mengambil data dokumen magang dan skripsi dari database.
        2. Membangun indeks BM25 + embedding SBERT untuk masing-masing kategori.
        3. Menyimpan hasil ke variabel global yang digunakan oleh fungsi pencarian.

    Side Effects:
        Mengubah variabel global: documents_magang, documents_skripsi,
        bm25_magang, bm25_skripsi, doc_embeddings_magang,
        doc_embeddings_skripsi, _tokenized_magang, _tokenized_skripsi.
    """
    global documents_magang, documents_skripsi
    global bm25_magang, bm25_skripsi
    global doc_embeddings_magang, doc_embeddings_skripsi
    global _tokenized_magang, _tokenized_skripsi

    print("Loading documents from database...")
    load_documents_from_db()

    print(f"Building search index for {len(documents_magang)} magang documents...")
    bm25_magang, doc_embeddings_magang, _tokenized_magang = build_search_index(documents_magang)

    print(f"Building search index for {len(documents_skripsi)} skripsi documents...")
    bm25_skripsi, doc_embeddings_skripsi, _tokenized_skripsi = build_search_index(documents_skripsi)

    print("Search index ready!")

# ==================================================
# RANKING
# ==================================================
def rank_documents(query, category="magang", mode="hybrid"):
    """
    Meranking dokumen berdasarkan relevansi terhadap query pencarian.

    Menggabungkan tiga metode scoring secara hybrid:
        - SBERT  (bobot 0.60): Pemahaman semantik dan parafrase menggunakan
          sentence embeddings multilingual.
        - BM25   (bobot 0.30): Pencocokan leksikal/keyword berbasis
          statistik term frequency dengan stemming.
        - Aho-Corasick (bobot 0.10): Bonus kehadiran keyword eksak
          menggunakan multi-pattern matching.

    Alur proses:
        1. Memilih dataset dan indeks sesuai kategori (magang/skripsi).
        2. Memperluas query dengan sinonim (expand_query).
        3. Menghitung skor BM25 dari token yang sudah di-stem.
        4. Menghitung skor SBERT dari cosine similarity embedding.
        5. Menghitung skor Aho-Corasick dari pencocokan keyword.
        6. Menggabungkan ketiga skor (hybrid) atau menggunakan satu metode
           saja sesuai parameter `mode`.
        7. Mengurutkan dokumen dari skor tertinggi ke terendah.

    Args:
        query (str): Query pencarian dari pengguna.
        category (str): Kategori dokumen — 'magang' atau 'skripsi'.
            Default: 'magang'.
        mode (str): Mode scoring — 'hybrid', 'bm25', 'sbert', atau 'aho'.
            Default: 'hybrid'.

    Returns:
        tuple: (sorted_indices, component_scores)
            - sorted_indices (list[int]): Indeks dokumen yang sudah diurutkan
              berdasarkan skor dari tinggi ke rendah.
            - component_scores (dict): Dictionary berisi array skor per
              komponen ('bm25', 'sbert', 'aho_corasick', 'final').
            Mengembalikan ([], {}) jika tidak ada dokumen atau indeks belum siap.
    """
    # Select appropriate data based on category
    if category == "magang":
        documents = documents_magang
        bm25 = bm25_magang
        doc_embeddings = doc_embeddings_magang
    else:
        documents = documents_skripsi
        bm25 = bm25_skripsi
        doc_embeddings = doc_embeddings_skripsi

    if not documents or bm25 is None or doc_embeddings is None:
        return [], {}

    n = len(documents)

    # --- Expand query with synonyms ---
    expanded_query = expand_query(query)

    # --- BM25 scores (use stemmed tokens) ---
    bm25_tokens = tokenize_and_stem(expanded_query)
    if bm25_tokens:
        bm25_raw = bm25.get_scores(bm25_tokens)
    else:
        bm25_raw = np.zeros(n)
    bm25_scores = normalize_scores(bm25_raw)

    # --- SBERT scores (use raw query — model handles semantics) ---
    q_emb = model.encode(
        query,
        convert_to_tensor=True
    )

    sbert_raw = util.cos_sim(q_emb, doc_embeddings)[0].cpu().numpy()
    sbert_scores = normalize_scores(sbert_raw)

    # --- Aho-Corasick scores (uses expanded query) ---
    aho_scores = aho_score_documents(expanded_query, documents)
    # Aho scores are already 0–1 from coverage+density formula

    # --- Combine scores ---
    W_BM25 = 0.30
    W_SBERT = 0.60
    W_AHO = 0.10

    if mode == "bm25":
        final_scores = bm25_scores
    elif mode == "sbert":
        final_scores = sbert_scores
    elif mode == "aho":
        final_scores = aho_scores
    else:  # hybrid
        final_scores = (
            W_BM25 * bm25_scores +
            W_SBERT * sbert_scores +
            W_AHO * aho_scores
        )

    # Build result list with indices for sorting
    scored_indices = list(range(n))
    scored_indices.sort(key=lambda i: final_scores[i], reverse=True)

    component_scores = {
        "bm25": bm25_scores,
        "sbert": sbert_scores,
        "aho_corasick": aho_scores,
        "final": final_scores,
    }

    return scored_indices, component_scores

# ==================================================
# SEARCH FUNCTION
# ==================================================
def search(query, category="magang", mode="hybrid", limit=20, min_score=0.3):
    """
    Fungsi utama pencarian dokumen laporan magang/skripsi.

    Melakukan pencarian hybrid (atau single-mode) lalu memfilter
    dan memformat hasilnya berdasarkan threshold skor minimum
    dan batas jumlah hasil.

    Alur proses:
        1. Memilih dataset sesuai kategori.
        2. Memanggil `rank_documents()` untuk mendapatkan ranking.
        3. Memfilter dokumen dengan skor ≥ min_score.
        4. Membatasi jumlah hasil sesuai parameter `limit`.
        5. Mengembalikan list hasil lengkap dengan detail skor per komponen.

    Args:
        query (str): Query pencarian dari pengguna.
        category (str): Kategori dokumen — 'magang' atau 'skripsi'.
            Default: 'magang'.
        mode (str): Mode scoring — 'hybrid', 'bm25', 'sbert', atau 'aho'.
            Default: 'hybrid'.
        limit (int): Jumlah maksimum hasil yang dikembalikan.
            Default: 20.
        min_score (float): Threshold skor minimum. Dokumen dengan skor
            di bawah nilai ini tidak disertakan dalam hasil.
            Default: 0.3.

    Returns:
        list[dict]: Daftar hasil pencarian, setiap item berupa dict:
            - 'id'    (int): ID laporan.
            - 'judul' (str): Judul laporan.
            - 'tipe'  (str): Tipe laporan ('magang' / 'skripsi').
            - 'score' (float): Skor akhir (hybrid/single), 4 desimal.
            - 'detail_scores' (dict): Skor per komponen
                - 'bm25' (float)
                - 'sbert' (float)
                - 'aho_corasick' (float)
            Mengembalikan list kosong jika tidak ada dokumen atau
            tidak ada hasil yang memenuhi threshold.
    """
    if category == "magang":
        documents = documents_magang
    else:
        documents = documents_skripsi

    if not documents:
        return []

    sorted_indices, component_scores = rank_documents(query, category, mode)

    if not sorted_indices:
        return []

    final = component_scores["final"]
    bm25_s = component_scores["bm25"]
    sbert_s = component_scores["sbert"]
    aho_s = component_scores["aho_corasick"]

    results = []
    for idx in sorted_indices:
        score = float(final[idx])

        # Stop early — since sorted descending, no further results will pass
        if score < min_score:
            break

        if len(results) >= limit:
            break

        doc = documents[idx]
        results.append({
            "id": doc["id"],
            "judul": doc["judul"],
            "tipe": doc.get("tipe", category),
            "score": round(score, 4),
            "detail_scores": {
                "bm25": round(float(bm25_s[idx]), 4),
                "aho_corasick": round(float(aho_s[idx]), 4),
                "sbert": round(float(sbert_s[idx]), 4),
            }
        })

    return results

# ==================================================
# INITIALIZE ON MODULE LOAD
# ==================================================
reload_documents()

# ==================================================
# MAIN (for testing)
# ==================================================
if __name__ == "__main__":
    test_queries = [
        "analisis perkara pidana",
        "machine learning klasifikasi",
        "sistem informasi",
        "pengaruh media sosial",
    ]

    for query in test_queries:
        print(f"\n{'='*60}")
        print(f"Query: '{query}'")
        print(f"Expanded: '{expand_query(query)}'")
        print(f"{'='*60}")

        for cat in ["magang", "skripsi"]:
            print(f"\n--- {cat.upper()} Results ---")
            results = search(query, category=cat, limit=5)
            if not results:
                print("  (no results)")
            for i, r in enumerate(results, 1):
                d = r['detail_scores']
                print(
                    f"  {i}. [{r['score']:.4f}] {r['judul']}\n"
                    f"     BM25={d['bm25']:.4f}  "
                    f"SBERT={d['sbert']:.4f}  "
                    f"Aho={d['aho_corasick']:.4f}"
                )
