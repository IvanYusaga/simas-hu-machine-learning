from sentence_transformers import SentenceTransformer, util
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
from rank_bm25 import BM25Okapi
import ahocorasick
import numpy as np

# ===== STOPWORD (Bahasa Indonesia) =====
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory

stopword_factory = StopWordRemoverFactory()
STOPWORDS = set(stopword_factory.get_stop_words())

# DATA CONTOH (sementara)
documents = [
    {"id": 1, "judul": "Analisis Proses Penanganan Perkara Pidana Umum di Pengadilan Negeri"},
    {"id": 2, "judul": "Peran Jaksa Penuntut Umum dalam Tahap Penuntutan Tindak Pidana"},
    {"id": 3, "judul": "Pelaksanaan Bantuan Hukum bagi Masyarakat Tidak Mampu di Lembaga Bantuan Hukum"},
    {"id": 4, "judul": "Tinjauan Yuridis Proses Penyidikan Tindak Pidana oleh Kepolisian"},
    {"id": 5, "judul": "Implementasi Mediasi sebagai Alternatif Penyelesaian Sengketa Perdata"},
    {"id": 6, "judul": "Peranan Notaris dalam Pembuatan dan Pengesahan Akta Otentik"},
    {"id": 7, "judul": "Prosedur Pendaftaran dan Peralihan Hak Atas Tanah di Kantor Pertanahan"},
    {"id": 8, "judul": "Perlindungan Hukum terhadap Anak yang Berhadapan dengan Hukum"},
    {"id": 9, "judul": "Analisis Penyusunan Perjanjian Kerja antara Perusahaan dan Pekerja"},
    {"id": 10, "judul": "Proses Penyelesaian Sengketa Hubungan Industrial di Dinas Tenaga Kerja"},
    {"id": 11, "judul": "Peran Advokat dalam Pendampingan Hukum Perkara Pidana"},
    {"id": 12, "judul": "Penerapan Asas Legalitas dalam Penanganan Perkara Pidana"},
    {"id": 13, "judul": "Mekanisme Administrasi Perkara di Pengadilan Negeri"},
    {"id": 14, "judul": "Prosedur Pengajuan Gugatan Perdata melalui Sistem E-Court"},
    {"id": 15, "judul": "Perlindungan Hukum Konsumen terhadap Produk Cacat"},
    {"id": 16, "judul": "Analisis Penanganan Perkara Kekerasan Dalam Rumah Tangg"},
    {"id": 17, "judul": "Peran Kejaksaan dalam Pelaksanaan Eksekusi Putusan Pengadilan"},
    {"id": 18, "judul": "Proses Registrasi dan Penjadwalan Perkara Pidana di Pengadilan"},
    {"id": 19, "judul": "Pelaksanaan Tugas Panitera dalam Administrasi Persidangan"},
    {"id": 20, "judul": "Penerapan Keadilan Restoratif dalam Penyelesaian Perkara Pidana Ringan"},
]

def preprocess(text):
    return text.lower().split()

def preprocess_for_aho(text):
    return [
        t for t in text.lower().split()
        if t not in STOPWORDS
    ]

# ===== BM25 =====
tokenized_docs = [preprocess(d["judul"]) for d in documents]
bm25 = BM25Okapi(tokenized_docs)

# ===== SBERT (model multilingual untuk Bahasa Indonesia) =====
model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
doc_embeddings = model.encode(
    [d["judul"] for d in documents],
    convert_to_tensor=True
)

# ===== Aho-Corasick =====
def build_aho_automaton(keywords):
    A = ahocorasick.Automaton()
    for idx, word in enumerate(keywords):
        A.add_word(word, (idx, word))
    A.make_automaton()
    return A

# ===== Search =====
def search(query):
    """
    Melakukan pencarian dengan menggabungkan skor dari:
    - BM25 (lexical matching)
    - Aho-Corasick (exact keyword matching)
    - SBERT (semantic similarity)
    
    Bobot default: BM25=0.5, Aho-Corasick=0.1, SBERT=0.4
    """
    # Validasi input
    if not query or not query.strip():
        return []
    
    tokens = preprocess(query)
    
    if not tokens:
        return []

    # ===== BM25 Scoring =====
    bm25_raw = bm25.get_scores(tokens)
    bm25_scores = (bm25_raw - bm25_raw.min()) / (bm25_raw.max() - bm25_raw.min() + 1e-9)

    # ===== SBERT Scoring =====
    query_embedding = model.encode(query, convert_to_tensor=True)
    sbert_raw = util.cos_sim(query_embedding, doc_embeddings)[0].cpu().numpy()
    sbert_scores = (sbert_raw - sbert_raw.min()) / (sbert_raw.max() - sbert_raw.min() + 1e-9)

    # ===== Aho-Corasick Scoring =====
    aho_tokens = preprocess_for_aho(query)
    automaton = build_aho_automaton(aho_tokens)
    aho_scores = []

    for doc in documents:
        text = doc["judul"].lower()
        found = False

        for _, _ in automaton.iter(text):
            found = True
            break

        aho_scores.append(1.0 if found else 0.0)

    # ===== Combine Scores dengan Weighted Sum =====
    # Bobot bisa disesuaikan berdasarkan kebutuhan
    WEIGHT_AHO = 0.2
    WEIGHT_BM25 = 0.4
    WEIGHT_SBERT = 0.4

    results = []
    for i, doc in enumerate(documents):
        combined_score = (
            WEIGHT_BM25 * bm25_scores[i] +
            WEIGHT_AHO * aho_scores[i] +
            WEIGHT_SBERT * sbert_scores[i]
        )

        results.append({
            "id": doc["id"],
            "judul": doc["judul"],
            "score": float(combined_score),
            # Detail skor untuk debugging/analisis
            "detail_scores": {
                "bm25": float(bm25_scores[i]),
                "aho_corasick": float(aho_scores[i]),
                "sbert": float(sbert_scores[i])
            }
        })

    # Urutkan berdasarkan skor tertinggi
    return sorted(results, key=lambda x: x["score"], reverse=True)


# ===== Testing =====
if __name__ == "__main__":
    test_queries = [
        "sistem informasi",
        "analisis sentimen",
        "pencarian skripsi",
        "web aplikasi",
        "machine learning"
    ]
    
    for q in test_queries:
        print(f"\n{'='*50}")
        print(f"Query: '{q}'")
        print(f"{'='*50}")
        results = search(q)
        for rank, r in enumerate(results, 1):
            print(f"{rank}. {r['judul']}")
            print(f"   Score: {r['score']:.4f}")
            print(f"   Detail: BM25={r['detail_scores']['bm25']:.4f}, "
                  f"Aho={r['detail_scores']['aho_corasick']:.4f}, "
                  f"SBERT={r['detail_scores']['sbert']:.4f}")
