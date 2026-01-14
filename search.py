from sentence_transformers import SentenceTransformer, util
from rank_bm25 import BM25Okapi
import ahocorasick
import numpy as np

# DATA CONTOH (sementara)
documents = [
    {"id": 1, "judul": "Sistem Informasi Magang Berbasis Web"},
    {"id": 2, "judul": "Analisis Sentimen Twitter Menggunakan BERT"},
    {"id": 3, "judul": "Aplikasi Pencarian Judul Skripsi dengan BM25"},
]


def preprocess(text):
    """
    Preprocessing teks: lowercase dan tokenisasi.
    Bisa ditambahkan stemming/stopword removal di sini.
    """
    return text.lower().split()


def normalize(scores):
    """
    Normalisasi skor ke rentang 0-1.
    Diperlukan karena BM25, Aho-Corasick, dan SBERT memiliki skala berbeda.
    """
    arr = np.array(scores, dtype=float)
    if arr.max() == arr.min():
        return np.zeros_like(arr)
    return (arr - arr.min()) / (arr.max() - arr.min())


# ===== BM25 =====
tokenized_docs = [preprocess(d["judul"]) for d in documents]
bm25 = BM25Okapi(tokenized_docs)

# ===== AHO-CORASICK (per kata, bukan per judul) =====
A = ahocorasick.Automaton()
for i, doc in enumerate(documents):
    for word in preprocess(doc["judul"]):
        # Tambahkan setiap kata sebagai pattern, bukan seluruh judul
        A.add_word(word, i)
A.make_automaton()

# ===== SBERT (model multilingual untuk Bahasa Indonesia) =====
model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
doc_embeddings = model.encode(
    [d["judul"] for d in documents],
    convert_to_tensor=True
)


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
    bm25_scores = normalize(bm25_raw)

    # ===== SBERT Scoring =====
    query_embedding = model.encode(query, convert_to_tensor=True)
    sbert_raw = util.cos_sim(query_embedding, doc_embeddings)[0]
    sbert_scores = normalize(sbert_raw.cpu().numpy())

    # ===== Aho-Corasick Scoring =====
    aho_scores_raw = [0] * len(documents)
    for _, idx in A.iter(query.lower()):
        aho_scores_raw[idx] += 1
    aho_scores = normalize(aho_scores_raw)

    # ===== Combine Scores dengan Weighted Sum =====
    # Bobot bisa disesuaikan berdasarkan kebutuhan
    WEIGHT_BM25 = 0.5
    WEIGHT_AHO = 0.1
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
