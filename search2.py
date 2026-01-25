from sentence_transformers import SentenceTransformer, util
from rank_bm25 import BM25Okapi
import ahocorasick
import numpy as np
import os
import pymysql
from dotenv import load_dotenv

from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory

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
# STOPWORD
# ==================================================
factory = StopWordRemoverFactory()
STOPWORDS = set(factory.get_stop_words())

# ==================================================
# SBERT MODEL (Load once)
# ==================================================
model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")

# ==================================================
# GLOBAL DATA STORAGE
# ==================================================
documents_magang = []
documents_skripsi = []
bm25_magang = None
bm25_skripsi = None
doc_embeddings_magang = None
doc_embeddings_skripsi = None

# ==================================================
# DATABASE CONNECTION
# ==================================================
def get_db_connection():
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
    Load documents from MySQL database.
    Only include laporan that:
    1. Have at least one record in verifikasi table (sudah diajukan)
    2. Don't have 'daftar ulang' status
    """
    global documents_magang, documents_skripsi
    
    conn = get_db_connection()
    try:
        with conn.cursor() as cursor:
            # Query untuk Laporan Magang/PKL
            # - HARUS ada record di tabel verifikasi (sudah diajukan)
            # - TIDAK BOLEH ada status 'daftar ulang'
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
                    SELECT 1 FROM verifikasi_laporan_magang_pkl v
                    WHERE v.id_laporan_magang_pkl = lm.id_laporan_magang_pkl
                    AND v.status_verifikasi = 'daftar ulang'
                )
            """
            cursor.execute(query_magang)
            documents_magang = cursor.fetchall()
            
            # Query untuk Laporan Skripsi
            # - HARUS ada record di tabel verifikasi (sudah diajukan)
            # - TIDAK BOLEH ada status 'daftar ulang'
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
                    SELECT 1 FROM verifikasi_laporan_skripsi v
                    WHERE v.id_laporan_skripsi = ls.id_laporan_skripsi
                    AND v.status_verifikasi = 'daftar ulang'
                )
            """
            cursor.execute(query_skripsi)
            documents_skripsi = cursor.fetchall()
            
    finally:
        conn.close()
    
    return documents_magang, documents_skripsi

# ==================================================
# PREPROCESSING
# ==================================================
def preprocess_basic(text):
    return text.lower().split()

def preprocess_for_aho(text):
    return [t for t in text.lower().split() if t not in STOPWORDS]

# ==================================================
# BUILD SEARCH INDEX
# ==================================================
def build_search_index(documents):
    """Build BM25 index and SBERT embeddings for documents"""
    if not documents:
        return None, None
    
    # BM25
    tokenized_docs = [preprocess_basic(d["judul"]) for d in documents]
    bm25 = BM25Okapi(tokenized_docs)
    
    # SBERT
    doc_embeddings = model.encode(
        [d["judul"] for d in documents],
        convert_to_tensor=True
    )
    
    return bm25, doc_embeddings

# ==================================================
# AHO-CORASICK
# ==================================================
def build_aho_automaton(keywords):
    A = ahocorasick.Automaton()
    for i, w in enumerate(keywords):
        A.add_word(w, (i, w))
    A.make_automaton()
    return A

def aho_count_scores(query, documents):
    if not documents:
        return np.array([])
    
    keywords = preprocess_for_aho(query)
    if not keywords:
        return np.zeros(len(documents))
    
    automaton = build_aho_automaton(keywords)
    scores = []
    
    for doc in documents:
        count = 0
        for _, (_, matched_word) in automaton.iter(doc["judul"].lower()):
            count += 1
        scores.append(count)
    
    scores = np.array(scores, dtype=float)
    if scores.max() > 0:
        scores = scores / scores.max()
    
    return scores

# ==================================================
# RELOAD DOCUMENTS (Called on startup and when needed)
# ==================================================
def reload_documents():
    """Reload documents from database and rebuild search indices"""
    global documents_magang, documents_skripsi
    global bm25_magang, bm25_skripsi
    global doc_embeddings_magang, doc_embeddings_skripsi
    
    print("Loading documents from database...")
    load_documents_from_db()
    
    print(f"Building search index for {len(documents_magang)} magang documents...")
    bm25_magang, doc_embeddings_magang = build_search_index(documents_magang)
    
    print(f"Building search index for {len(documents_skripsi)} skripsi documents...")
    bm25_skripsi, doc_embeddings_skripsi = build_search_index(documents_skripsi)
    
    print("Search index ready!")

# ==================================================
# RANKING
# ==================================================
def rank_documents(query, category="magang", mode="hybrid"):
    """Rank documents by relevance to query"""
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
        return [], None, None, None
    
    # BM25 scores
    bm25_tokens = preprocess_basic(query)
    bm25_raw = bm25.get_scores(bm25_tokens)
    bm25_scores = (bm25_raw - bm25_raw.min()) / (bm25_raw.max() - bm25_raw.min() + 1e-9)
    
    # SBERT scores
    q_emb = model.encode(query, convert_to_tensor=True)
    sbert_raw = util.cos_sim(q_emb, doc_embeddings)[0].cpu().numpy()
    sbert_scores = (sbert_raw - sbert_raw.min()) / (sbert_raw.max() - sbert_raw.min() + 1e-9)
    
    # Aho-Corasick scores
    aho_scores = aho_count_scores(query, documents)
    
    # Combine scores
    scores = []
    for i, doc in enumerate(documents):
        if mode == "bm25":
            score = bm25_scores[i]
        elif mode == "sbert":
            score = sbert_scores[i]
        elif mode == "aho":
            score = aho_scores[i]
        else:  # hybrid
            score = (
                0.6 * bm25_scores[i] +
                0.2 * aho_scores[i] +
                0.2 * sbert_scores[i]
            )
        scores.append((doc["id"], score))
    
    scores.sort(key=lambda x: x[1], reverse=True)
    return scores, bm25_scores, aho_scores, sbert_scores

# ==================================================
# SEARCH FUNCTION
# ==================================================
def search(query, category="magang", mode="hybrid", limit=20, min_score=0.5):
    """
    Main search function
    
    Args:
        query: Search query string
        category: "magang" or "skripsi"
        mode: "hybrid", "bm25", "sbert", or "aho"
        limit: Maximum number of results to return
        min_score: Minimum score threshold (default: 0.5)
    
    Returns:
        List of search results with scores >= min_score
    """
    if category == "magang":
        documents = documents_magang
    else:
        documents = documents_skripsi
    
    if not documents:
        return []
    
    ranked, bm25_s, aho_s, sbert_s = rank_documents(query, category, mode)
    
    results = []
    for doc_id, _ in ranked[:limit]:
        idx = next(i for i, d in enumerate(documents) if d["id"] == doc_id)
        doc = documents[idx]
        
        if mode == "bm25":
            score = bm25_s[idx]
        elif mode == "sbert":
            score = sbert_s[idx]
        elif mode == "aho":
            score = aho_s[idx]
        else:
            score = (
                0.6 * bm25_s[idx] +
                0.2 * aho_s[idx] +
                0.2 * sbert_s[idx]
            )
        
        # Filter: only include results with score >= min_score
        if score < min_score:
            continue
        
        results.append({
            "id": doc["id"],
            "judul": doc["judul"],
            "tipe": doc.get("tipe", category),
            "score": float(score),
            "detail_scores": {
                "bm25": float(bm25_s[idx]),
                "aho_corasick": float(aho_s[idx]),
                "sbert": float(sbert_s[idx]),
            }
        })
    
    return results

# ==================================================
# INITIALIZE ON MODULE LOAD
# ==================================================
# Load documents when module is first imported
reload_documents()

# ==================================================
# MAIN (for testing)
# ==================================================
if __name__ == "__main__":
    # Test search
    query = "analisis perkara pidana"
    
    print(f"\n=== Searching for: '{query}' ===\n")
    
    print("--- Magang Results ---")
    results_magang = search(query, category="magang")
    for i, r in enumerate(results_magang[:5], 1):
        print(f"{i}. [{r['score']:.3f}] {r['judul']}")
    
    print("\n--- Skripsi Results ---")
    results_skripsi = search(query, category="skripsi")
    for i, r in enumerate(results_skripsi[:5], 1):
        print(f"{i}. [{r['score']:.3f}] {r['judul']}")
