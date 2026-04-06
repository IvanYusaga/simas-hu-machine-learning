from sentence_transformers import SentenceTransformer, util
from rank_bm25 import BM25Okapi
import ahocorasick
import numpy as np
import re
import os
import pymysql
from dotenv import load_dotenv

from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
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
# NLP TOOLS (Load once)
# ==================================================
_stopword_factory = StopWordRemoverFactory()
STOPWORDS = set(_stopword_factory.get_stop_words())

_stemmer_factory = StemmerFactory()
stemmer = _stemmer_factory.create_stemmer()

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
    2. Are leaf nodes (no children)
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
# PREPROCESSING
# ==================================================
_RE_NON_ALPHANUM = re.compile(r'[^a-z0-9\s]')

def tokenize(text):
    """Lowercase, strip punctuation, split into tokens."""
    text = text.lower()
    text = _RE_NON_ALPHANUM.sub(' ', text)
    return text.split()

def tokenize_and_stem(text):
    """Tokenize, remove stopwords, and apply Indonesian stemming."""
    tokens = tokenize(text)
    result = []
    for t in tokens:
        if t in STOPWORDS or len(t) <= 1:
            continue
        stemmed = stemmer.stem(t)
        if stemmed:
            result.append(stemmed)
    return result

def tokenize_remove_stopwords(text):
    """Tokenize and remove stopwords (no stemming)."""
    tokens = tokenize(text)
    return [t for t in tokens if t not in STOPWORDS and len(t) > 1]

# ==================================================
# QUERY EXPANSION
# ==================================================
def expand_query(query):
    """
    Expand query with synonyms to improve recall.
    Returns the expanded query string.
    """
    query_lower = query.lower()
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
    """Build BM25 index and SBERT embeddings for documents."""
    if not documents:
        return None, None, []

    # BM25 — use stemmed tokens for better matching
    tokenized_docs = [tokenize_and_stem(d["judul"]) for d in documents]
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
    """Build Aho-Corasick automaton from keyword list."""
    A = ahocorasick.Automaton()
    for i, w in enumerate(keywords):
        A.add_word(w, (i, w))
    A.make_automaton()
    return A

def aho_score_documents(query, documents):
    """
    Score documents using Aho-Corasick multi-pattern matching.
    
    Improvements over original:
    - Uses stemmed keywords for better root-word matching
    - Word-boundary checking to prevent partial matches
    - Scores incorporate unique keyword coverage ratio
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
        doc_text = doc["judul"].lower()
        doc_text_stemmed = stemmer.stem(doc_text)
        # Combine original + stemmed doc text for matching
        combined_text = doc_text + " " + doc_text_stemmed

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
    Robust min-max normalization.
    Handles edge cases: empty arrays, all-zero, single element.
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
    """Reload documents from database and rebuild search indices."""
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
    Rank documents by relevance to query.
    
    Hybrid weights (tuned for Indonesian academic titles):
      - SBERT  : 0.45 — best at understanding meaning & paraphrases
      - BM25   : 0.35 — strong lexical/keyword match with stemming
      - Aho-C  : 0.20 — exact keyword presence bonus
    
    Returns: (results_list, raw_component_scores_dict)
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
    q_emb = model.encode(query, convert_to_tensor=True)
    sbert_raw = util.cos_sim(q_emb, doc_embeddings)[0].cpu().numpy()
    sbert_scores = normalize_scores(sbert_raw)

    # --- Aho-Corasick scores (uses expanded query) ---
    aho_scores = aho_score_documents(expanded_query, documents)
    # Aho scores are already 0–1 from coverage+density formula

    # --- Combine scores ---
    W_BM25 = 0.35
    W_SBERT = 0.45
    W_AHO = 0.20

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
    Main search function.

    Args:
        query: Search query string
        category: "magang" or "skripsi"
        mode: "hybrid", "bm25", "sbert", or "aho"
        limit: Maximum number of results to return
        min_score: Minimum score threshold (default: 0.3)

    Returns:
        List of search results with scores >= min_score
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
