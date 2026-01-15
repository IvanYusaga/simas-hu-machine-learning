from sentence_transformers import SentenceTransformer, util
from rank_bm25 import BM25Okapi
import ahocorasick
import numpy as np
import json
import os

from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory

# ==================================================
# STOPWORD
# ==================================================
factory = StopWordRemoverFactory()
STOPWORDS = set(factory.get_stop_words())

# ==================================================
# DATA (Load dari file JSON)
# ==================================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
with open(os.path.join(SCRIPT_DIR, "documents.json"), "r", encoding="utf-8") as f:
    documents = json.load(f)

# ==================================================
# PREPROCESSING
# ==================================================
def preprocess_basic(text):
    return text.lower().split()

def preprocess_for_aho(text):
    return [t for t in text.lower().split() if t not in STOPWORDS]

# ==================================================
# BM25
# ==================================================
tokenized_docs = [preprocess_basic(d["judul"]) for d in documents]
bm25 = BM25Okapi(tokenized_docs)

# ==================================================
# SBERT
# ==================================================
model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
doc_embeddings = model.encode(
    [d["judul"] for d in documents],
    convert_to_tensor=True
)

# ==================================================
# AHO
# ==================================================
def build_aho_automaton(keywords):
    A = ahocorasick.Automaton()
    for i, w in enumerate(keywords):
        A.add_word(w, (i, w))
    A.make_automaton()
    return A

# ==================================================
# RANKING (INTI)
# ==================================================
def rank_documents(query, mode="hybrid"):
    # ----- BM25 -----
    bm25_tokens = preprocess_basic(query)
    bm25_raw = bm25.get_scores(bm25_tokens)
    bm25_scores = (bm25_raw - bm25_raw.min()) / (bm25_raw.max() - bm25_raw.min() + 1e-9)

    # ----- SBERT -----
    q_emb = model.encode(query, convert_to_tensor=True)
    sbert_raw = util.cos_sim(q_emb, doc_embeddings)[0].cpu().numpy()
    sbert_scores = (sbert_raw - sbert_raw.min()) / (sbert_raw.max() - sbert_raw.min() + 1e-9)

    # ----- AHO -----
    aho_tokens = preprocess_for_aho(query)
    automaton = build_aho_automaton(aho_tokens)

    aho_scores = []
    for doc in documents:
        found = False
        for _, _ in automaton.iter(doc["judul"].lower()):
            found = True
            break
        aho_scores.append(1.0 if found else 0.0)

    # ----- COMBINE -----
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
                0.5 * bm25_scores[i] +
                0.3 * aho_scores[i] +
                0.2 * sbert_scores[i]
            )
        scores.append((doc["id"], score))

    scores.sort(key=lambda x: x[1], reverse=True)
    return scores, bm25_scores, aho_scores, sbert_scores

# ==================================================
# SEARCH (UNTUK test_search.py)
# ==================================================
def search(query, mode="hybrid"):
    ranked, bm25_s, aho_s, sbert_s = rank_documents(query, mode)

    results = []
    for doc_id, _ in ranked:
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
                0.5 * bm25_s[idx] +
                0.3 * aho_s[idx] +
                0.2 * sbert_s[idx]
            )

        results.append({
            "id": doc["id"],
            "judul": doc["judul"],
            "score": float(score),
            "detail_scores": {
                "bm25": float(bm25_s[idx]),
                "aho_corasick": float(aho_s[idx]),
                "sbert": float(sbert_s[idx]),
            }
        })

    return results

# ==================================================
# METRIK EVALUASI
# ==================================================
def precision_at_k(ranked_ids, relevant, k):
    ranked_k = ranked_ids[:k]
    return sum(1 for d in ranked_k if d in relevant) / k

def recall_at_k(ranked_ids, relevant, k):
    ranked_k = ranked_ids[:k]
    return sum(1 for d in ranked_k if d in relevant) / len(relevant)

def evaluate_model(queries, ground_truth, mode, k_values=[5, 10, 15]):
    results = {}
    
    for k in k_values:
        p_total, r_total = 0, 0

        for qid, query in queries.items():
            ranked, _, _, _ = rank_documents(query, mode)
            ranked_ids = [doc_id for doc_id, _ in ranked]
            relevant = ground_truth[qid]

            p_total += precision_at_k(ranked_ids, relevant, k)
            r_total += recall_at_k(ranked_ids, relevant, k)

        n = len(queries)
        results[f"Precision@{k}"] = p_total / n
        results[f"Recall@{k}"] = r_total / n
    
    return results

# ==================================================
# MAIN (EVALUASI)
# ==================================================
if __name__ == "__main__":
    queries = {
        "Q1": "administrasi perkara di pengadilan",
        # "Q2": "penanganan perkara pidana",
    }

    ground_truth = {
        "Q1": {13, 82, 19, 37, 18},
        # "Q2": {1, 12, 16, 20, 18},
    }

    for mode in ["bm25", "sbert", "aho", "hybrid"]:
        result = evaluate_model(queries, ground_truth, mode, k_values=[5, 10, 15])
        print(f"\nModel: {mode.upper()}")
        for k, v in result.items():
            print(f"{k}: {v:.2f}")
