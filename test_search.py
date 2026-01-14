import warnings
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress TensorFlow warnings
warnings.filterwarnings('ignore')

from search import search

print("=" * 60)
print("TESTING SEARCH ENGINE")
print("=" * 60)

test_queries = [
    "sistem informasi",
    "analisis sentimen",
    "pencarian skripsi",
]

for q in test_queries:
    print(f"\nQuery: '{q}'")
    print("-" * 40)
    results = search(q)
    for rank, r in enumerate(results, 1):
        print(f"{rank}. {r['judul']}")
        print(f"   Score: {r['score']:.4f}")
        d = r['detail_scores']
        print(f"   BM25={d['bm25']:.4f}, Aho={d['aho_corasick']:.4f}, SBERT={d['sbert']:.4f}")
