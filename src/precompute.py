"""
precompute.py
Offline pre-computation step (network OK, no time limit here).

Produces:
  - data/candidate_text.jsonl  : one short text blob per candidate (for embedding)
  - data/embeddings.npy        : (N, D) float32 embedding matrix
  - data/candidate_ids.json    : list of candidate_ids matching embeddings row order
  - data/jd_embedding.npy       : (D,) embedding for the JD requirements text

Usage:
  python src/precompute.py --candidates data/candidates.jsonl --out-dir data/
"""

import argparse
import json
import os
import sys

import numpy as np


def candidate_to_text(c):
    profile = c["profile"]
    parts = [
        profile.get("current_title", ""),
        profile.get("headline", ""),
        profile.get("summary", ""),
    ]
    for role in c.get("career_history", [])[:4]:
        parts.append(f"{role.get('title','')} at {role.get('company','')}: {role.get('description','')}")
    skill_names = [s["name"] for s in c.get("skills", [])]
    if skill_names:
        parts.append("Skills: " + ", ".join(skill_names))
    return " ".join(p for p in parts if p)


def load_candidates(path):
    candidates = []
    opener = open
    if path.endswith(".gz"):
        import gzip
        opener = gzip.open
    with opener(path, "rt", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                candidates.append(json.loads(line))
    return candidates


def get_embedder():
    """Returns a function text -> np.ndarray, using sentence-transformers if
    available, else falling back to TF-IDF + SVD (sklearn, always available)."""
    try:
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer("all-MiniLM-L6-v2")

        def embed(texts):
            return np.asarray(model.encode(texts, batch_size=64, show_progress_bar=True, normalize_embeddings=True), dtype=np.float32)

        return embed, "sentence-transformers/all-MiniLM-L6-v2"
    except ImportError:
        print("[precompute] sentence-transformers not available, falling back to TF-IDF+SVD", file=sys.stderr)
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.decomposition import TruncatedSVD
        from sklearn.preprocessing import normalize

        vectorizer = TfidfVectorizer(max_features=20000, stop_words="english", ngram_range=(1, 2))
        svd = TruncatedSVD(n_components=256, random_state=42)

        state = {}

        def embed(texts):
            if "fitted" not in state:
                tfidf = vectorizer.fit_transform(texts)
                emb = svd.fit_transform(tfidf)
                state["fitted"] = True
            else:
                tfidf = vectorizer.transform(texts)
                emb = svd.transform(tfidf)
            emb = normalize(emb)
            return emb.astype(np.float32)

        return embed, "tfidf-svd-256"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidates", required=True, help="Path to candidates.jsonl or .jsonl.gz")
    ap.add_argument("--jd-profile", default="data/jd_profile.json")
    ap.add_argument("--out-dir", default="data")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    print("[precompute] loading candidates...")
    candidates = load_candidates(args.candidates)
    print(f"[precompute] loaded {len(candidates)} candidates")

    texts = [candidate_to_text(c) for c in candidates]
    ids = [c["candidate_id"] for c in candidates]

    with open(args.jd_profile) as f:
        jd = json.load(f)
    jd_text = jd["jd_semantic_text"]

    embed, model_name = get_embedder()
    print(f"[precompute] embedding with: {model_name}")

    # IMPORTANT: for TF-IDF fallback, fit on candidates+JD together so vocab is shared
    if model_name == "tfidf-svd-256":
        all_texts = texts + [jd_text]
        all_emb = embed(all_texts)
        cand_emb = all_emb[:-1]
        jd_emb = all_emb[-1]
    else:
        cand_emb = embed(texts)
        jd_emb = embed([jd_text])[0]

    np.save(os.path.join(args.out_dir, "embeddings.npy"), cand_emb)
    np.save(os.path.join(args.out_dir, "jd_embedding.npy"), jd_emb)
    with open(os.path.join(args.out_dir, "candidate_ids.json"), "w") as f:
        json.dump(ids, f)
    with open(os.path.join(args.out_dir, "embedding_meta.json"), "w") as f:
        json.dump({"model": model_name, "n_candidates": len(ids), "dim": int(cand_emb.shape[1])}, f, indent=2)

    print(f"[precompute] done. embeddings shape={cand_emb.shape}")


if __name__ == "__main__":
    main()
