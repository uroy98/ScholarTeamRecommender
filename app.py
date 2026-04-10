import json
import os
import traceback
from pathlib import Path

import numpy as np
from flask import Flask, jsonify, render_template, request
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
INDEX_PATH = DATA_DIR / "scholar_index.json"
EMBED_PATH = DATA_DIR / "scholar_embeddings.npy"
IDS_PATH = DATA_DIR / "scholar_ids.npy"

# Change this once, and your name will appear on the top right of the page.
USER_DISPLAY_NAME = "Upasana"

app = Flask(__name__)


class ScholarRecommender:
    def __init__(self):
        self.records = []
        self.record_ids = []
        self.profile_texts = []
        self.vectorizer = None
        self.tfidf_matrix = None
        self.embeddings = None
        self.embeddings_norm = None
        self.embedder = None
        self.embedding_mode = "lexical_only"
        self.embedding_warning = None
        self.id_to_record_index = {}
        self.id_to_embedding_index = {}
        self._load_data()

    def _load_data(self):
        if not INDEX_PATH.exists():
            raise FileNotFoundError(
                f"{INDEX_PATH} not found. Run prepare_data.py first."
            )

        with open(INDEX_PATH, "r", encoding="utf-8") as f:
            self.records = json.load(f)

        self.record_ids = [str(r["scholar_id"]) for r in self.records]
        self.id_to_record_index = {sid: i for i, sid in enumerate(self.record_ids)}
        self.profile_texts = [r.get("profile_text", "") for r in self.records]

        self.vectorizer = TfidfVectorizer(stop_words="english", max_features=20000)
        self.tfidf_matrix = self.vectorizer.fit_transform(self.profile_texts)

        if EMBED_PATH.exists() and IDS_PATH.exists():
            self.embeddings = np.load(EMBED_PATH)
            emb_ids = np.load(IDS_PATH, allow_pickle=True)
            emb_ids = [str(x) for x in emb_ids.tolist()]
            self.id_to_embedding_index = {sid: i for i, sid in enumerate(emb_ids)}

            if self.embeddings.ndim != 2:
                self.embedding_warning = "Embeddings file exists, but it is not a 2D matrix. Falling back to lexical ranking."
                self.embeddings = None
                self.embeddings_norm = None
                return

            norms = np.linalg.norm(self.embeddings, axis=1, keepdims=True)
            norms = np.clip(norms, 1e-12, None)
            self.embeddings_norm = self.embeddings / norms

            # Try to load a sentence-transformers query encoder.
            # This only works directly when the scholar embeddings were generated
            # using the same text-embedding model.
            try:
                from sentence_transformers import SentenceTransformer

                self.embedder = SentenceTransformer("all-MiniLM-L6-v2")
                dummy = self.embedder.encode(["test"], normalize_embeddings=True)
                query_dim = int(dummy.shape[1])
                scholar_dim = int(self.embeddings.shape[1])
                if query_dim == scholar_dim:
                    self.embedding_mode = "hybrid"
                else:
                    self.embedding_mode = "lexical_only"
                    self.embedding_warning = (
                        "Embedding dimension mismatch: your scholar embeddings are "
                        f"{scholar_dim}-dimensional, but the built-in query encoder outputs "
                        f"{query_dim}-dimensional vectors. The UI still works, but it is using lexical ranking only. "
                        "To enable embedding similarity, generate scholar embeddings with build_embeddings.py, "
                        "or replace embed_query() with your own query-to-scholar-space encoder."
                    )
            except Exception as e:
                self.embedding_mode = "lexical_only"
                self.embedding_warning = (
                    "Could not load sentence-transformers. The UI is using lexical ranking only. "
                    f"Details: {str(e)}"
                )
        else:
            self.embedding_warning = (
                "Embeddings not found. Place scholar_embeddings.npy and scholar_ids.npy inside the data folder, "
                "or run build_embeddings.py. The UI is currently using lexical ranking only."
            )

    def embed_query(self, query_text: str):
        if self.embedding_mode != "hybrid" or self.embedder is None:
            return None
        vec = self.embedder.encode([query_text], normalize_embeddings=True)
        return vec[0]

    def lexical_scores(self, query_text: str):
        q = self.vectorizer.transform([query_text])
        return cosine_similarity(q, self.tfidf_matrix).ravel()

    def embedding_scores(self, query_text: str):
        q_emb = self.embed_query(query_text)
        if q_emb is None or self.embeddings_norm is None:
            return None

        aligned_scores = np.full(len(self.records), -1.0, dtype=np.float32)
        for sid, record_idx in self.id_to_record_index.items():
            emb_idx = self.id_to_embedding_index.get(sid)
            if emb_idx is not None:
                aligned_scores[record_idx] = float(self.embeddings_norm[emb_idx] @ q_emb)
        return aligned_scores

    @staticmethod
    def _normalize_scores(scores):
        scores = np.asarray(scores, dtype=np.float32)
        finite_mask = np.isfinite(scores)
        if not finite_mask.any():
            return np.zeros_like(scores)
        valid = scores[finite_mask]
        mn, mx = float(valid.min()), float(valid.max())
        out = np.zeros_like(scores)
        if abs(mx - mn) < 1e-12:
            out[finite_mask] = 1.0
        else:
            out[finite_mask] = (valid - mn) / (mx - mn)
        return out

    def build_match_reasons(self, record, query_text: str, max_items=3):
        tokens = [t.lower() for t in query_text.split() if len(t) > 2]
        reasons = []

        for award in record.get("awards", [])[:20]:
            title = award.get("title", "")
            text = f"{title} {' '.join(award.get('keywords', []))}".lower()
            if any(tok in text for tok in tokens):
                reasons.append(f"Award: {title}")
            if len(reasons) >= max_items:
                return reasons

        for paper in record.get("papers", [])[:20]:
            title = paper.get("title", "")
            text = f"{title} {paper.get('venue', '')}".lower()
            if any(tok in text for tok in tokens):
                reasons.append(f"Paper: {title}")
            if len(reasons) >= max_items:
                return reasons

        if not reasons:
            if record.get("primary_affiliation"):
                reasons.append(f"Affiliation: {record['primary_affiliation']}")
            if record.get("research_keywords"):
                reasons.append(
                    "Keywords: " + ", ".join(record["research_keywords"][:5])
                )

        return reasons[:max_items]

    def _get_candidate_vectors_for_diversity(self, candidate_indices):
        if self.embeddings_norm is None:
            return None
        vecs = []
        valid_positions = []
        for idx in candidate_indices:
            sid = self.records[idx]["scholar_id"]
            emb_idx = self.id_to_embedding_index.get(str(sid))
            if emb_idx is None:
                vecs.append(None)
            else:
                vecs.append(self.embeddings_norm[emb_idx])
                valid_positions.append(idx)
        return vecs

    def recommend_team(self, query_text: str, team_size: int = 5, lambda_relevance: float = 0.75):
        lex = self.lexical_scores(query_text)
        emb = self.embedding_scores(query_text)

        lex_n = self._normalize_scores(lex)
        if emb is None:
            final_scores = lex_n
            used_mode = "lexical_only"
        else:
            emb_n = self._normalize_scores(emb)
            final_scores = 0.65 * emb_n + 0.35 * lex_n
            used_mode = "hybrid"

        # Restrict selection to a manageable candidate set first.
        top_pool_size = min(150, len(self.records))
        candidate_indices = np.argsort(-final_scores)[:top_pool_size].tolist()

        selected = []
        remaining = candidate_indices.copy()

        # For diversity, use scholar embeddings if available. If not, the team becomes top-k by relevance.
        while remaining and len(selected) < team_size:
            if not selected or self.embeddings_norm is None:
                best_idx = max(remaining, key=lambda i: final_scores[i])
                selected.append(best_idx)
                remaining.remove(best_idx)
                continue

            selected_vecs = []
            for idx in selected:
                sid = str(self.records[idx]["scholar_id"])
                emb_idx = self.id_to_embedding_index.get(sid)
                if emb_idx is not None:
                    selected_vecs.append(self.embeddings_norm[emb_idx])

            best_idx = None
            best_mmr = -1e9
            for idx in remaining:
                sid = str(self.records[idx]["scholar_id"])
                emb_idx = self.id_to_embedding_index.get(sid)
                penalty = 0.0
                if emb_idx is not None and selected_vecs:
                    cand_vec = self.embeddings_norm[emb_idx]
                    penalty = max(float(cand_vec @ sv) for sv in selected_vecs)
                mmr = lambda_relevance * float(final_scores[idx]) - (1.0 - lambda_relevance) * penalty
                if mmr > best_mmr:
                    best_mmr = mmr
                    best_idx = idx

            selected.append(best_idx)
            remaining.remove(best_idx)

        results = []
        for rank, idx in enumerate(selected, start=1):
            record = self.records[idx]
            results.append(
                {
                    "rank": rank,
                    "scholar_id": record["scholar_id"],
                    "name": record.get("name", "Unknown"),
                    "primary_affiliation": record.get("primary_affiliation", ""),
                    "emails": record.get("emails", []),
                    "award_count": record.get("award_count", 0),
                    "paper_count": record.get("paper_count", 0),
                    "citation_count": record.get("citation_count", 0),
                    "h_index": record.get("h_index", 0),
                    "roles": record.get("roles", []),
                    "research_keywords": record.get("research_keywords", [])[:8],
                    "top_awards": [a.get("title", "") for a in record.get("awards", [])[:3]],
                    "top_papers": [p.get("title", "") for p in record.get("papers", [])[:3]],
                    "score": round(float(final_scores[idx]), 4),
                    "match_reasons": self.build_match_reasons(record, query_text),
                }
            )

        return {
            "query": query_text,
            "team_size": team_size,
            "results": results,
            "ranking_mode": used_mode,
            "warning": self.embedding_warning,
        }


try:
    recommender = ScholarRecommender()
    STARTUP_ERROR = None
except Exception:
    recommender = None
    STARTUP_ERROR = traceback.format_exc()


@app.route("/")
def index():
    return render_template(
        "index.html",
        user_name=USER_DISPLAY_NAME,
        startup_error=STARTUP_ERROR,
        scholar_count=(len(recommender.records) if recommender else 0),
    )


@app.route("/api/recommend", methods=["POST"])
def api_recommend():
    if recommender is None:
        return jsonify({"error": "Backend data could not be loaded.", "details": STARTUP_ERROR}), 500

    payload = request.get_json(force=True)
    query = (payload.get("query") or "").strip()
    team_size = int(payload.get("team_size", 5))
    team_size = max(1, min(team_size, 10))

    if not query:
        return jsonify({"error": "Please enter a natural-language project query."}), 400

    response = recommender.recommend_team(query, team_size=team_size)
    return jsonify(response)


if __name__ == "__main__":
    # app.run(debug=True) #for running locally at http://127.0.0.1:5000
    # app.run(host="0.0.0.0", port=5000, debug=False) #For hosting in aws
    app.run(host="0.0.0.0", port=5000)
