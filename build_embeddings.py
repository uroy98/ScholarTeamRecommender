import json
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

BASE_DIR = Path(__file__).resolve().parent
INDEX_PATH = BASE_DIR / "data" / "scholar_index.json"
EMBED_PATH = BASE_DIR / "data" / "scholar_embeddings.npy"
IDS_PATH = BASE_DIR / "data" / "scholar_ids.npy"
MODEL_NAME = "all-MiniLM-L6-v2"


def main():
    if not INDEX_PATH.exists():
        raise FileNotFoundError(
            f"{INDEX_PATH} not found. Run prepare_data.py first."
        )

    with open(INDEX_PATH, "r", encoding="utf-8") as f:
        records = json.load(f)

    texts = [r.get("profile_text", "") for r in records]
    ids = np.array([r["scholar_id"] for r in records], dtype=object)

    print(f"Loading model: {MODEL_NAME}")
    model = SentenceTransformer(MODEL_NAME)

    print("Encoding scholar profiles...")
    embeddings = model.encode(
        texts,
        batch_size=64,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=False,
    ).astype(np.float32)

    np.save(EMBED_PATH, embeddings)
    np.save(IDS_PATH, ids)

    print(f"Saved embeddings to: {EMBED_PATH}")
    print(f"Saved IDs to: {IDS_PATH}")
    print(f"Embeddings shape: {embeddings.shape}")


if __name__ == "__main__":
    main()
