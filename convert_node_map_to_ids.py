import json
import numpy as np
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
NODE_MAP_PATH = DATA_DIR / "node_id_map.json"
OUT_PATH = DATA_DIR / "scholar_ids.npy"

with open(NODE_MAP_PATH, "r", encoding="utf-8") as f:
    node_map = json.load(f)

if "scholar_ids" not in node_map:
    raise ValueError("node_id_map.json does not contain a 'scholar_ids' field.")

scholar_ids = node_map["scholar_ids"]

if not isinstance(scholar_ids, list):
    raise ValueError("'scholar_ids' must be a list.")

scholar_ids = np.array([str(x) for x in scholar_ids], dtype=object)
np.save(OUT_PATH, scholar_ids)

print(f"Saved: {OUT_PATH}")
print(f"Total scholar IDs: {len(scholar_ids)}")
print("Example IDs:", scholar_ids[:5].tolist())