import json
from pathlib import Path

import numpy as np

BASE_DIR = Path(__file__).resolve().parent
JSON_DIR = BASE_DIR / "data" / "nsf_json"
OUTPUT_JSON = BASE_DIR / "data" / "scholar_index.json"
OUTPUT_IDS = BASE_DIR / "data" / "scholar_ids.npy"


def safe_list(value):
    return value if isinstance(value, list) else ([] if value is None else [value])


def compress_awards(awards):
    out = []
    for a in safe_list(awards):
        out.append(
            {
                "nsf_award_id": a.get("nsf_award_id"),
                "title": a.get("title", ""),
                "start_date": a.get("start_date"),
                "exp_date": a.get("exp_date"),
                "program_elements": safe_list(a.get("program_elements")),
                "keywords": safe_list(a.get("keywords")),
            }
        )
    return out


def compress_papers(papers, max_papers=25):
    out = []
    for p in safe_list(papers)[:max_papers]:
        out.append(
            {
                "s2_paper_id": p.get("s2_paper_id"),
                "title": p.get("title", ""),
                "year": p.get("year"),
                "venue": p.get("venue", ""),
                "citation_count": p.get("citation_count", 0),
                "abstract": p.get("abstract") or "",
            }
        )
    return out


def build_profile_text(record):
    pieces = []
    pieces.append(record.get("name", ""))
    pieces.append(record.get("primary_affiliation", ""))
    pieces.extend(record.get("all_affiliations", []))
    pieces.extend(record.get("roles", []))
    pieces.extend(record.get("research_keywords", []))

    for award in record.get("awards", []):
        pieces.append(award.get("title", ""))
        pieces.extend(award.get("keywords", []))
        pieces.extend(award.get("program_elements", []))

    for paper in record.get("papers", []):
        pieces.append(paper.get("title", ""))
        pieces.append(paper.get("venue", ""))
        pieces.append((paper.get("abstract") or "")[:1000])

    return " ".join(str(x) for x in pieces if x)


def load_one_file(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    nsf = raw.get("nsf_profile", {})
    s2 = raw.get("s2_profile", {})

    # scholar_id = str(
    #     nsf.get("nsf_pi_id")
    #     or s2.get("s2_author_id")
    #     or path.stem
    # )

    nsf_pi_id = nsf.get("nsf_pi_id")
    
    if nsf_pi_id:
        scholar_id = f"scholar::nsf_{str(nsf_pi_id).zfill(9)}"
    else:
        scholar_id = str(s2.get("s2_author_id") or path.stem)

    awards = compress_awards(nsf.get("awards", []))
    papers = compress_papers(s2.get("papers", []), max_papers=25)

    record = {
        "scholar_id": scholar_id,
        "nsf_pi_id": nsf.get("nsf_pi_id"),
        "s2_author_id": s2.get("s2_author_id"),
        "name": nsf.get("name") or s2.get("name") or "Unknown",
        "primary_affiliation": nsf.get("primary_affiliation") or "",
        "all_affiliations": safe_list(nsf.get("all_affiliations")),
        "emails": safe_list(nsf.get("emails")),
        "roles": safe_list(nsf.get("roles")),
        "award_count": int(nsf.get("award_count") or len(awards)),
        "paper_count": int(s2.get("paper_count") or len(papers)),
        "citation_count": int(s2.get("citation_count") or 0),
        "h_index": int(s2.get("h_index") or 0),
        "leadership_score": nsf.get("leadership_score"),
        "experience_years": nsf.get("experience_years"),
        "research_keywords": safe_list(nsf.get("keywords")),
        "awards": awards,
        "papers": papers,
        "source_file": path.name,
    }
    record["profile_text"] = build_profile_text(record)
    return record


def main():
    files = sorted(JSON_DIR.glob("*.json"))
    if not files:
        raise FileNotFoundError(
            f"No JSON files found in {JSON_DIR}. Put your 5,099 scholar JSON files there first."
        )

    records = []
    for fp in files:
        try:
            records.append(load_one_file(fp))
        except Exception as e:
            print(f"Skipping {fp.name}: {e}")

    OUTPUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2)

    ids = np.array([r["scholar_id"] for r in records], dtype=object)
    np.save(OUTPUT_IDS, ids)

    print(f"Saved flattened scholar index to: {OUTPUT_JSON}")
    print(f"Saved scholar IDs to: {OUTPUT_IDS}")
    print(f"Total scholars indexed: {len(records)}")


if __name__ == "__main__":
    main()
