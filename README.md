# Redrob Hackathon — Intelligent Candidate Ranker

Ranks 100,000 candidate profiles against a Senior AI Engineer JD for Redrob AI.

## Approach

Two-phase pipeline:

**Phase 1 — Precompute (offline, one-time)**
- Converts each candidate profile into a text blob (title + summary + career history + skills)
- Embeds using `sentence-transformers` (all-MiniLM-L6-v2)
- Saves embeddings to disk

**Phase 2 — Rank (CPU only, no network, under 5 minutes)**
- Loads precomputed embeddings
- Scores each candidate on 6 components:
  - Semantic similarity (embedding cosine sim vs JD)
  - Title/seniority fit
  - Must-have skill coverage (with keyword-stuffer detection)
  - Experience fit
  - Location fit
  - Nice-to-have skills
- Applies career trajectory penalties (research-only, consulting-only, no recent code)
- Applies behavioral signal multiplier (activity, response rate, notice period)
- Applies honeypot penalty (inconsistent/impossible profiles)
- Outputs top 100 candidates as ranked CSV

## Reproduce

```bash
# Step 1 - Install dependencies
pip install -r requirements.txt

# Step 2 - Precompute embeddings (one-time, needs network)
python src/precompute.py --candidates data/candidates.jsonl --out-dir data/

# Step 3 - Run ranking (CPU only, no network, under 5 min)
python src/rank.py --candidates data/candidates.jsonl --out output/submission.csv

# Step 4 - Validate
python src/validate_submission.py output/submission.csv
```

## Project Structure