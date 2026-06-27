# Resume Ranker

An intelligent candidate ranking system that ranks job applicants against a job description the way a great recruiter would — not by keyword matching, but by understanding actual fit.

## What it does

Given a job description and a pool of candidate profiles, this system outputs a ranked shortlist of the best-fit candidates with a per-candidate reasoning string explaining why they ranked where they did.

## How it works

Two-phase pipeline:

**Phase 1 — Precompute (run once, offline)**
Converts each candidate profile into a text representation (title + summary + career history + skills) and embeds it using `sentence-transformers`. Also embeds the job description requirements. Saves everything to disk.

**Phase 2 — Rank (fast, CPU-only, no network)**
Loads precomputed embeddings and scores every candidate on 6 components:
- Semantic similarity (embedding cosine similarity vs JD)
- Title & seniority fit
- Must-have skill coverage (trust-weighted by proficiency, duration, endorsements, assessment scores)
- Experience fit
- Location fit
- Nice-to-have skills

Applies multiplicative penalties for problematic career patterns (research-only, consulting-only, no recent hands-on coding, non-NLP specialists) and a behavioral signal multiplier based on platform activity signals (response rate, activity recency, notice period). Also detects and penalizes inconsistent/suspicious profiles.

Outputs a ranked CSV with `candidate_id, rank, score, reasoning`.

## Project structure
src/

features.py           # rule-based feature extraction

precompute.py         # offline embedding generation

rank.py               # main ranking script

validate_submission.py

data/

jd_profile.json       # structured JD requirements

sandbox/

app.py                # Streamlit demo app

sample_candidates.json

output/

submission.csv
## Setup

```bash
# create environment
conda create -n resume-ranker python=3.11 -y
conda activate resume-ranker

# install dependencies
pip install -r requirements.txt
```

## Usage

```bash
# Step 1 — precompute embeddings (one-time, needs network for model download)
python src/precompute.py --candidates data/candidates.jsonl --out-dir data/

# Step 2 — run ranking (CPU only, no network required)
python src/rank.py --candidates data/candidates.jsonl --out output/submission.csv

# Step 3 — validate output format
python src/validate_submission.py output/submission.csv
```

## Tech stack

| Tool | Version | Purpose |
|---|---|---|
| sentence-transformers | 5.6.0 | Semantic embeddings (all-MiniLM-L6-v2) |
| numpy | 2.1.3 | Fast batch cosine similarity |
| scikit-learn | 1.6.1 | TF-IDF + SVD fallback embedder |
| pandas | 2.2.3 | Data handling |
| streamlit | 1.45.1 | Demo sandbox UI |
| torch | 2.12.1 | Backend for sentence-transformers |

## Live Demo

[Streamlit Sandbox](https://resume-ranker-trm3isrjry53ybmvivwazr.streamlit.app) — run the ranker on a 50-candidate sample in the browser.

## Performance

- 100,000 candidates ranked in ~78 seconds on CPU
- 8 GB RAM usage
- No GPU required
- No network calls during ranking
