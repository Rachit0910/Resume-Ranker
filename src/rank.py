"""
rank.py
The ranking step. Must satisfy: <=5 min wall clock, <=16GB RAM, CPU only, no network.

Reads:
  - candidates.jsonl(.gz)
  - data/jd_profile.json
  - data/embeddings.npy, data/jd_embedding.npy, data/candidate_ids.json  (precomputed)

Writes:
  - output/submission.csv  (candidate_id, rank, score, reasoning)

Usage:
  python src/rank.py --candidates data/candidates.jsonl --out output/submission.csv
"""

import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from features import (
    title_fit_score,
    experience_fit_score,
    must_have_skill_coverage,
    nice_to_have_score,
    career_trajectory_penalty,
    location_fit_score,
    behavioral_signal_multiplier,
    honeypot_flags,
)
from precompute import load_candidates


def load_jsonl_or_gz(path):
    return load_candidates(path)


def cosine_sim_batch(cand_emb, jd_emb):
    # embeddings assumed pre-normalized; if not, normalize here
    norms = np.linalg.norm(cand_emb, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    cand_norm = cand_emb / norms
    jd_norm = jd_emb / (np.linalg.norm(jd_emb) + 1e-9)
    return cand_norm @ jd_norm


def build_reasoning(candidate, scores, flags):
    profile = candidate["profile"]
    title = profile.get("current_title", "Unknown role")
    yoe = profile.get("years_of_experience", 0)
    location = profile.get("location", "")
    sig = candidate.get("redrob_signals", {})

    bits = []
    bits.append(f"{title} with {yoe:.1f} yrs experience")

    if scores["must_have_skills_coverage"] >= 0.6:
        bits.append("strong coverage of retrieval/embeddings/ranking skill areas")
    elif scores["must_have_skills_coverage"] >= 0.35:
        bits.append("partial coverage of core ML retrieval/ranking skills")
    else:
        bits.append("limited evidence of core embeddings/retrieval/ranking experience")

    if scores["career_trajectory_penalty_flags"]:
        flag_str = ", ".join(scores["career_trajectory_penalty_flags"])
        bits.append(f"concern: {flag_str.replace('_', ' ')}")

    rr = sig.get("recruiter_response_rate")
    last_active = sig.get("last_active_date")
    if rr is not None:
        bits.append(f"recruiter response rate {rr:.2f}, last active {last_active}")

    if location:
        bits.append(f"based in {location}")

    np_days = sig.get("notice_period_days")
    if np_days is not None:
        bits.append(f"notice period {np_days}d")

    if flags:
        bits.append(f"data-quality flag: {flags[0]}")

    return "; ".join(bits) + "."


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidates", required=True)
    ap.add_argument("--jd-profile", default="data/jd_profile.json")
    ap.add_argument("--embeddings", default="data/embeddings.npy")
    ap.add_argument("--jd-embedding", default="data/jd_embedding.npy")
    ap.add_argument("--candidate-ids", default="data/candidate_ids.json")
    ap.add_argument("--out", default="output/submission.csv")
    ap.add_argument("--top-n", type=int, default=100)
    args = ap.parse_args()

    t0 = time.time()

    with open(args.jd_profile) as f:
        jd = json.load(f)

    print("[rank] loading candidates...")
    candidates = load_jsonl_or_gz(args.candidates)
    print(f"[rank] loaded {len(candidates)} candidates in {time.time()-t0:.1f}s")

    # embeddings (precomputed)
    use_embeddings = (
        os.path.exists(args.embeddings)
        and os.path.exists(args.jd_embedding)
        and os.path.exists(args.candidate_ids)
    )
    if use_embeddings:
        cand_emb = np.load(args.embeddings)
        jd_emb = np.load(args.jd_embedding)
        with open(args.candidate_ids) as f:
            emb_ids = json.load(f)
        emb_index = {cid: i for i, cid in enumerate(emb_ids)}
        sims = cosine_sim_batch(cand_emb, jd_emb)
        sim_lookup = {cid: float((sims[i] + 1) / 2) for cid, i in emb_index.items()}  # rescale [-1,1] -> [0,1]
        print(f"[rank] loaded precomputed embeddings ({cand_emb.shape})")
    else:
        sim_lookup = {}
        print("[rank] WARNING: no precomputed embeddings found, semantic score = 0.5 for all")

    w = jd["weights"]
    results = []

    for c in candidates:
        cid = c["candidate_id"]

        semantic = sim_lookup.get(cid, 0.5)
        title_fit = title_fit_score(c, jd)
        exp_fit = experience_fit_score(c, jd)
        skill_cov = must_have_skill_coverage(c, jd)
        nice_to_have = nice_to_have_score(c, jd)
        loc_fit = location_fit_score(c, jd)
        traj_penalty, traj_flags = career_trajectory_penalty(c, jd)
        behavioral_mult = behavioral_signal_multiplier(c)
        hp_flags = honeypot_flags(c)

        base_score = (
            w["semantic_similarity"] * semantic
            + w["title_seniority_fit"] * title_fit
            + w["must_have_skills_coverage"] * skill_cov
            + w["experience_fit"] * exp_fit
            + w["location_fit"] * loc_fit
            + w["skill_trust_quality"] * nice_to_have
        )

        # career trajectory acts as a multiplicative penalty
        score = base_score * traj_penalty

        # behavioral signal multiplier, dampened so it can't dominate skill fit
        behavioral_weight = jd["behavioral_signal_weight"]
        score = score * ((1 - behavioral_weight) + behavioral_weight * behavioral_mult)

        # honeypot penalty
        if hp_flags:
            score *= max(0.05, 1.0 - 0.35 * len(hp_flags))

        score = max(0.0, min(1.0, score))

        results.append({
            "candidate_id": cid,
            "score": score,
            "candidate": c,
            "scores": {
                "semantic_similarity": semantic,
                "title_fit": title_fit,
                "experience_fit": exp_fit,
                "must_have_skills_coverage": skill_cov,
                "nice_to_have": nice_to_have,
                "location_fit": loc_fit,
                "career_trajectory_penalty": traj_penalty,
                "career_trajectory_penalty_flags": traj_flags,
                "behavioral_mult": behavioral_mult,
            },
            "honeypot_flags": hp_flags,
        })

    print(f"[rank] scored all candidates in {time.time()-t0:.1f}s")

    # sort: score desc, then candidate_id asc for ties
    results.sort(key=lambda r: (-r["score"], r["candidate_id"]))

    top = results[:args.top_n]

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    import csv
    with open(args.out, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["candidate_id", "rank", "score", "reasoning"])
        for rank, r in enumerate(top, start=1):
            reasoning = build_reasoning(r["candidate"], r["scores"], r["honeypot_flags"])
            writer.writerow([r["candidate_id"], rank, f"{r['score']:.6f}", reasoning])

    elapsed = time.time() - t0
    print(f"[rank] wrote {args.out} ({len(top)} rows) in {elapsed:.1f}s total")

    if elapsed > 300:
        print("[rank] WARNING: exceeded 5-minute budget!", file=sys.stderr)


if __name__ == "__main__":
    main()
