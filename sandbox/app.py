import streamlit as st
import json
import sys
import os

# allow importing from src/
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

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

st.set_page_config(page_title="Redrob Candidate Ranker", layout="wide")

st.title("🎯 Redrob Candidate Ranker — Sandbox")
st.caption(
    "Rule-based ranking demo on a 50-candidate sample. "
    "The full pipeline (with semantic embeddings) runs offline on the 100K candidate pool — "
    "see src/rank.py in the repo. This sandbox skips the embedding step for speed/portability "
    "and renormalizes the remaining weights so scores stay comparable."
)

# ---------------------------------------------------------------------------
# Load JD profile
# ---------------------------------------------------------------------------
JD_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "jd_profile.json")
with open(JD_PATH) as f:
    jd = json.load(f)

with st.expander("📋 Job Description (structured requirements)"):
    st.json(jd)

# ---------------------------------------------------------------------------
# Load sample candidates
# ---------------------------------------------------------------------------
CAND_PATH = os.path.join(os.path.dirname(__file__), "sample_candidates.json")
with open(CAND_PATH) as f:
    candidates = json.load(f)

st.write(f"Loaded **{len(candidates)}** sample candidates.")

# ---------------------------------------------------------------------------
# Run ranking (no embeddings in sandbox -- pure rule-based score for demo)
# ---------------------------------------------------------------------------
if st.button("🚀 Run Ranking"):
    w = jd["weights"]

    # semantic_similarity is skipped here (no precomputed embeddings loaded in
    # the sandbox for speed/portability). Renormalize the remaining weights so
    # the score scale stays comparable to the full pipeline in src/rank.py.
    used_weight_sum = (
        w["title_seniority_fit"]
        + w["must_have_skills_coverage"]
        + w["experience_fit"]
        + w["location_fit"]
        + w["skill_trust_quality"]
    )

    results = []

    for c in candidates:
        title_fit = title_fit_score(c, jd)
        exp_fit = experience_fit_score(c, jd)
        skill_cov = must_have_skill_coverage(c, jd)
        nice_to_have = nice_to_have_score(c, jd)
        loc_fit = location_fit_score(c, jd)
        traj_penalty, traj_flags = career_trajectory_penalty(c, jd)
        behavioral_mult = behavioral_signal_multiplier(c)
        hp_flags = honeypot_flags(c)

        base_score = (
            w["title_seniority_fit"] * title_fit
            + w["must_have_skills_coverage"] * skill_cov
            + w["experience_fit"] * exp_fit
            + w["location_fit"] * loc_fit
            + w["skill_trust_quality"] * nice_to_have
        ) / used_weight_sum

        score = base_score * traj_penalty

        bw = jd["behavioral_signal_weight"]
        score = score * ((1 - bw) + bw * behavioral_mult)

        if hp_flags:
            score *= max(0.05, 1.0 - 0.35 * len(hp_flags))

        score = max(0.0, min(1.0, score))

        results.append({
            "candidate_id": c["candidate_id"],
            "title": c["profile"]["current_title"],
            "years_exp": c["profile"]["years_of_experience"],
            "location": c["profile"]["location"],
            "score": round(score, 4),
            "title_fit": round(title_fit, 2),
            "skill_coverage": round(skill_cov, 2),
            "experience_fit": round(exp_fit, 2),
            "location_fit": round(loc_fit, 2),
            "behavioral_mult": round(behavioral_mult, 2),
            "trajectory_flags": ", ".join(traj_flags) if traj_flags else "-",
            "honeypot_flags": ", ".join(hp_flags) if hp_flags else "-",
        })

    results.sort(key=lambda r: -r["score"])

    st.subheader("Ranked Results")
    st.dataframe(results, use_container_width=True)

    top = results[0]
    st.success(f"Top candidate: **{top['title']}** ({top['candidate_id']}) — score: {top['score']}")

    flagged = [r for r in results if r["honeypot_flags"] != "-"]
    if flagged:
        st.warning(f"{len(flagged)} candidate(s) in this sample were flagged with data-quality concerns (visible in the honeypot_flags column).")
else:
    st.info("Click 'Run Ranking' to score the sample candidates against the JD.")