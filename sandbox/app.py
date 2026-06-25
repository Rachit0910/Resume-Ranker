import streamlit as st
import json
import sys
import os

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

# ---- page config ----
st.set_page_config(
    page_title="Redrob Candidate Ranker",
    page_icon="🎯",
    layout="wide",
)

# ---- custom CSS ----
st.markdown("""
<style>
    /* main background */
    .stApp {
        background: linear-gradient(135deg, #0f1b3d 0%, #1a2f6e 50%, #0f1b3d 100%);
        color: #ffffff;
    }

    /* sidebar bhi dark */
    section[data-testid="stSidebar"] {
        background-color: #0a1228;
    }

    /* header */
    .main-header {
        background: linear-gradient(90deg, #1e3a8a, #3b82f6);
        padding: 2rem 2.5rem;
        border-radius: 16px;
        margin-bottom: 2rem;
        box-shadow: 0 8px 32px rgba(59,130,246,0.3);
    }
    .main-header h1 {
        color: white;
        font-size: 2.4rem;
        font-weight: 800;
        margin: 0;
        letter-spacing: -0.5px;
    }
    .main-header p {
        color: #bfdbfe;
        margin: 0.5rem 0 0 0;
        font-size: 1rem;
    }

    /* metric cards */
    .metric-card {
        background: rgba(255,255,255,0.07);
        border: 1px solid rgba(255,255,255,0.12);
        border-radius: 12px;
        padding: 1.2rem 1.5rem;
        text-align: center;
        backdrop-filter: blur(10px);
    }
    .metric-card .value {
        font-size: 2rem;
        font-weight: 800;
        color: #60a5fa;
    }
    .metric-card .label {
        font-size: 0.8rem;
        color: #94a3b8;
        margin-top: 0.2rem;
        text-transform: uppercase;
        letter-spacing: 1px;
    }

    /* rank cards */
    .rank-card {
        background: rgba(255,255,255,0.06);
        border: 1px solid rgba(255,255,255,0.1);
        border-radius: 14px;
        padding: 1.2rem 1.5rem;
        margin-bottom: 0.8rem;
        transition: all 0.2s;
        backdrop-filter: blur(8px);
    }
    .rank-card:hover {
        background: rgba(59,130,246,0.15);
        border-color: rgba(59,130,246,0.4);
    }
    .rank-badge {
        background: linear-gradient(135deg, #1d4ed8, #3b82f6);
        color: white;
        border-radius: 50%;
        width: 38px;
        height: 38px;
        display: inline-flex;
        align-items: center;
        justify-content: center;
        font-weight: 800;
        font-size: 1rem;
        margin-right: 1rem;
    }
    .rank-badge.gold { background: linear-gradient(135deg, #d97706, #fbbf24); }
    .rank-badge.silver { background: linear-gradient(135deg, #6b7280, #d1d5db); }
    .rank-badge.bronze { background: linear-gradient(135deg, #92400e, #d97706); }

    .candidate-title {
        font-size: 1.05rem;
        font-weight: 700;
        color: #f0f9ff;
    }
    .candidate-meta {
        font-size: 0.82rem;
        color: #94a3b8;
        margin-top: 0.15rem;
    }
    .score-bar-bg {
        background: rgba(255,255,255,0.1);
        border-radius: 99px;
        height: 6px;
        margin-top: 0.6rem;
    }
    .score-bar-fill {
        background: linear-gradient(90deg, #3b82f6, #60a5fa);
        border-radius: 99px;
        height: 6px;
    }
    .flag-badge {
        background: rgba(239,68,68,0.2);
        color: #fca5a5;
        border: 1px solid rgba(239,68,68,0.3);
        border-radius: 6px;
        padding: 2px 8px;
        font-size: 0.72rem;
        margin-left: 0.5rem;
    }
    .concern-badge {
        background: rgba(251,191,36,0.15);
        color: #fde68a;
        border: 1px solid rgba(251,191,36,0.3);
        border-radius: 6px;
        padding: 2px 8px;
        font-size: 0.72rem;
        margin-left: 0.5rem;
    }

    /* score pills */
    .pill {
        display: inline-block;
        background: rgba(59,130,246,0.15);
        border: 1px solid rgba(59,130,246,0.3);
        color: #93c5fd;
        border-radius: 99px;
        padding: 2px 10px;
        font-size: 0.75rem;
        margin: 2px;
    }

    /* button */
    .stButton > button {
        background: linear-gradient(135deg, #1d4ed8, #3b82f6) !important;
        color: white !important;
        border: none !important;
        border-radius: 10px !important;
        padding: 0.65rem 2rem !important;
        font-weight: 700 !important;
        font-size: 1rem !important;
        box-shadow: 0 4px 15px rgba(59,130,246,0.4) !important;
        transition: all 0.2s !important;
        width: 100%;
    }
    .stButton > button:hover {
        box-shadow: 0 6px 20px rgba(59,130,246,0.6) !important;
        transform: translateY(-1px) !important;
    }

    /* expander */
    .streamlit-expanderHeader {
        background: rgba(255,255,255,0.06) !important;
        border-radius: 10px !important;
        color: #bfdbfe !important;
    }

    /* dataframe */
    .stDataFrame {
        border-radius: 12px;
        overflow: hidden;
    }

    /* hide streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)


# ---- header ----
st.markdown("""
<div class="main-header">
    <h1> Candidate Ranker</h1>
    <p>Intelligent candidate discovery — ranks the way a great recruiter actually thinks, not by keywords.</p>
</div>
""", unsafe_allow_html=True)


# ---- load data ----
JD_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "jd_profile.json")
with open(JD_PATH) as f:
    jd = json.load(f)

CAND_PATH = os.path.join(os.path.dirname(__file__), "sample_candidates.json")
with open(CAND_PATH) as f:
    candidates = json.load(f)


# ---- top info row ----
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.markdown("""
    <div class="metric-card">
        <div class="value">50</div>
        <div class="label">Sample Candidates</div>
    </div>""", unsafe_allow_html=True)
with col2:
    st.markdown("""
    <div class="metric-card">
        <div class="value">6</div>
        <div class="label">Scoring Components</div>
    </div>""", unsafe_allow_html=True)
with col3:
    st.markdown("""
    <div class="metric-card">
        <div class="value">100K</div>
        <div class="label">Full Pool Size</div>
    </div>""", unsafe_allow_html=True)
with col4:
    st.markdown("""
    <div class="metric-card">
        <div class="value">&lt;90s</div>
        <div class="label">Full Run Time</div>
    </div>""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ---- JD expander ----
with st.expander("📋 View Job Description (structured requirements)"):
    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown(f"**Role:** {jd['role_title']}")
        st.markdown(f"**Company:** {jd['company']}")
        st.markdown(f"**Experience:** {jd['experience_years_ideal_min']}–{jd['experience_years_ideal_max']} years (min {jd['experience_years_hard_floor']} yrs)")
        st.markdown("**Must-have skill groups:**")
        for group, keywords in jd["must_have_skill_groups"].items():
            st.markdown(f"- `{group}`: {', '.join(keywords[:3])}...")
    with col_b:
        st.markdown("**Preferred locations:**")
        st.markdown(", ".join(jd["preferred_locations_india"]))
        st.markdown("**Preferred titles:**")
        st.markdown(", ".join(jd["preferred_titles_keywords"][:6]) + "...")
        st.markdown("**Disqualifiers:**")
        st.markdown(", ".join(jd["consulting_companies"]) + " (consulting-only), research-only, non-NLP specialist")

st.markdown("<br>", unsafe_allow_html=True)

# ---- run button ----
run = st.button("🚀 Run Ranking")

if run:
    w = jd["weights"]

    # sandbox mein embeddings nahi hain, baaki weights renormalize karo
    used_weight_sum = (
        w["title_seniority_fit"]
        + w["must_have_skills_coverage"]
        + w["experience_fit"]
        + w["location_fit"]
        + w["skill_trust_quality"]
    )

    results = []
    for c in candidates:
        title_fit       = title_fit_score(c, jd)
        exp_fit         = experience_fit_score(c, jd)
        skill_cov       = must_have_skill_coverage(c, jd)
        nice_to_have    = nice_to_have_score(c, jd)
        loc_fit         = location_fit_score(c, jd)
        traj_penalty, traj_flags = career_trajectory_penalty(c, jd)
        behavioral_mult = behavioral_signal_multiplier(c)
        hp_flags        = honeypot_flags(c)

        base_score = (
            w["title_seniority_fit"]         * title_fit
            + w["must_have_skills_coverage"] * skill_cov
            + w["experience_fit"]            * exp_fit
            + w["location_fit"]              * loc_fit
            + w["skill_trust_quality"]       * nice_to_have
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
            "score": score,
            "title_fit": round(title_fit, 2),
            "skill_coverage": round(skill_cov, 2),
            "experience_fit": round(exp_fit, 2),
            "location_fit": round(loc_fit, 2),
            "behavioral_mult": round(behavioral_mult, 2),
            "trajectory_flags": traj_flags,
            "honeypot_flags": hp_flags,
        })

    results.sort(key=lambda r: -r["score"])

    # ---- summary metrics ----
    flagged_count = sum(1 for r in results if r["honeypot_flags"])
    concern_count = sum(1 for r in results if r["trajectory_flags"])

    m1, m2, m3 = st.columns(3)
    with m1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="value" style="color:#34d399">{results[0]['score']:.4f}</div>
            <div class="label">Top Score</div>
        </div>""", unsafe_allow_html=True)
    with m2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="value" style="color:#fbbf24">{concern_count}</div>
            <div class="label">Trajectory Concerns</div>
        </div>""", unsafe_allow_html=True)
    with m3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="value" style="color:#f87171">{flagged_count}</div>
            <div class="label">Honeypot Flags</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ---- tabs ----
    tab1, tab2 = st.tabs(["🏆 Ranked Results", "📊 Full Breakdown Table"])

    with tab1:
        st.markdown("### Top Candidates")
        for i, r in enumerate(results[:15]):
            rank = i + 1
            score_pct = int(r["score"] * 100)

            if rank == 1:
                badge_class = "gold"
            elif rank == 2:
                badge_class = "silver"
            elif rank == 3:
                badge_class = "bronze"
            else:
                badge_class = ""

            hp_html = ""
            if r["honeypot_flags"]:
                hp_html = '<span class="flag-badge">⚠ honeypot flag</span>'

            concern_html = ""
            if r["trajectory_flags"]:
                concern_html = f'<span class="concern-badge">⚡ {", ".join(r["trajectory_flags"]).replace("_"," ")}</span>'

            pills = f"""
            <span class="pill">title {r['title_fit']}</span>
            <span class="pill">skills {r['skill_coverage']}</span>
            <span class="pill">exp {r['experience_fit']}</span>
            <span class="pill">loc {r['location_fit']}</span>
            <span class="pill">behav {r['behavioral_mult']}</span>
            """

        for i, r in enumerate(results[:15]):
            rank = i + 1
            score_pct = int(r["score"] * 100)

            if rank == 1:
              badge_class = "gold"
            elif rank == 2:
              badge_class = "silver"
            elif rank == 3:
              badge_class = "bronze"
            else:
              badge_class = ""

            hp_html = '<span class="flag-badge">⚠ honeypot flag</span>' if r["honeypot_flags"] else ""
            concern_html = f'<span class="concern-badge">⚡ {", ".join(r["trajectory_flags"]).replace("_"," ")}</span>' if r["trajectory_flags"] else ""

            pills = "".join([
            f'<span class="pill">title {r["title_fit"]}</span>',
            f'<span class="pill">skills {r["skill_coverage"]}</span>',
            f'<span class="pill">exp {r["experience_fit"]}</span>',
            f'<span class="pill">loc {r["location_fit"]}</span>',
            f'<span class="pill">behav {r["behavioral_mult"]}</span>',
            ])

            html = (
            f'<div class="rank-card">'
            f'<div style="display:flex;align-items:center;">'
            f'<span class="rank-badge {badge_class}">#{rank}</span>'
            f'<div style="flex:1">'
            f'<div class="candidate-title">{r["title"]}{hp_html}{concern_html}</div>'
            f'<div class="candidate-meta">{r["candidate_id"]} &middot; {r["years_exp"]} yrs &middot; {r["location"]}</div>'
            f'<div style="margin-top:0.5rem">{pills}</div>'
            f'</div>'
            f'<div style="text-align:right;min-width:70px">'
            f'<div style="font-size:1.3rem;font-weight:800;color:#60a5fa">{r["score"]:.4f}</div>'
            f'<div style="font-size:0.75rem;color:#64748b">score</div>'
            f'</div>'
            f'</div>'
            f'<div class="score-bar-bg">'
            f'<div class="score-bar-fill" style="width:{score_pct}%"></div>'
            f'</div>'
            f'</div>'
            )
            st.markdown(html, unsafe_allow_html=True)

    with tab2:
        table_data = [{
            "rank": i+1,
            "candidate_id": r["candidate_id"],
            "title": r["title"],
            "score": round(r["score"], 4),
            "years_exp": r["years_exp"],
            "location": r["location"],
            "title_fit": r["title_fit"],
            "skill_coverage": r["skill_coverage"],
            "experience_fit": r["experience_fit"],
            "location_fit": r["location_fit"],
            "behavioral_mult": r["behavioral_mult"],
            "trajectory_flags": ", ".join(r["trajectory_flags"]) if r["trajectory_flags"] else "-",
            "honeypot_flags": ", ".join(r["honeypot_flags"]) if r["honeypot_flags"] else "-",
        } for i, r in enumerate(results)]
        st.dataframe(table_data, use_container_width=True)

else:
    # ---- placeholder state ----
    st.markdown("""
    <div style="
        background: rgba(255,255,255,0.04);
        border: 1px dashed rgba(255,255,255,0.15);
        border-radius: 16px;
        padding: 3rem;
        text-align: center;
        color: #64748b;
    ">
        <div style="font-size:3rem">🎯</div>
        <div style="font-size:1.1rem; margin-top:1rem; color:#94a3b8">
            Click <b style="color:#60a5fa">Run Ranking</b> to score 50 sample candidates against the JD
        </div>
        <div style="font-size:0.85rem; margin-top:0.5rem">
            Full pipeline runs on 100K candidates in under 90 seconds
        </div>
    </div>
    """, unsafe_allow_html=True)