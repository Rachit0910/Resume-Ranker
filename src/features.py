"""
features.py
Extracts rule-based features from a candidate record for ranking against the JD.
No network, no GPU. Pure Python / numpy.
"""

import re
from datetime import date, datetime


def _safe_lower(s):
    return (s or "").lower()


def parse_date(s):
    if not s:
        return None
    try:
        return datetime.strptime(s, "%Y-%m-%d").date()
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Title / seniority
# ---------------------------------------------------------------------------

def title_fit_score(candidate, jd):
    title = _safe_lower(candidate["profile"]["current_title"])

    for neg in jd["negative_titles_keywords"]:
        if neg in title:
            return 0.0

    for pos in jd["preferred_titles_keywords"]:
        if pos in title:
            return 1.0

    # fall back: check career history for any matching titles (recent roles)
    hist = candidate.get("career_history", [])
    hist_sorted = sorted(hist, key=lambda r: r.get("start_date") or "", reverse=True)
    for role in hist_sorted[:2]:
        t = _safe_lower(role.get("title"))
        for pos in jd["preferred_titles_keywords"]:
            if pos in t:
                return 0.5

    return 0.15  # ambiguous title, neither clearly positive nor negative


# ---------------------------------------------------------------------------
# Experience fit
# ---------------------------------------------------------------------------

def experience_fit_score(candidate, jd):
    yoe = candidate["profile"].get("years_of_experience", 0)

    if yoe < jd["experience_years_hard_floor"]:
        return 0.0

    lo, hi = jd["experience_years_ideal_min"], jd["experience_years_ideal_max"]
    if lo <= yoe <= hi:
        return 1.0

    if yoe < lo:
        # linear ramp from hard_floor to ideal_min
        floor = jd["experience_years_hard_floor"]
        return max(0.0, (yoe - floor) / (lo - floor)) * 0.8

    # over the ideal max -- gentle decay, overqualified isn't disqualifying
    over = yoe - hi
    return max(0.3, 1.0 - 0.05 * over)


# ---------------------------------------------------------------------------
# Must-have skill coverage (weighted by recency, proficiency, endorsements,
# and assessment scores -- to penalize "keyword stuffing")
# ---------------------------------------------------------------------------

PROFICIENCY_WEIGHT = {"beginner": 0.4, "intermediate": 0.7, "advanced": 1.0, "expert": 1.0}


def _skill_trust(skill, assessment_scores):
    """Returns a 0-1 multiplier reflecting how 'real' a claimed skill looks."""
    name = skill["name"]
    prof = PROFICIENCY_WEIGHT.get(skill.get("proficiency", "beginner"), 0.4)
    months = skill.get("duration_months", 0)
    endorsements = skill.get("endorsements", 0)

    # duration sanity: claiming advanced/expert with very low duration is suspicious
    duration_factor = min(1.0, months / 12.0) if months else 0.2

    endorsement_factor = min(1.0, 0.3 + endorsements / 50.0)

    assess = assessment_scores.get(name)
    if assess is not None:
        assess_factor = assess / 100.0
    else:
        assess_factor = 0.6  # neutral if no assessment exists

    trust = prof * duration_factor * 0.4 + assess_factor * 0.4 + endorsement_factor * 0.2
    return max(0.0, min(1.0, trust))


def must_have_skill_coverage(candidate, jd):
    skills = candidate.get("skills", [])
    skill_names = {_safe_lower(s["name"]): s for s in skills}
    assessment_scores = candidate.get("redrob_signals", {}).get("skill_assessment_scores", {})

    career_text = " ".join(
        _safe_lower(r.get("description", "")) + " " + _safe_lower(r.get("title", ""))
        for r in candidate.get("career_history", [])
    )
    summary_text = _safe_lower(candidate["profile"].get("summary", ""))
    full_text = career_text + " " + summary_text

    group_scores = []
    for group_name, keywords in jd["must_have_skill_groups"].items():
        best = 0.0
        for kw in keywords:
            # check skills list (with trust weighting)
            for sk_name, sk in skill_names.items():
                if kw in sk_name or sk_name in kw:
                    trust = _skill_trust(sk, assessment_scores)
                    best = max(best, trust)
            # check career history / summary text for evidence even if not in skills list
            if kw in full_text:
                best = max(best, 0.55)
        group_scores.append(best)

    return sum(group_scores) / len(group_scores) if group_scores else 0.0


def nice_to_have_score(candidate, jd):
    skills = candidate.get("skills", [])
    skill_names = {_safe_lower(s["name"]) for s in skills}
    career_text = " ".join(_safe_lower(r.get("description", "")) for r in candidate.get("career_history", []))

    hits = 0
    for kw in jd["nice_to_have_skills"]:
        if any(kw in sn or sn in kw for sn in skill_names) or kw in career_text:
            hits += 1
    return min(1.0, hits / max(1, len(jd["nice_to_have_skills"])) * 2.0)  # scaled, capped at 1


# ---------------------------------------------------------------------------
# Career trajectory penalties (research-only, consulting-only, title-chaser,
# tech-lead-no-code, non-NLP specialist)
# ---------------------------------------------------------------------------

def career_trajectory_penalty(candidate, jd):
    """Returns a multiplier in [0,1]. 1.0 = no penalty."""
    hist = candidate.get("career_history", [])
    profile = candidate["profile"]
    penalty = 1.0
    flags = []

    industries = [_safe_lower(r.get("industry", "")) for r in hist] + [_safe_lower(profile.get("current_industry", ""))]
    titles = [_safe_lower(r.get("title", "")) for r in hist] + [_safe_lower(profile.get("current_title", ""))]
    companies = [_safe_lower(r.get("company", "")) for r in hist] + [_safe_lower(profile.get("current_company", ""))]

    # research-only, no production deployment
    research_signals = sum(
        1 for ind in industries if any(r in ind for r in jd["research_only_red_flag_industries"])
    ) + sum(
        1 for t in titles if any(r in t for r in jd["research_only_red_flag_titles"])
    )
    if research_signals >= max(1, len(hist)) and len(hist) > 0:
        penalty *= 0.25
        flags.append("research_only")

    # consulting-only career
    consulting_hits = sum(
        1 for c in companies if any(cc in c for cc in jd["consulting_companies"])
    )
    if consulting_hits >= len(hist) and len(hist) > 0:
        penalty *= 0.35
        flags.append("consulting_only")

    # title-chaser: short stints (<18 months) AND escalating seniority labels
    # (Junior -> Senior -> Staff -> Principal/Lead/Director) across companies.
    SENIORITY_LEVELS = [
        ("principal", 5), ("director", 5), ("staff", 4), ("lead", 4),
        ("senior", 3), ("ii", 2), ("i", 1), ("junior", 1), ("associate", 1),
    ]

    def seniority_level(t):
        for kw, lvl in SENIORITY_LEVELS:
            if kw in t:
                return lvl
        return 2  # default mid-level

    hist_chrono = sorted(hist, key=lambda r: r.get("start_date") or "")
    short_stints = sum(1 for r in hist_chrono if (r.get("duration_months") or 999) < 18)
    levels = [seniority_level(_safe_lower(r.get("title", ""))) for r in hist_chrono]
    escalating = all(b >= a for a, b in zip(levels, levels[1:])) and len(set(levels)) > 1

    if len(hist_chrono) >= 3 and short_stints >= len(hist_chrono) - 1 and escalating:
        penalty *= 0.6
        flags.append("title_chaser")

    # senior/tech-lead with no recent coding (>=18 months)
    current_title = _safe_lower(profile.get("current_title", ""))
    if any(t in current_title for t in jd["tech_lead_no_code_titles"]):
        cur_role = next((r for r in hist if r.get("is_current")), None)
        months_in_role = cur_role.get("duration_months", 0) if cur_role else 0
        if months_in_role >= 18:
            penalty *= 0.4
            flags.append("no_recent_code")

    # non-NLP specialist (CV/speech/robotics) without NLP/IR exposure
    is_non_nlp_specialist = any(
        any(k in t for k in jd["non_nlp_specialist_titles_keywords"]) for t in titles
    )
    if is_non_nlp_specialist:
        full_text = " ".join(_safe_lower(r.get("description", "")) for r in hist) + " " + _safe_lower(profile.get("summary", ""))
        nlp_terms = ["nlp", "retrieval", "search", "ranking", "language model", "text"]
        if not any(t in full_text for t in nlp_terms):
            penalty *= 0.45
            flags.append("non_nlp_specialist")

    return penalty, flags


# ---------------------------------------------------------------------------
# Location fit
# ---------------------------------------------------------------------------

def location_fit_score(candidate, jd):
    location = _safe_lower(candidate["profile"].get("location", ""))
    country = _safe_lower(candidate["profile"].get("country", ""))
    willing_to_relocate = candidate.get("redrob_signals", {}).get("willing_to_relocate", False)

    in_preferred_city = any(city in location for city in jd["preferred_locations_india"])
    in_india = country == jd["preferred_country"]

    if in_preferred_city:
        return 1.0
    if in_india:
        return 0.8 if willing_to_relocate else 0.55
    # outside india
    return 0.35 if willing_to_relocate else 0.1


# ---------------------------------------------------------------------------
# Behavioral signal multiplier (redrob_signals)
# ---------------------------------------------------------------------------

def behavioral_signal_multiplier(candidate, today=None):
    """Returns a multiplier roughly in [0.2, 1.15]."""
    sig = candidate.get("redrob_signals", {})
    if today is None:
        today = date(2026, 6, 15)

    score = 1.0

    # recency of activity
    last_active = parse_date(sig.get("last_active_date"))
    if last_active:
        days_inactive = (today - last_active).days
        if days_inactive <= 14:
            score *= 1.10
        elif days_inactive <= 30:
            score *= 1.0
        elif days_inactive <= 90:
            score *= 0.75
        elif days_inactive <= 180:
            score *= 0.5
        else:
            score *= 0.25

    # open to work
    if sig.get("open_to_work_flag"):
        score *= 1.05
    else:
        score *= 0.9

    # recruiter response rate
    rr = sig.get("recruiter_response_rate")
    if rr is not None:
        score *= (0.6 + 0.5 * rr)  # 0 -> 0.6x, 1 -> 1.1x

    # interview completion rate
    icr = sig.get("interview_completion_rate")
    if icr is not None:
        score *= (0.7 + 0.4 * icr)

    # notice period (shorter is better, JD wants <30 days)
    np_days = sig.get("notice_period_days")
    if np_days is not None:
        if np_days <= 30:
            score *= 1.05
        elif np_days <= 60:
            score *= 0.95
        elif np_days <= 90:
            score *= 0.85
        else:
            score *= 0.7

    # verification trust
    verified = sum([sig.get("verified_email", False), sig.get("verified_phone", False), sig.get("linkedin_connected", False)])
    score *= (0.85 + 0.05 * verified)

    return max(0.05, min(1.3, score))


# ---------------------------------------------------------------------------
# Honeypot / consistency checks
# ---------------------------------------------------------------------------

def honeypot_flags(candidate):
    """Returns a list of strings describing detected inconsistencies."""
    flags = []
    hist = candidate.get("career_history", [])
    profile = candidate["profile"]
    skills = candidate.get("skills", [])
    sig = candidate.get("redrob_signals", {})

    # expert/advanced proficiency with ~0 duration
    for s in skills:
        if s.get("proficiency") in ("expert", "advanced") and s.get("duration_months", 0) <= 1:
            flags.append(f"zero_duration_advanced_skill:{s['name']}")

    # too many "expert" skills (implausible breadth at expert level)
    expert_count = sum(1 for s in skills if s.get("proficiency") == "expert")
    if expert_count >= 8:
        flags.append("too_many_expert_skills")

    # years_of_experience inconsistent with sum of career history durations
    total_months = sum(r.get("duration_months", 0) for r in hist)
    yoe_months = profile.get("years_of_experience", 0) * 12
    if total_months > 0 and yoe_months > 0:
        if total_months > yoe_months * 1.6 or total_months < yoe_months * 0.4:
            flags.append("experience_history_mismatch")

    # education years overlapping implausibly with full-time career start
    for edu in candidate.get("education", []):
        for role in hist:
            sd = parse_date(role.get("start_date"))
            if sd and edu.get("end_year") and sd.year < edu["end_year"] - 1:
                flags.append("career_before_education_end")
                break

    # offer_acceptance_rate / interview_completion_rate out-of-range sanity (schema allows -1 sentinel)
    oar = sig.get("offer_acceptance_rate")
    if oar is not None and oar > 1.0:
        flags.append("offer_acceptance_rate_out_of_range")

    # github_activity_score absurdly high with no related skills
    gh = sig.get("github_activity_score", -1)
    if gh and gh > 90:
        skill_text = " ".join(_safe_lower(s["name"]) for s in skills)
        if not any(k in skill_text for k in ["python", "engineer", "ml", "software", "code", "ai", "data"]):
            flags.append("github_score_unsupported")

    return flags
