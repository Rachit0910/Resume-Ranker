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


# title match karta hai JD ke preferred/negative keywords se
def title_fit_score(candidate, jd):
    title = _safe_lower(candidate["profile"]["current_title"])

    for neg in jd["negative_titles_keywords"]:
        if neg in title:
            return 0.0

    for pos in jd["preferred_titles_keywords"]:
        if pos in title:
            return 1.0

    # current title match nahi hua toh recent history check karo
    hist = candidate.get("career_history", [])
    hist_sorted = sorted(hist, key=lambda r: r.get("start_date") or "", reverse=True)
    for role in hist_sorted[:2]:
        t = _safe_lower(role.get("title"))
        for pos in jd["preferred_titles_keywords"]:
            if pos in t:
                return 0.5

    return 0.15


# experience years vs JD ki ideal range
def experience_fit_score(candidate, jd):
    yoe = candidate["profile"].get("years_of_experience", 0)

    if yoe < jd["experience_years_hard_floor"]:
        return 0.0

    lo, hi = jd["experience_years_ideal_min"], jd["experience_years_ideal_max"]
    if lo <= yoe <= hi:
        return 1.0

    if yoe < lo:
        floor = jd["experience_years_hard_floor"]
        return max(0.0, (yoe - floor) / (lo - floor)) * 0.8

    # overqualified hai, thoda decay
    over = yoe - hi
    return max(0.3, 1.0 - 0.05 * over)


PROFICIENCY_WEIGHT = {"beginner": 0.4, "intermediate": 0.7, "advanced": 1.0, "expert": 1.0}


def _skill_trust(skill, assessment_scores):
    # kitna genuine lagta hai yeh skill claim
    name = skill["name"]
    prof = PROFICIENCY_WEIGHT.get(skill.get("proficiency", "beginner"), 0.4)
    months = skill.get("duration_months", 0)
    endorsements = skill.get("endorsements", 0)

    duration_factor = min(1.0, months / 12.0) if months else 0.2
    endorsement_factor = min(1.0, 0.3 + endorsements / 50.0)

    assess = assessment_scores.get(name)
    assess_factor = assess / 100.0 if assess is not None else 0.6

    trust = prof * duration_factor * 0.4 + assess_factor * 0.4 + endorsement_factor * 0.2
    return max(0.0, min(1.0, trust))


# 4 must-have skill groups check karta hai, keyword stuffers ko pakadta hai
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
            for sk_name, sk in skill_names.items():
                if kw in sk_name or sk_name in kw:
                    trust = _skill_trust(sk, assessment_scores)
                    best = max(best, trust)
            # skills list mein nahi hai but career history mein mention hai
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
    return min(1.0, hits / max(1, len(jd["nice_to_have_skills"])) * 2.0)


# research-only, consulting-only, title-chaser wagera patterns ko penalize karta hai
def career_trajectory_penalty(candidate, jd):
    hist = candidate.get("career_history", [])
    profile = candidate["profile"]
    penalty = 1.0
    flags = []

    industries = [_safe_lower(r.get("industry", "")) for r in hist] + [_safe_lower(profile.get("current_industry", ""))]
    titles = [_safe_lower(r.get("title", "")) for r in hist] + [_safe_lower(profile.get("current_title", ""))]
    companies = [_safe_lower(r.get("company", "")) for r in hist] + [_safe_lower(profile.get("current_company", ""))]

    # poora career research mein, koi production deployment nahi
    research_signals = sum(
        1 for ind in industries if any(r in ind for r in jd["research_only_red_flag_industries"])
    ) + sum(
        1 for t in titles if any(r in t for r in jd["research_only_red_flag_titles"])
    )
    if research_signals >= max(1, len(hist)) and len(hist) > 0:
        penalty *= 0.25
        flags.append("research_only")

    # TCS/Infosys type companies mein poora career
    consulting_hits = sum(
        1 for c in companies if any(cc in c for cc in jd["consulting_companies"])
    )
    if consulting_hits >= len(hist) and len(hist) > 0:
        penalty *= 0.35
        flags.append("consulting_only")

    # short stints ke saath seniority escalation — asli growth nahi
    SENIORITY_LEVELS = [
        ("principal", 5), ("director", 5), ("staff", 4), ("lead", 4),
        ("senior", 3), ("ii", 2), ("i", 1), ("junior", 1), ("associate", 1),
    ]

    def seniority_level(t):
        for kw, lvl in SENIORITY_LEVELS:
            if kw in t:
                return lvl
        return 2

    hist_chrono = sorted(hist, key=lambda r: r.get("start_date") or "")
    short_stints = sum(1 for r in hist_chrono if (r.get("duration_months") or 999) < 18)
    levels = [seniority_level(_safe_lower(r.get("title", ""))) for r in hist_chrono]
    escalating = all(b >= a for a, b in zip(levels, levels[1:])) and len(set(levels)) > 1

    if len(hist_chrono) >= 3 and short_stints >= len(hist_chrono) - 1 and escalating:
        penalty *= 0.6
        flags.append("title_chaser")

    # 18+ months manager/lead role mein — probably code nahi likha
    current_title = _safe_lower(profile.get("current_title", ""))
    if any(t in current_title for t in jd["tech_lead_no_code_titles"]):
        cur_role = next((r for r in hist if r.get("is_current")), None)
        months_in_role = cur_role.get("duration_months", 0) if cur_role else 0
        if months_in_role >= 18:
            penalty *= 0.4
            flags.append("no_recent_code")

    # CV/robotics/speech wale jinka NLP se koi lena dena nahi
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
    return 0.35 if willing_to_relocate else 0.1


# redrob_signals se candidate ki actual availability/responsiveness judge karta hai
def behavioral_signal_multiplier(candidate, today=None):
    sig = candidate.get("redrob_signals", {})
    if today is None:
        today = date(2026, 6, 15)

    score = 1.0

    # kitne din se inactive hai
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

    if sig.get("open_to_work_flag"):
        score *= 1.05
    else:
        score *= 0.9

    rr = sig.get("recruiter_response_rate")
    if rr is not None:
        score *= (0.6 + 0.5 * rr)

    icr = sig.get("interview_completion_rate")
    if icr is not None:
        score *= (0.7 + 0.4 * icr)

    # notice period — 30 din se kam ideal hai
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

    verified = sum([sig.get("verified_email", False), sig.get("verified_phone", False), sig.get("linkedin_connected", False)])
    score *= (0.85 + 0.05 * verified)

    return max(0.05, min(1.3, score))


# impossible profiles pakadta hai — honeypot detection
def honeypot_flags(candidate):
    flags = []
    hist = candidate.get("career_history", [])
    profile = candidate["profile"]
    skills = candidate.get("skills", [])
    sig = candidate.get("redrob_signals", {})

    # expert/advanced skill but 0 months use kiya — sus hai
    for s in skills:
        if s.get("proficiency") in ("expert", "advanced") and s.get("duration_months", 0) <= 1:
            flags.append(f"zero_duration_advanced_skill:{s['name']}")

    # itne saare expert skills ek insaan mein possible nahi
    expert_count = sum(1 for s in skills if s.get("proficiency") == "expert")
    if expert_count >= 8:
        flags.append("too_many_expert_skills")

    # stated experience aur actual history match nahi karti
    total_months = sum(r.get("duration_months", 0) for r in hist)
    yoe_months = profile.get("years_of_experience", 0) * 12
    if total_months > 0 and yoe_months > 0:
        if total_months > yoe_months * 1.6 or total_months < yoe_months * 0.4:
            flags.append("experience_history_mismatch")

    # career education khatam hone se pehle shuru ho gayi
    for edu in candidate.get("education", []):
        for role in hist:
            sd = parse_date(role.get("start_date"))
            if sd and edu.get("end_year") and sd.year < edu["end_year"] - 1:
                flags.append("career_before_education_end")
                break

    oar = sig.get("offer_acceptance_rate")
    if oar is not None and oar > 1.0:
        flags.append("offer_acceptance_rate_out_of_range")

    # high github score but koi technical skill nahi — fishy
    gh = sig.get("github_activity_score", -1)
    if gh and gh > 90:
        skill_text = " ".join(_safe_lower(s["name"]) for s in skills)
        if not any(k in skill_text for k in ["python", "engineer", "ml", "software", "code", "ai", "data"]):
            flags.append("github_score_unsupported")

    return flags