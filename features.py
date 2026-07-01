"""
features.py — All feature extraction and scoring functions.
Approach A: Structured Feature Ranker for Redrob Hackathon.

Each function takes a candidate dict and returns a float score.
All scores are normalized to [0, 1] (or [0.5, 1.15] for avail_mod).
"""

import math
import re
from datetime import datetime, date

from config import (
    TIER1_SKILLS, TIER2_SKILLS, TIER3_SKILLS, ALL_JD_SKILLS,
    SKILL_TIER_WEIGHT, PROFICIENCY_WEIGHT,
    TECH_TITLE_RE, ML_AI_TITLE_RE, NON_TECH_TITLES,
    CAREER_EVIDENCE_RE, PRODUCTION_RE,
    CONSULTING_FIRMS,
    PRODUCT_INDUSTRIES,
    TIER1_LOCATIONS, TIER2_LOCATIONS, INDIA_KEYWORDS,
    CV_SPEECH_ROBOTICS_SKILLS, NLP_IR_SKILLS,
    experience_score,
)

# Reference date for "days since last active" — use a fixed date for reproducibility
REFERENCE_DATE = date(2026, 6, 25)


def _normalize(text: str) -> str:
    """Lowercase and strip a string for matching."""
    return (text or "").strip().lower()


def _is_consulting(company_name: str) -> bool:
    """Check if a company name matches the consulting firms list."""
    name = _normalize(company_name)
    for firm in CONSULTING_FIRMS:
        if firm in name:
            return True
    return False


def _is_tech_title(title: str) -> bool:
    """Check if title matches any tech/engineering pattern."""
    for pattern in TECH_TITLE_RE:
        if pattern.search(title):
            return True
    return False


def _is_ml_ai_title(title: str) -> bool:
    """Check if title matches high-value ML/AI patterns."""
    for pattern in ML_AI_TITLE_RE:
        if pattern.search(title):
            return True
    return False


def _is_non_tech_title(title: str) -> bool:
    """Check if title matches known non-tech titles."""
    title_lower = _normalize(title)
    for nt in NON_TECH_TITLES:
        if nt in title_lower:
            return True
    return False


def _has_career_evidence(descriptions: list[str]) -> tuple[float, list[str]]:
    """
    Check career descriptions for evidence of building ranking/retrieval/recsys.

    Returns:
        (score: float [0-1], evidence_snippets: list[str])
    """
    evidence_count = 0
    snippets = []
    for desc in descriptions:
        if not desc:
            continue
        for pattern in CAREER_EVIDENCE_RE:
            match = pattern.search(desc)
            if match:
                evidence_count += 1
                # Extract a short snippet around the match
                start = max(0, match.start() - 20)
                end = min(len(desc), match.end() + 30)
                snippets.append(desc[start:end].strip())
                break  # one match per description is enough

    if evidence_count >= 3:
        return 1.0, snippets
    elif evidence_count == 2:
        return 0.8, snippets
    elif evidence_count == 1:
        return 0.5, snippets
    return 0.0, snippets


def _has_production_evidence(descriptions: list[str]) -> float:
    """Check if career descriptions mention production/deployment. Returns [0-1]."""
    prod_count = 0
    for desc in descriptions:
        if not desc:
            continue
        for pattern in PRODUCTION_RE:
            if pattern.search(desc):
                prod_count += 1
                break
    if prod_count >= 2:
        return 1.0
    elif prod_count == 1:
        return 0.6
    return 0.0


# ═══════════════════════════════════════════════════════════════════════════════
# FEATURE 1: Title + Career Fit (the decisive anti-stuffer feature)
# ═══════════════════════════════════════════════════════════════════════════════

def title_career_fit(candidate: dict) -> tuple[float, dict]:
    """
    Score how well the candidate's title + career history fits the JD.

    This is the primary anti-stuffer feature. Skills only matter when
    corroborated by title and career evidence.

    Returns: (score [0-1], detail_dict)
    """
    profile = candidate.get("profile", {})
    career = candidate.get("career_history", [])

    current_title = _normalize(profile.get("current_title", ""))
    all_titles = [_normalize(job.get("title", "")) for job in career]
    all_descriptions = [job.get("description", "") or "" for job in career]

    score = 0.0
    detail = {"current_title": current_title, "is_tech": False, "is_ml_ai": False,
              "career_evidence": 0.0, "production_evidence": 0.0}

    # Current title scoring
    if _is_ml_ai_title(current_title):
        score += 0.40
        detail["is_ml_ai"] = True
        detail["is_tech"] = True
    elif _is_tech_title(current_title):
        score += 0.25
        detail["is_tech"] = True
    elif _is_non_tech_title(current_title):
        score += 0.0  # No credit for non-tech titles
    else:
        score += 0.05  # Unknown title, small benefit of the doubt

    # Career history title bonus — any past ML/AI title is a strong signal
    has_past_ml = any(_is_ml_ai_title(t) for t in all_titles)
    has_past_tech = any(_is_tech_title(t) for t in all_titles)

    if has_past_ml:
        score += 0.15
    elif has_past_tech:
        score += 0.08

    # Career evidence — the most important sub-signal
    career_ev, snippets = _has_career_evidence(all_descriptions)
    score += career_ev * 0.30
    detail["career_evidence"] = career_ev
    detail["evidence_snippets"] = snippets

    # Production evidence
    prod_ev = _has_production_evidence(all_descriptions)
    score += prod_ev * 0.15
    detail["production_evidence"] = prod_ev

    return min(score, 1.0), detail


# ═══════════════════════════════════════════════════════════════════════════════
# FEATURE 2: Trust-Weighted Skill Score
# ═══════════════════════════════════════════════════════════════════════════════

def skill_trust_score(candidate: dict, title_is_tech: bool) -> tuple[float, dict]:
    """
    Compute trust-weighted skill overlap with JD-relevant skills.
    Skills only "count" when the title/career corroborates them (anti-stuffer gate).

    Returns: (score [0-1], detail_dict)
    """
    skills = candidate.get("skills", [])
    if not skills:
        return 0.0, {"matched_skills": [], "raw_score": 0.0}

    # If title is non-tech, heavily discount skill scores (stuffer protection)
    gate_multiplier = 1.0 if title_is_tech else 0.15

    total_trust = 0.0
    max_possible = 0.0
    matched = []

    for skill in skills:
        name = _normalize(skill.get("name", ""))
        prof = _normalize(skill.get("proficiency", "intermediate"))
        endorsements = skill.get("endorsements", 0) or 0
        duration_months = skill.get("duration_months", 0) or 0

        # Determine tier
        if name in TIER1_SKILLS:
            tier_w = SKILL_TIER_WEIGHT["tier1"]
        elif name in TIER2_SKILLS:
            tier_w = SKILL_TIER_WEIGHT["tier2"]
        elif name in TIER3_SKILLS:
            tier_w = SKILL_TIER_WEIGHT["tier3"]
        else:
            continue  # Not a JD-relevant skill

        prof_w = PROFICIENCY_WEIGHT.get(prof, 0.3)
        trust = prof_w * math.log1p(duration_months) * math.log1p(endorsements) * tier_w

        total_trust += trust
        max_possible += 1.0 * math.log1p(60) * math.log1p(50) * SKILL_TIER_WEIGHT["tier1"]

        matched.append({
            "name": skill.get("name", ""),
            "proficiency": prof,
            "duration_months": duration_months,
            "endorsements": endorsements,
            "trust": round(trust, 3),
        })

    if max_possible == 0:
        raw_score = 0.0
    else:
        raw_score = min(total_trust / (max_possible * 0.3), 1.0)  # soft cap

    final_score = raw_score * gate_multiplier

    # Sort matched skills by trust score (highest first)
    matched.sort(key=lambda x: x["trust"], reverse=True)

    return final_score, {"matched_skills": matched[:8], "raw_score": round(raw_score, 4)}


# ═══════════════════════════════════════════════════════════════════════════════
# FEATURE 3: Experience Fit
# ═══════════════════════════════════════════════════════════════════════════════

def experience_fit(candidate: dict) -> tuple[float, dict]:
    """Score experience band fit (ideal 6-8 years). Returns (score, detail)."""
    yoe = candidate.get("profile", {}).get("years_of_experience", 0) or 0
    score = experience_score(yoe)
    return score, {"years_of_experience": yoe}


# ═══════════════════════════════════════════════════════════════════════════════
# FEATURE 4: Product Company Signal
# ═══════════════════════════════════════════════════════════════════════════════

def product_company_signal(candidate: dict) -> tuple[float, dict]:
    """
    Score based on product-company vs consulting experience.
    Returns (score [0-1], detail).
    """
    career = candidate.get("career_history", [])
    if not career:
        return 0.3, {"product_jobs": 0, "consulting_jobs": 0, "total_jobs": 0}

    product_jobs = 0
    consulting_jobs = 0
    product_months = 0
    total_months = 0

    for job in career:
        company = job.get("company", "")
        industry = _normalize(job.get("industry", ""))
        company_size = job.get("company_size", "")
        duration = job.get("duration_months", 0) or 0
        total_months += duration

        if _is_consulting(company):
            consulting_jobs += 1
        else:
            # Check if it's likely a product company
            is_product = False
            if any(ind in industry for ind in PRODUCT_INDUSTRIES):
                is_product = True
            elif company_size in ("1-10", "11-50", "51-200", "201-500"):
                # Smaller companies are more likely product companies
                is_product = True

            if is_product:
                product_jobs += 1
                product_months += duration
            else:
                # Unknown — give partial credit
                product_jobs += 0.3

    total_jobs = len(career)

    if total_jobs == 0:
        return 0.3, {"product_jobs": 0, "consulting_jobs": 0, "total_jobs": 0}

    # Score based on ratio of product experience
    product_ratio = product_jobs / total_jobs if total_jobs > 0 else 0
    consulting_ratio = consulting_jobs / total_jobs if total_jobs > 0 else 0

    score = product_ratio * 0.7 + (1.0 - consulting_ratio) * 0.3

    return min(score, 1.0), {
        "product_jobs": product_jobs,
        "consulting_jobs": consulting_jobs,
        "total_jobs": total_jobs,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# FEATURE 5: Location Fit
# ═══════════════════════════════════════════════════════════════════════════════

def location_fit(candidate: dict) -> tuple[float, dict]:
    """Score location fit for the JD. Returns (score [0-1], detail)."""
    profile = candidate.get("profile", {})
    signals = candidate.get("redrob_signals", {})

    location = _normalize(profile.get("location", ""))
    country = _normalize(profile.get("country", ""))
    willing_to_relocate = signals.get("willing_to_relocate", False)

    # Check tier 1 locations (Pune/Noida)
    for loc in TIER1_LOCATIONS:
        if loc in location:
            return 1.0, {"location": location, "tier": "tier1"}

    # Check tier 2 locations (other major Indian cities)
    for loc in TIER2_LOCATIONS:
        if loc in location:
            return 0.85, {"location": location, "tier": "tier2"}

    # Check if in India
    is_india = "india" in country or "india" in location
    if is_india:
        if willing_to_relocate:
            return 0.65, {"location": location, "tier": "india_relocatable"}
        else:
            return 0.45, {"location": location, "tier": "india_other"}

    # Outside India
    if willing_to_relocate:
        return 0.25, {"location": location, "tier": "international_relocatable"}

    return 0.10, {"location": location, "tier": "international"}


# ═══════════════════════════════════════════════════════════════════════════════
# FEATURE 6: Education Signal (light tiebreaker)
# ═══════════════════════════════════════════════════════════════════════════════

def education_signal(candidate: dict) -> tuple[float, dict]:
    """Light education tiebreaker. Returns (score [0-1], detail)."""
    education = candidate.get("education", [])
    if not education:
        return 0.3, {"degree": "none", "field": "none", "tier": "none"}

    best_score = 0.3
    best_detail = {}

    for edu in education:
        score = 0.3
        field = _normalize(edu.get("field_of_study", ""))
        degree = _normalize(edu.get("degree", ""))
        tier = _normalize(edu.get("tier", ""))

        # Relevant field boost
        relevant_fields = {"computer science", "cs", "information technology", "it",
                          "artificial intelligence", "machine learning", "data science",
                          "electronics", "ece", "electrical", "mathematics", "statistics",
                          "computational"}
        if any(f in field for f in relevant_fields):
            score += 0.25

        # Degree level boost
        if any(d in degree for d in ("phd", "ph.d", "doctorate")):
            score += 0.20
        elif any(d in degree for d in ("m.tech", "m.e.", "m.s.", "msc", "m.sc", "master")):
            score += 0.15
        elif any(d in degree for d in ("b.tech", "b.e.", "bsc", "b.sc", "bachelor")):
            score += 0.05

        # Tier boost
        if tier in ("tier_1", "tier1"):
            score += 0.20
        elif tier in ("tier_2", "tier2"):
            score += 0.10
        elif tier in ("tier_3", "tier3"):
            score += 0.05

        if score > best_score:
            best_score = score
            best_detail = {"degree": degree, "field": field, "tier": tier}

    return min(best_score, 1.0), best_detail


# ═══════════════════════════════════════════════════════════════════════════════
# PENALTY 1: Keyword Stuffer
# ═══════════════════════════════════════════════════════════════════════════════

def keyword_stuffer_penalty(candidate: dict, title_detail: dict) -> float:
    """
    Penalty for non-tech title + many AI skills (keyword stuffer).
    Returns penalty value [0-1] (higher = more penalized).
    """
    profile = candidate.get("profile", {})
    skills = candidate.get("skills", [])
    current_title = _normalize(profile.get("current_title", ""))

    is_non_tech = _is_non_tech_title(current_title)
    is_tech = title_detail.get("is_tech", False)

    if not is_non_tech:
        return 0.0

    # Count AI-relevant skills
    ai_skill_count = sum(
        1 for s in skills
        if _normalize(s.get("name", "")) in ALL_JD_SKILLS
    )

    # Non-tech title with many AI skills = stuffer
    career_evidence = title_detail.get("career_evidence", 0)

    if ai_skill_count >= 6 and career_evidence < 0.3:
        return 1.0  # Strong stuffer signal
    elif ai_skill_count >= 4 and career_evidence < 0.3:
        return 0.8
    elif ai_skill_count >= 3 and is_non_tech and not is_tech:
        return 0.5
    elif is_non_tech and not is_tech:
        return 0.3

    return 0.0


# ═══════════════════════════════════════════════════════════════════════════════
# PENALTY 2: Consulting-Only
# ═══════════════════════════════════════════════════════════════════════════════

def consulting_only_penalty(candidate: dict) -> float:
    """
    Penalty for ALL career history at consulting firms.
    Reduced if there's any product-company experience.
    Returns penalty [0-1].
    """
    career = candidate.get("career_history", [])
    if not career:
        return 0.0

    consulting_count = sum(1 for job in career if _is_consulting(job.get("company", "")))

    if consulting_count == len(career):
        return 1.0  # All consulting
    elif consulting_count > len(career) * 0.7:
        return 0.5  # Mostly consulting
    elif consulting_count > 0:
        return 0.1  # Some consulting
    return 0.0


# ═══════════════════════════════════════════════════════════════════════════════
# PENALTY 3: Research-Only
# ═══════════════════════════════════════════════════════════════════════════════

def research_only_penalty(candidate: dict) -> float:
    """
    Penalty for pure research / academic with no production deployment.
    Returns penalty [0-1].
    """
    career = candidate.get("career_history", [])
    if not career:
        return 0.0

    all_titles = [_normalize(job.get("title", "")) for job in career]
    all_descs = [job.get("description", "") or "" for job in career]

    research_titles = sum(
        1 for t in all_titles
        if "research" in t and not any(w in t for w in ["engineer", "developer"])
    )

    # Check for production language
    has_production = _has_production_evidence(all_descs) > 0.3

    if research_titles == len(career) and not has_production:
        return 1.0  # Pure research, no production
    elif research_titles > len(career) * 0.7 and not has_production:
        return 0.6
    return 0.0


# ═══════════════════════════════════════════════════════════════════════════════
# PENALTY 4: CV / Speech / Robotics Specialist
# ═══════════════════════════════════════════════════════════════════════════════

def cv_speech_penalty(candidate: dict) -> float:
    """
    Penalty for candidates whose primary skills are CV/Speech/Robotics
    with little NLP/IR exposure.
    Returns penalty [0-1].
    """
    skills = candidate.get("skills", [])
    if not skills:
        return 0.0

    cv_speech_count = 0
    nlp_ir_count = 0

    for skill in skills:
        name = _normalize(skill.get("name", ""))
        if name in CV_SPEECH_ROBOTICS_SKILLS:
            cv_speech_count += 1
        if name in NLP_IR_SKILLS:
            nlp_ir_count += 1

    if cv_speech_count >= 3 and nlp_ir_count <= 1:
        return 0.8  # Primarily CV/Speech, little NLP/IR
    elif cv_speech_count >= 2 and nlp_ir_count == 0:
        return 0.5
    return 0.0


# ═══════════════════════════════════════════════════════════════════════════════
# PENALTY 5: Job Hopper
# ═══════════════════════════════════════════════════════════════════════════════

def job_hopper_penalty(candidate: dict) -> float:
    """
    Penalty for frequent job changes (avg tenure < 1.5 years).
    JD explicitly says: "we need someone who plans to be here for 3+ years."
    Returns penalty [0-1].
    """
    career = candidate.get("career_history", [])
    if len(career) <= 1:
        return 0.0

    durations = [job.get("duration_months", 0) or 0 for job in career]
    avg_tenure = sum(durations) / len(durations) if durations else 0

    if avg_tenure < 12:  # Less than 1 year average
        return 0.8
    elif avg_tenure < 18:  # Less than 1.5 years average
        return 0.5
    elif avg_tenure < 24:  # Less than 2 years average
        return 0.2
    return 0.0


# ═══════════════════════════════════════════════════════════════════════════════
# BEHAVIORAL AVAILABILITY MODIFIER
# ═══════════════════════════════════════════════════════════════════════════════

def availability_modifier(candidate: dict) -> tuple[float, dict]:
    """
    Behavioral availability modifier — bounded multiplier [0.5, 1.15].
    A cold candidate is discounted but not zeroed.

    Factors: response rate, recency, open-to-work, interview completion, notice period.
    Returns: (modifier, detail_dict)
    """
    signals = candidate.get("redrob_signals", {})

    response_rate = signals.get("recruiter_response_rate", 0.5) or 0
    open_to_work = signals.get("open_to_work_flag", False)
    interview_completion = signals.get("interview_completion_rate", 0.5) or 0
    notice_days = signals.get("notice_period_days", 60) or 60

    # Days since last active
    last_active_str = signals.get("last_active_date", "")
    if last_active_str:
        try:
            last_active = date.fromisoformat(last_active_str)
            days_inactive = (REFERENCE_DATE - last_active).days
        except (ValueError, TypeError):
            days_inactive = 180  # assume cold
    else:
        days_inactive = 180

    # ── Factor 1: Recruiter response rate (biggest behavioral signal) ────
    # Weight: 35%
    if response_rate >= 0.7:
        resp_score = 1.0
    elif response_rate >= 0.4:
        resp_score = 0.7 + (response_rate - 0.4) / 0.3 * 0.3
    elif response_rate >= 0.1:
        resp_score = 0.3 + (response_rate - 0.1) / 0.3 * 0.4
    else:
        resp_score = 0.2

    # ── Factor 2: Recency / Activity ─────────────────────────────────────
    # Weight: 25%
    if days_inactive <= 14:
        recency_score = 1.0
    elif days_inactive <= 30:
        recency_score = 0.9
    elif days_inactive <= 60:
        recency_score = 0.75
    elif days_inactive <= 120:
        recency_score = 0.5
    else:
        recency_score = 0.3

    # ── Factor 3: Open to work ───────────────────────────────────────────
    # Weight: 15%
    otw_score = 1.0 if open_to_work else 0.4

    # ── Factor 4: Interview completion rate ──────────────────────────────
    # Weight: 15%
    if interview_completion >= 0.8:
        interview_score = 1.0
    elif interview_completion >= 0.5:
        interview_score = 0.7
    else:
        interview_score = 0.4

    # ── Factor 5: Notice period ──────────────────────────────────────────
    # Weight: 10%
    if notice_days <= 30:
        notice_score = 1.0
    elif notice_days <= 60:
        notice_score = 0.7
    elif notice_days <= 90:
        notice_score = 0.4
    else:
        notice_score = 0.2

    # Weighted combination
    raw = (
        0.35 * resp_score +
        0.25 * recency_score +
        0.15 * otw_score +
        0.15 * interview_score +
        0.10 * notice_score
    )

    # Map to [0.5, 1.15] — never zero, never more than 15% boost
    modifier = 0.5 + raw * 0.65

    detail = {
        "response_rate": response_rate,
        "days_inactive": days_inactive,
        "open_to_work": open_to_work,
        "interview_completion": interview_completion,
        "notice_days": notice_days,
        "modifier": round(modifier, 4),
    }

    return round(modifier, 4), detail


# ═══════════════════════════════════════════════════════════════════════════════
# REASONING GENERATOR
# ═══════════════════════════════════════════════════════════════════════════════

def generate_reasoning(candidate: dict, features: dict) -> str:
    """
    Generate a reasoning string from actual matched features.
    No hallucination — every number is pulled from the profile.
    """
    profile = candidate.get("profile", {})
    signals = candidate.get("redrob_signals", {})

    yoe = profile.get("years_of_experience", 0) or 0
    title = profile.get("current_title", "Unknown")
    company = profile.get("current_company", "Unknown")

    parts = []

    # Core identity
    parts.append(f"{yoe:.1f} yrs exp; {title} at {company}")

    # Career evidence
    title_detail = features.get("title_career_detail", {})
    if title_detail.get("career_evidence", 0) >= 0.5:
        snippets = title_detail.get("evidence_snippets", [])
        if snippets:
            # Take shortest snippet
            snippet = min(snippets, key=len)[:80]
            parts.append(f"career evidence: {snippet}")
        else:
            parts.append("strong career evidence of building ranking/retrieval systems")

    # Top matched skills
    skill_detail = features.get("skill_detail", {})
    matched = skill_detail.get("matched_skills", [])
    if matched:
        top_skills = [s["name"] for s in matched[:4]]
        parts.append(f"key skills: {', '.join(top_skills)}")

    # Location
    loc_detail = features.get("location_detail", {})
    location = loc_detail.get("location", "")
    if location:
        parts.append(f"location: {location}")

    # Availability concern (if any)
    avail_detail = features.get("avail_detail", {})
    concerns = []
    resp_rate = avail_detail.get("response_rate", 0)
    days_inactive = avail_detail.get("days_inactive", 0)

    if resp_rate < 0.3:
        concerns.append(f"low response rate ({resp_rate:.2f})")
    if days_inactive > 90:
        concerns.append(f"inactive {days_inactive}d")
    if not avail_detail.get("open_to_work", True):
        concerns.append("not open to work")

    if concerns:
        parts.append(f"concerns: {'; '.join(concerns)}")

    # Penalties
    penalties = []
    if features.get("stuffer_penalty", 0) > 0.3:
        penalties.append("non-tech title with AI skills (possible stuffer)")
    if features.get("consulting_penalty", 0) > 0.5:
        penalties.append("consulting-heavy career")
    if features.get("honeypot", False):
        penalties.append("HONEYPOT — impossible profile data")

    if penalties:
        parts.append(f"flags: {'; '.join(penalties)}")

    reasoning = ". ".join(parts) + "."

    # Ensure it's not too long (keep under 300 chars for CSV readability)
    if len(reasoning) > 350:
        reasoning = reasoning[:347] + "..."

    return reasoning
