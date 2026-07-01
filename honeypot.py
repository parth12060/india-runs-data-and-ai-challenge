"""
honeypot.py — High-precision honeypot / impossible-profile detection.

DESIGN NOTE (why this is conservative):
    The dataset contains ~80 honeypots with *subtly impossible* profiles.
    They are forced to relevance tier 0 in the ground truth, and a submission
    with >10% honeypots in its top 100 is DISQUALIFIED.

    Naive impossibility heuristics badly over-trigger. Empirically, on the real
    100K pool:
        - "skill duration > years_of_experience"   flags ~9,800  (FALSE POSITIVES)
        - "sum of tenures > yoe"                    flags ~2,800  (FALSE POSITIVES)
    Both fire on normal candidates because skill/role durations legitimately
    overlap and the synthetic data sets skill durations high.

    So we ONLY use *internal contradictions* — cases where the profile's own
    numbers disagree with each other. These are near-certain honeypots and do
    not zero real fits. The union of the three rules below catches ~65 of the
    ~80 on the real pool, with no observed false positives.

Rules (a candidate is a honeypot if ANY fire):
    R1  Expert proficiency in a skill with 0 months of use.
    R2  A job's stated duration_months contradicts its own start/end dates
        (claims > 1.5x the actual calendar span, and by > 12 months).
    R3  years_of_experience far exceeds the calendar span since the earliest
        job start (claims > 1.6x the timeline, and by > 3 years).

All dates compared against a FIXED reference date for reproducibility.
"""

from datetime import date

# Fixed reference date — never use date.today() (non-reproducible across runs).
REFERENCE_DATE = date(2026, 6, 25)


def _parse(d: str):
    try:
        return date.fromisoformat(d)
    except (ValueError, TypeError):
        return None


def is_honeypot(candidate: dict) -> tuple[bool, str]:
    """
    Return (is_honeypot, reason). reason is '' when not a honeypot.
    Uses only high-precision internal-contradiction rules.
    """
    profile = candidate.get("profile", {}) or {}
    skills = candidate.get("skills", []) or []
    career = candidate.get("career_history", []) or []
    yoe = profile.get("years_of_experience", 0) or 0

    # ── R1: Expert proficiency with 0 months used ────────────────────────────
    for s in skills:
        prof = (s.get("proficiency") or "").lower()
        dur = s.get("duration_months", 0) or 0
        if prof == "expert" and dur == 0:
            return True, f"Expert proficiency in '{s.get('name')}' with 0 months of use"

    # ── R2: Job duration contradicts its own start/end calendar span ─────────
    for j in career:
        sd = _parse(j.get("start_date") or "")
        ed = _parse(j.get("end_date") or "") or REFERENCE_DATE
        claimed = j.get("duration_months", 0) or 0
        if sd and claimed > 0:
            actual = (ed - sd).days / 30.44
            if actual > 0 and claimed > actual * 1.5 and (claimed - actual) > 12:
                return True, (
                    f"Role at '{j.get('company')}' claims {claimed} months but its "
                    f"own dates span only ~{actual:.0f} months"
                )

    # ── R3: Total experience far exceeds the career timeline ─────────────────
    starts = [_parse(j.get("start_date") or "") for j in career]
    starts = [s for s in starts if s]
    if starts:
        earliest = min(starts)
        span_years = (REFERENCE_DATE - earliest).days / 365.25
        if span_years > 0 and yoe > span_years * 1.6 and (yoe - span_years) > 3:
            return True, (
                f"Claims {yoe:.1f} years of experience but first job began only "
                f"~{span_years:.1f} years ago"
            )

    return False, ""
