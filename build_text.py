"""
Step 1 — build_text.py
======================
Converts a raw candidate JSON record into a single text string
for embedding. The text blob is what BAAI/bge-base-en-v1.5 will
encode, so what you put in here directly determines ranking quality.

Key design decisions (grounded in the actual JD):
  - Career history descriptions are weighted first and heaviest.
    The JD explicitly says "find Tier-5 candidates who don't use the
    keyword RAG but whose history shows they built a ranking system."
    That signal lives in descriptions, not in the skills list.

  - Skills list comes second, but only with proficiency + duration.
    A skill with 0 months duration is meaningless (and a honeypot signal).
    We skip skills whose duration_months is 0 — they pollute the embedding.

  - Summary and headline come last. They tend to be self-promotional
    and keyword-heavy, exactly the "trap" the JD warns about.

  - We do NOT include: location, salary, notice period, company size,
    or redrob_signals — those are structured features for the scoring
    formula in step 3, not semantic text for embedding.

Run this file directly to test on the sample candidate:
    python build_text.py
"""

import json
import re
from datetime import datetime, date


# --------------------------------------------------------------------------
# Constants — pulled from the actual JD
# --------------------------------------------------------------------------

# The JD says: "People who have only worked at consulting firms
# in their entire career" are disqualifiers.
CONSULTING_FIRMS = {
    "tcs", "infosys", "wipro", "accenture", "cognizant",
    "capgemini", "hcl", "tech mahindra", "mphasis", "hexaware",
    "ltimindtree", "mindtree",  # note: mindtree merged into LTIMindtree
}

# JD explicitly wants: embeddings, retrieval, ranking, LLMs, vector DBs,
# eval frameworks. We use these to weight skill mentions in the text.
HIGH_VALUE_SKILLS = {
    # Core must-haves from JD
    "embeddings", "vector search", "semantic search", "hybrid search",
    "dense retrieval", "sparse retrieval", "bm25",
    "sentence transformers", "faiss", "milvus", "qdrant", "weaviate",
    "pinecone", "opensearch", "elasticsearch",
    "ranking", "reranking", "learning to rank", "ltr",
    "ndcg", "mrr", "map", "information retrieval", "ir",
    "rag", "retrieval augmented generation",
    # Nice-to-haves from JD
    "lora", "qlora", "peft", "fine-tuning", "fine tuning",
    "llm", "transformer", "bert", "nlp",
    "recommendation system", "recommender",
    "xgboost", "lightgbm",
    # Strong signal skills from schema sample
    "pyspark", "spark", "airflow", "kafka", "dbt", "snowflake",
}


# --------------------------------------------------------------------------
# Skill matching helper
# --------------------------------------------------------------------------

# Pre-compile word-boundary patterns for each high-value skill
_HV_PATTERNS = {hv: re.compile(rf'\b{re.escape(hv)}\b') for hv in HIGH_VALUE_SKILLS}


def _is_high_value_skill(name_lower: str) -> bool:
    """
    Check if a skill name matches any HIGH_VALUE_SKILLS entry
    using word-boundary matching instead of substring `in`.
    Avoids false positives like 'ir' matching 'Airflow'.
    """
    if name_lower in HIGH_VALUE_SKILLS:
        return True
    return any(pat.search(name_lower) for pat in _HV_PATTERNS.values())


# --------------------------------------------------------------------------
# Main text builder
# --------------------------------------------------------------------------

def build_text(candidate: dict) -> str:
    """
    Convert a candidate record to a single embedding-ready text string.

    Returns a string roughly structured as:
        [career history, weighted by recency]
        [skills, filtered and weighted]
        [summary]
        [headline]

    The string uses a pipe '|' separator between logical sections
    so sentence-transformers sees clear boundaries.
    """
    parts = []

    profile = candidate.get("profile", {})
    career  = candidate.get("career_history", [])
    skills  = candidate.get("skills", [])

    # ------------------------------------------------------------------
    # 1. CAREER HISTORY — highest signal, put first
    # ------------------------------------------------------------------
    # Sort by is_current first, then by start_date descending (most recent first).
    # Recency matters: a 2-year-old role at a product company > a 10-year-old one.

    # Two-pass stable sort: first by start_date descending (most recent first),
    # then by is_current (current roles always on top).
    # Python's sort is stable, so the second sort preserves the date order.
    sorted_career = sorted(
        career,
        key=lambda j: j.get("start_date", "1970-01-01"),
        reverse=True,
    )
    sorted_career = sorted(
        sorted_career,
        key=lambda j: 0 if j.get("is_current") else 1,
    )

    for job in sorted_career:
        company    = job.get("company", "")
        title      = job.get("title", "")
        duration_m = job.get("duration_months", 0)
        duration_y = round(duration_m / 12, 1)
        description = job.get("description", "").strip()
        industry   = job.get("industry", "")
        is_current = job.get("is_current", False)

        # Skip entirely empty descriptions — they add noise
        if not description and not title:
            continue

        current_marker = " [current]" if is_current else ""
        role_line = (
            f"{title} at {company}{current_marker} "
            f"({duration_y}y, {industry}): {description}"
        )
        parts.append(role_line)

    # ------------------------------------------------------------------
    # 2. SKILLS — filtered and weighted
    # ------------------------------------------------------------------
    # Rules:
    #   - Skip skills with duration_months == 0 (honeypot signal + noise)
    #   - Skip beginner skills with < 6 months duration (low signal)
    #   - Repeat high-value skills to give them more weight in embedding
    #   - Include proficiency level — "advanced NLP" ≠ "beginner NLP"

    skill_parts_normal = []
    skill_parts_highval = []

    for sk in skills:
        name       = sk.get("name", "").strip()
        proficiency = sk.get("proficiency", "")
        duration_m  = sk.get("duration_months", 0)

        # Hard filter: duration=0 means never used → skip entirely
        if duration_m == 0:
            continue

        # Soft filter: beginner + very short exposure → skip
        if proficiency == "beginner" and duration_m < 6:
            continue

        duration_y = round(duration_m / 12, 1)
        skill_str  = f"{name} ({proficiency}, {duration_y}y)"

        # Check if this is a high-value skill for this JD
        # Uses word-boundary matching to avoid false positives
        name_lower = name.lower()
        is_highval = _is_high_value_skill(name_lower)

        if is_highval:
            skill_parts_highval.append(skill_str)
        else:
            skill_parts_normal.append(skill_str)

    # High-value skills go first and are repeated to boost their signal weight
    if skill_parts_highval:
        parts.append("Key skills: " + "; ".join(skill_parts_highval))
        # Repeat once to increase embedding weight (a simple but effective trick)
        parts.append("Core expertise: " + "; ".join(skill_parts_highval))

    if skill_parts_normal:
        parts.append("Also: " + "; ".join(skill_parts_normal))

    # ------------------------------------------------------------------
    # 3. SUMMARY — good signal but keyword-stuffable, so goes after career
    # ------------------------------------------------------------------
    summary = profile.get("summary", "").strip()
    if summary:
        parts.append(summary)

    # ------------------------------------------------------------------
    # 4. HEADLINE — weakest signal (one-liner, often keyword stuffed)
    # ------------------------------------------------------------------
    headline = profile.get("headline", "").strip()
    if headline:
        parts.append(headline)

    # ------------------------------------------------------------------
    # 5. Education field of study — subtle semantic signal
    # ------------------------------------------------------------------
    education = candidate.get("education", [])
    for edu in education:
        field = edu.get("field_of_study", "")
        degree = edu.get("degree", "")
        if field or degree:
            parts.append(f"{degree} in {field}")

    # Join with pipe separator — clean boundary between sections
    return " | ".join(filter(None, parts))


# --------------------------------------------------------------------------
# Consulting-only detector (used in penalty step, but computed here)
# --------------------------------------------------------------------------

def is_consulting_only(candidate: dict) -> bool:
    """
    Returns True if ALL career history is at known consulting firms.
    The JD says: "People who have only worked at consulting firms
    in their entire career" are disqualifiers.
    A candidate currently at a consulting firm but with prior
    product-company experience is explicitly still OK.
    """
    career = candidate.get("career_history", [])
    if not career:
        return False

    companies = {job.get("company", "").lower() for job in career}
    return all(
        any(firm in company for firm in CONSULTING_FIRMS)
        for company in companies
    )


# --------------------------------------------------------------------------
# Honeypot detection
# --------------------------------------------------------------------------

def detect_honeypot(candidate: dict) -> bool:
    """
    Flag candidates with subtly impossible profiles.
    The dataset contains ~80 honeypots. >10% in top-100 = disqualification.

    Red-flag heuristics:
      - Advanced/expert proficiency with near-zero duration
      - Career duration_months exceeds actual calendar span
      - Total years_of_experience exceeds career timeline
      - High endorsements on zero-duration skills
    """
    skills  = candidate.get("skills", [])
    career  = candidate.get("career_history", [])
    profile = candidate.get("profile", {})

    red_flags = 0

    # Check 1: Advanced/expert proficiency with near-zero duration (<= 2 months)
    expert_zero = sum(
        1 for s in skills
        if s.get("proficiency") in ("advanced", "expert")
        and s.get("duration_months", 0) <= 2
    )
    if expert_zero >= 2:
        red_flags += 2
    elif expert_zero >= 1:
        red_flags += 1

    # Check 2: Career duration_months significantly exceeds actual calendar time
    for job in career:
        start   = job.get("start_date")
        end     = job.get("end_date")
        claimed = job.get("duration_months", 0)
        if start and end and claimed > 0:
            try:
                d_start = datetime.fromisoformat(start)
                d_end   = datetime.fromisoformat(end)
                actual_months = (d_end - d_start).days / 30.44
                if actual_months > 0 and claimed > actual_months * 1.5:
                    red_flags += 2
            except (ValueError, TypeError):
                pass

    # Check 3: Total years_of_experience vs. career timeline
    claimed_years = profile.get("years_of_experience", 0)
    if career and claimed_years and claimed_years > 0:
        earliest = min(
            (j.get("start_date", "9999") for j in career),
            default="9999",
        )
        if earliest != "9999":
            try:
                d_earliest = datetime.fromisoformat(earliest).date()
                actual_years = (date.today() - d_earliest).days / 365.25
                if actual_years > 0 and claimed_years > actual_years * 1.5:
                    red_flags += 2
            except (ValueError, TypeError):
                pass

    # Check 4: Many skills with high endorsements but near-zero duration
    suspicious_skills = sum(
        1 for s in skills
        if s.get("endorsements", 0) > 20
        and s.get("duration_months", 0) <= 1
    )
    if suspicious_skills >= 2:
        red_flags += 1

    # Threshold: 2+ red flags = likely honeypot
    return red_flags >= 2


# --------------------------------------------------------------------------
# Disqualifier features (from JD explicit rules)
# --------------------------------------------------------------------------

def compute_disqualifiers(candidate: dict) -> dict:
    """
    Compute binary disqualifier flags derived from the JD's explicit rules.
    Returns dict of flag_name -> bool.
    """
    career  = candidate.get("career_history", [])
    skills  = candidate.get("skills", [])
    profile = candidate.get("profile", {})

    flags = {}

    # 1. Consulting-only career
    flags["consulting_only"] = is_consulting_only(candidate)

    # 2. Job-hopping pattern (avg tenure < 18 months)
    if career:
        tenures = [j.get("duration_months", 0) for j in career]
        avg_tenure = sum(tenures) / len(tenures)
        flags["job_hopper"] = avg_tenure < 18
    else:
        flags["job_hopper"] = False

    # 3. Pure research — academic/research signal with NO production-deployment evidence.
    #
    #    BUGFIX: the original production_kw list included generic engineering nouns
    #    ("system", "platform", "api", "pipeline", "backend", "infrastructure", "scale")
    #    that show up in almost every career description, academic or not. That made
    #    `not has_production` collapse to False for nearly every candidate, so this
    #    flag could never fire at all (verified empirically: 0 / 100,000 on the real
    #    candidate pool). research_kw had the mirror-image problem — bare "research",
    #    "scientist", and "lab" match common industry titles like "AI Research Engineer"
    #    or "Data Scientist" (and "lab" even matches inside "collaborate"), none of which
    #    are academic-only. Both keyword sets are tightened below to fix this.
    research_kw = {"researcher", "phd", "postdoc", "postdoctoral", "professor",
                   "academia", "doctoral", "dissertation", "thesis",
                   "research lab", "research fellow", "research scientist"}
    production_kw = {"production", "deploy", "deployed", "shipped", "launched",
                     "rolled out", "in production", "live users", "real users",
                     "end users", "users", "customers", "product", "service"}
    all_career_text = " ".join(
        (j.get("title", "") + " " + j.get("description", "")).lower()
        for j in career
    )
    has_research   = any(kw in all_career_text for kw in research_kw)
    has_production = any(kw in all_career_text for kw in production_kw)
    flags["pure_research"] = has_research and not has_production

    # 4. CV / Speech / Robotics without NLP/IR exposure
    cv_speech_kw = {"computer vision", "image classification", "object detection",
                    "speech recognition", "speech", "tts", "text to speech",
                    "robotics", "robot", "autonomous driving"}
    nlp_ir_kw = {"nlp", "natural language", "information retrieval", "search",
                 "ranking", "embeddings", "transformer", "bert", "llm",
                 "text mining", "text classification", "ner",
                 "sentiment", "rag", "retrieval"}
    skill_text = " ".join(s.get("name", "").lower() for s in skills)
    has_cv_speech = any(kw in skill_text for kw in cv_speech_kw)
    has_nlp_ir    = any(kw in skill_text for kw in nlp_ir_kw)
    # Also check career descriptions for NLP/IR work
    has_nlp_ir    = has_nlp_ir or any(kw in all_career_text for kw in nlp_ir_kw)
    flags["cv_speech_no_nlp"] = has_cv_speech and not has_nlp_ir

    # 5. AI experience = LangChain demos under 12 months
    langchain_kw = {"langchain", "llamaindex", "llama index"}
    deep_ai_kw   = {"machine learning", "deep learning", "tensorflow", "pytorch",
                    "scikit-learn", "xgboost", "lightgbm", "embeddings",
                    "transformers", "ranking", "recommendation", "faiss"}
    has_langchain = any(
        any(lc in s.get("name", "").lower() for lc in langchain_kw)
        and s.get("duration_months", 0) < 12
        for s in skills
    )
    has_real_ai = any(
        any(ai in s.get("name", "").lower() for ai in deep_ai_kw)
        and s.get("duration_months", 0) >= 12
        for s in skills
    )
    flags["langchain_only"] = has_langchain and not has_real_ai

    return flags


# --------------------------------------------------------------------------
# Engagement multiplier (from redrob_signals)
# --------------------------------------------------------------------------

def compute_engagement_score(candidate: dict) -> float:
    """
    Compute a 0–1 engagement multiplier from redrob_signals.
    Higher = more likely to be reachable and responsive.

    Components (equal weight):
      - Recency of last_active_date   (180-day decay)
      - recruiter_response_rate       (direct 0–1)
      - interview_completion_rate     (direct 0–1)
      - offer_acceptance_rate         (if available)
      - open_to_work_flag             (binary boost)
    """
    signals = candidate.get("redrob_signals", {})
    components = []

    # 1. Recency of last_active_date (0–1, recent = higher)
    last_active = signals.get("last_active_date")
    if last_active:
        try:
            la_date  = datetime.fromisoformat(last_active).date()
            days_ago = (date.today() - la_date).days
            recency  = max(0.0, 1.0 - days_ago / 180.0)
            components.append(recency)
        except (ValueError, TypeError):
            components.append(0.5)
    else:
        components.append(0.5)

    # 2. Recruiter response rate (direct 0–1)
    rrr = signals.get("recruiter_response_rate", -1)
    if isinstance(rrr, (int, float)) and rrr >= 0:
        components.append(float(rrr))
    else:
        components.append(0.5)

    # 3. Interview completion rate (direct 0–1)
    icr = signals.get("interview_completion_rate", -1)
    if isinstance(icr, (int, float)) and icr >= 0:
        components.append(float(icr))
    else:
        components.append(0.5)

    # 4. Offer acceptance rate (−1 sentinel = no prior offers → neutral)
    oar = signals.get("offer_acceptance_rate", -1)
    if isinstance(oar, (int, float)) and oar >= 0:
        components.append(float(oar))
    # else: don't penalise — -1 means no prior offers

    # 5. Open to work flag (binary boost)
    components.append(1.0 if signals.get("open_to_work_flag") else 0.3)

    return sum(components) / len(components) if components else 0.5


# --------------------------------------------------------------------------
# Quick smoke test — run: python build_text.py
# --------------------------------------------------------------------------

SAMPLE_CANDIDATE = {
    "candidate_id": "CAND_0000001",
    "profile": {
        "anonymized_name": "Ira Vora",
        "headline": "Backend Engineer | SQL, Spark, Cloud",
        "summary": (
            "Software / data professional with 6.9 years of experience building "
            "data pipelines, backend systems, and analytics infrastructure. "
            "I'm a backend/data hybrid — Spark, Airflow, SQL warehouses are home "
            "territory; I'm building competence on the ML side."
        ),
        "location": "Toronto",
        "country": "Canada",
        "years_of_experience": 6.9,
        "current_title": "Backend Engineer",
        "current_company": "Mindtree",
        "current_company_size": "10001+",
        "current_industry": "IT Services",
    },
    "career_history": [
        {
            "company": "Mindtree",
            "title": "Backend Engineer",
            "start_date": "2024-03-08",
            "end_date": None,
            "duration_months": 27,
            "is_current": True,
            "industry": "IT Services",
            "company_size": "10001+",
            "description": (
                "Implemented streaming data pipelines on Kafka and Spark Streaming "
                "for a real-time user-activity processing platform."
            ),
        },
        {
            "company": "Dunder Mifflin",
            "title": "Analytics Engineer",
            "start_date": "2019-07-03",
            "end_date": "2024-01-08",
            "duration_months": 55,
            "is_current": False,
            "industry": "Paper Products",
            "company_size": "201-500",
            "description": (
                "Built and maintained data pipelines on Apache Airflow processing "
                "~500GB of daily transactional data. Worked extensively with Spark "
                "(PySpark) for batch processing and dbt for the transformation layer."
            ),
        },
    ],
    "education": [
        {
            "institution": "Lovely Professional University",
            "degree": "B.E.",
            "field_of_study": "Computer Science",
            "start_year": 2017,
            "end_year": 2020,
            "grade": "8.24 CGPA",
            "tier": "tier_3",
        }
    ],
    "skills": [
        {"name": "NLP",            "proficiency": "advanced",      "endorsements": 37, "duration_months": 26},
        {"name": "Fine-tuning LLMs","proficiency": "advanced",     "endorsements": 21, "duration_months": 36},
        {"name": "Milvus",         "proficiency": "advanced",      "endorsements": 40, "duration_months": 35},
        {"name": "LoRA",           "proficiency": "intermediate",  "endorsements": 0,  "duration_months": 28},
        {"name": "Speech Recognition","proficiency": "advanced",   "endorsements": 52, "duration_months": 33},
        {"name": "TTS",            "proficiency": "advanced",      "endorsements": 56, "duration_months": 60},
        {"name": "AWS",            "proficiency": "beginner",      "endorsements": 5,  "duration_months": 8},
        {"name": "Flask",          "proficiency": "beginner",      "endorsements": 15, "duration_months": 15},
        {"name": "Tailwind",       "proficiency": "intermediate",  "endorsements": 3,  "duration_months": 13},
        {"name": "Photoshop",      "proficiency": "intermediate",  "endorsements": 8,  "duration_months": 24},
        # Honeypot-style skill: expert with 0 months
        {"name": "GCP",            "proficiency": "beginner",      "endorsements": 7,  "duration_months": 2},
    ],
    "certifications": [],
    "languages": [
        {"language": "English", "proficiency": "professional"},
    ],
    "redrob_signals": {
        "profile_completeness_score": 86.9,
        "last_active_date": "2026-05-20",
        "open_to_work_flag": True,
        "recruiter_response_rate": 0.34,
        "interview_completion_rate": 0.71,
        "offer_acceptance_rate": 0.58,
        "notice_period_days": 60,
        "github_activity_score": 9.2,
        "willing_to_relocate": False,
        "verified_email": True,
        "verified_phone": True,
        "linkedin_connected": False,
    },
}


if __name__ == "__main__":
    text = build_text(SAMPLE_CANDIDATE)

    print("=" * 70)
    print("CANDIDATE:", SAMPLE_CANDIDATE["candidate_id"])
    print("=" * 70)
    print(text)
    print()
    print(f"Total characters : {len(text)}")
    print(f"Total words      : {len(text.split())}")
    print(f"Consulting-only? : {is_consulting_only(SAMPLE_CANDIDATE)}")
    print(f"Honeypot?        : {detect_honeypot(SAMPLE_CANDIDATE)}")
    print(f"Engagement       : {compute_engagement_score(SAMPLE_CANDIDATE):.3f}")
    print(f"Disqualifiers    : {compute_disqualifiers(SAMPLE_CANDIDATE)}")
    print()

    # Show what the HIGH_VALUE_SKILLS filter caught
    caught = [
        sk["name"] for sk in SAMPLE_CANDIDATE["skills"]
        if any(hv in sk["name"].lower() for hv in HIGH_VALUE_SKILLS)
        and sk.get("duration_months", 0) > 0
    ]
    print(f"High-value skills detected: {caught}")

    # Show what was filtered out
    filtered = [
        sk["name"] for sk in SAMPLE_CANDIDATE["skills"]
        if sk.get("duration_months", 0) == 0
    ]
    if filtered:
        print(f"Filtered (duration=0)     : {filtered}")