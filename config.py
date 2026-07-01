"""
config.py — All tunable constants, keyword lists, and scoring weights.
Approach A: Structured Feature Ranker for Redrob Hackathon.

Modify weights here to tune ranking without touching feature logic.
"""

import re

# ═══════════════════════════════════════════════════════════════════════════════
# 1. JD-RELEVANT SKILL KEYWORDS (tiered by relevance to the Senior AI Engineer JD)
# ═══════════════════════════════════════════════════════════════════════════════

# Tier 1 — Core skills the JD explicitly asks for (highest weight)
TIER1_SKILLS = {
    "embeddings", "embedding", "sentence-transformers", "sentence transformers",
    "retrieval", "information retrieval", "hybrid search", "hybrid retrieval",
    "vector database", "vector db", "vector search",
    "ranking", "learning to rank", "learning-to-rank",
    "recommendation", "recommendation system", "recommender system", "recsys",
    "search", "search engine", "search system",
    "faiss", "pinecone", "weaviate", "qdrant", "milvus", "opensearch",
    "elasticsearch", "elastic search",
    "ndcg", "mrr", "map", "evaluation framework", "a/b testing", "ab testing",
    "bge", "e5", "openai embeddings",
    "ranking system", "retrieval system", "matching system",
}

# Tier 2 — Strong supporting skills (good signal)
TIER2_SKILLS = {
    "nlp", "natural language processing",
    "llm", "large language model", "large language models",
    "fine-tuning", "fine tuning", "finetuning", "lora", "qlora", "peft",
    "transformer", "transformers", "bert", "gpt",
    "pytorch", "tensorflow", "keras",
    "rag", "retrieval augmented generation", "retrieval-augmented generation",
    "langchain", "llamaindex", "llama index",
    "huggingface", "hugging face",
    "text classification", "text mining", "named entity recognition", "ner",
    "sentiment analysis", "topic modeling",
    "xgboost", "lightgbm", "catboost",
    "feature engineering", "feature store",
}

# Tier 3 — Supporting / general engineering (weak signal alone)
TIER3_SKILLS = {
    "python", "sql", "java", "scala", "go",
    "machine learning", "ml", "deep learning", "dl",
    "data pipeline", "data pipelines", "etl", "data engineering",
    "spark", "pyspark", "kafka", "airflow", "apache beam",
    "docker", "kubernetes", "k8s",
    "aws", "gcp", "azure", "cloud",
    "mlflow", "weights & biases", "wandb", "mlops",
    "ci/cd", "git", "linux",
    "distributed systems", "microservices",
    "api", "rest api", "grpc", "fastapi", "flask",
    "statistical modeling", "statistics",
    "data science", "data analysis",
    "bentoml", "triton", "model serving",
}

# Combined set for quick lookup
ALL_JD_SKILLS = TIER1_SKILLS | TIER2_SKILLS | TIER3_SKILLS

# Tier weights for trust-weighted skill scoring
SKILL_TIER_WEIGHT = {
    "tier1": 3.0,
    "tier2": 2.0,
    "tier3": 1.0,
    "none": 0.0,   # skill not relevant to JD
}

# Proficiency weights
PROFICIENCY_WEIGHT = {
    "expert": 1.0,
    "advanced": 0.8,
    "intermediate": 0.5,
    "beginner": 0.2,
}

# ═══════════════════════════════════════════════════════════════════════════════
# 2. TITLE PATTERNS
# ═══════════════════════════════════════════════════════════════════════════════

# Tech / ML / AI / Data titles (regex patterns, case-insensitive)
TECH_TITLE_PATTERNS = [
    r"\b(ml|machine\s*learning)\s*(engineer|developer|scientist|lead|architect|specialist)\b",
    r"\b(ai|artificial\s*intelligence)\s*(engineer|developer|scientist|lead|architect|specialist)\b",
    r"\bdata\s*scientist\b",
    r"\bnlp\s*(engineer|scientist|developer|specialist|researcher)\b",
    r"\b(senior|staff|principal|lead)?\s*(software|backend|fullstack|full[\s-]?stack)\s*(engineer|developer)\b",
    r"\bdata\s*engineer\b",
    r"\bdata\s*analyst\b",
    r"\bresearch\s*(engineer|scientist)\b",
    r"\bplatform\s*engineer\b",
    r"\bdevops\s*engineer\b",
    r"\bsite\s*reliability\s*engineer\b",
    r"\bsoftware\s*(engineer|developer|architect)\b",
    r"\bbackend\s*(engineer|developer)\b",
    r"\bfrontend\s*(engineer|developer)\b",
    r"\bfull[\s-]?stack\s*(engineer|developer)\b",
    r"\bcloud\s*(engineer|architect)\b",
    r"\binfrastructure\s*engineer\b",
    r"\b(deep\s*learning|dl)\s*(engineer|scientist|researcher)\b",
    r"\bcomputer\s*vision\s*(engineer|scientist|researcher)\b",
    r"\bspeech\s*(engineer|scientist|researcher)\b",
    r"\brobotics\s*(engineer|scientist|researcher)\b",
    r"\bapplied\s*scientist\b",
    r"\bquantitative\s*(analyst|developer|researcher)\b",
    r"\bcto\b",
    r"\bvp\s*(of\s*)?(engineering|technology)\b",
    r"\btech\s*lead\b",
    r"\bengineering\s*(manager|lead|director)\b",
]

# Compiled for speed (done once at import)
TECH_TITLE_RE = [re.compile(p, re.IGNORECASE) for p in TECH_TITLE_PATTERNS]

# High-value ML/AI title patterns (subset for extra boosting)
ML_AI_TITLE_PATTERNS = [
    r"\b(ml|machine\s*learning)\s*(engineer|developer|scientist|lead|architect|specialist)\b",
    r"\b(ai|artificial\s*intelligence)\s*(engineer|developer|scientist|lead|architect|specialist)\b",
    r"\bdata\s*scientist\b",
    r"\bnlp\s*(engineer|scientist|developer|specialist|researcher)\b",
    r"\b(deep\s*learning|dl)\s*(engineer|scientist|researcher)\b",
    r"\bapplied\s*scientist\b",
]

ML_AI_TITLE_RE = [re.compile(p, re.IGNORECASE) for p in ML_AI_TITLE_PATTERNS]

# Non-tech titles (keyword-stuffer signals)
NON_TECH_TITLES = {
    "hr manager", "human resources manager", "hr executive", "hr specialist",
    "content writer", "copywriter", "technical writer",
    "marketing manager", "digital marketing", "marketing executive", "marketing specialist",
    "accountant", "accounting manager", "finance manager", "financial analyst",
    "sales executive", "sales manager", "business development",
    "graphic designer", "ui designer", "visual designer",
    "mechanical engineer", "civil engineer", "electrical engineer", "chemical engineer",
    "customer support", "customer service", "support executive", "support specialist",
    "operations manager", "operations executive", "operations analyst",
    "project manager", "program manager", "scrum master",
    "teacher", "professor", "lecturer",
    "lawyer", "legal", "advocate",
    "doctor", "physician", "nurse",
    "architect",  # building architect, not software
    "supply chain", "logistics",
    "quality analyst", "quality manager",  # non-software QA
    "recruiter", "talent acquisition",
    "admin", "executive assistant", "office manager",
}

# ═══════════════════════════════════════════════════════════════════════════════
# 3. CAREER EVIDENCE PATTERNS
# ═══════════════════════════════════════════════════════════════════════════════

# Patterns that indicate the candidate actually BUILT ranking/retrieval/recsys
CAREER_EVIDENCE_PATTERNS = [
    r"\b(built|shipped|designed|implemented|developed|created|architected|deployed|owned|led)\b"
    r"[^.]{0,80}"
    r"\b(ranking|recommendation|retrieval|search|matching|recsys|recommender|re-ranking|re ranking|"
    r"candidate[\s-]?scoring|relevance|information[\s-]?retrieval|query[\s-]?understanding|"
    r"vector[\s-]?search|hybrid[\s-]?search|embedding[\s-]?based|semantic[\s-]?search)\b",

    r"\b(ranking|recommendation|retrieval|search|matching|recsys|recommender)\s*(system|engine|pipeline|model|service|platform|infrastructure)\b",

    r"\b(embeddings?|vector[\s-]?db|vector[\s-]?database|faiss|pinecone|weaviate|qdrant|milvus|elasticsearch|opensearch)\b"
    r"[^.]{0,60}"
    r"\b(production|deployed|served|real[\s-]?users|scale|million|billion)\b",

    r"\b(ndcg|mrr|map|precision@|recall@|a/b\s*test|offline[\s-]?eval|online[\s-]?eval)\b",

    r"\bend[\s-]?to[\s-]?end\b[^.]{0,60}\b(ml|ai|machine\s*learning|model|pipeline)\b",
]

CAREER_EVIDENCE_RE = [re.compile(p, re.IGNORECASE) for p in CAREER_EVIDENCE_PATTERNS]

# Production / deployment language in career descriptions
PRODUCTION_PATTERNS = [
    r"\b(production|deployed|shipped|launched|served|real[\s-]?users|real[\s-]?traffic|live|scale|scaled|million|billion)\b",
    r"\b(end[\s-]?to[\s-]?end|full[\s-]?stack|infra|infrastructure|pipeline)\b",
    r"\b(monitoring|alerting|on[\s-]?call|sla|latency|throughput|reliability)\b",
]

PRODUCTION_RE = [re.compile(p, re.IGNORECASE) for p in PRODUCTION_PATTERNS]

# ═══════════════════════════════════════════════════════════════════════════════
# 4. COMPANY CLASSIFICATION
# ═══════════════════════════════════════════════════════════════════════════════

# Major consulting / services firms (JD explicitly flags these)
CONSULTING_FIRMS = {
    "tcs", "tata consultancy", "tata consultancy services",
    "infosys",
    "wipro",
    "accenture",
    "cognizant", "cognizant technology",
    "capgemini",
    "hcl", "hcl technologies",
    "mindtree",
    "tech mahindra",
    "lti", "l&t infotech", "ltimindtree",
    "mphasis",
    "hexaware",
    "persistent systems",
    "zensar",
    "cyient",
    "niit", "niit technologies",
    "virtusa",
    "birlasoft",
    "coforge",
    "sonata software",
    "mastek",
    "happiest minds",
    "kpit",
    "deloitte",
    "ey", "ernst & young", "ernst and young",
    "kpmg",
    "pwc", "pricewaterhousecoopers",
    "ibm", "ibm consulting",
    "dxc technology",
    "atos",
    "ntt data",
}

# Industries that suggest product companies
PRODUCT_INDUSTRIES = {
    "internet", "software", "saas", "technology", "fintech",
    "e-commerce", "ecommerce", "marketplace",
    "gaming", "media", "entertainment",
    "healthtech", "edtech", "agritech",
    "automotive", "electric vehicles",
    "telecommunications",
    "financial services", "banking",
    "cloud computing",
    "cybersecurity",
    "artificial intelligence",
}

# ═══════════════════════════════════════════════════════════════════════════════
# 5. LOCATION DATA
# ═══════════════════════════════════════════════════════════════════════════════

# Priority locations from JD
TIER1_LOCATIONS = {"pune", "noida"}
TIER2_LOCATIONS = {"hyderabad", "mumbai", "delhi", "delhi ncr", "new delhi",
                   "gurgaon", "gurugram", "bangalore", "bengaluru", "chennai",
                   "kolkata"}
INDIA_KEYWORDS = {"india"}

# ═══════════════════════════════════════════════════════════════════════════════
# 6. CV / SPEECH / ROBOTICS — Specialist Penalty
# ═══════════════════════════════════════════════════════════════════════════════

CV_SPEECH_ROBOTICS_SKILLS = {
    "computer vision", "image classification", "object detection",
    "image segmentation", "yolo", "opencv", "image processing",
    "speech recognition", "speech synthesis", "tts", "text to speech",
    "asr", "automatic speech recognition", "voice",
    "robotics", "ros", "robot operating system", "slam",
    "gans", "generative adversarial", "image generation",
    "3d reconstruction", "point cloud", "lidar",
}

NLP_IR_SKILLS = {
    "nlp", "natural language processing", "information retrieval",
    "text classification", "named entity recognition", "ner",
    "sentiment analysis", "topic modeling", "text mining",
    "retrieval", "search", "ranking", "recommendation",
    "embeddings", "embedding", "sentence-transformers",
    "llm", "large language model", "transformer", "transformers",
    "bert", "gpt", "rag", "langchain",
    "elasticsearch", "opensearch", "faiss", "pinecone",
}

# ═══════════════════════════════════════════════════════════════════════════════
# 7. SCORING WEIGHTS (the main tuning knobs)
# ═══════════════════════════════════════════════════════════════════════════════

WEIGHTS = {
    # ── Positive features (sum = 1.00) ───────────────────────────────────────
    # Option A structured signals + the Option B semantic signal, fused.
    "title_career_fit":        0.26,   # A: decisive anti-stuffer (title + career evidence)
    "semantic_fit":            0.22,   # B: dense (embedding) + sparse (BM25) JD similarity  <- NEW
    "skill_trust":             0.15,   # A: trust-weighted skill overlap
    "experience_fit":          0.12,   # A: 6-8 yr band fit
    "product_company":         0.09,   # A: product vs services
    "career_evidence_bonus":   0.08,   # A: built ranking/retrieval/recsys
    "location_fit":            0.05,   # A: Pune/Noida > metros > relocatable
    "education":               0.03,   # A: light tiebreaker

    # ── Penalties (subtracted; can drive a stuffer's fit_score to 0) ─────────
    "consulting_only":         0.15,
    "research_only":           0.10,
    "cv_speech_only":          0.10,
    "keyword_stuffer":         0.25,
    "job_hopper":              0.05,
}

# ═══════════════════════════════════════════════════════════════════════════════
# 7b. SEMANTIC FUSION (Option B) — how the embedding layer blends in
# ═══════════════════════════════════════════════════════════════════════════════
# semantic_fit = DENSE_WEIGHT * norm(cosine(JD, candidate)) + SPARSE_WEIGHT * norm(BM25)
# Both components are min-max normalized across the full pool at rank time. The
# blended value is ONE additive feature inside the structured score above — never
# the ranker itself. Structured penalties + the honeypot gate still dominate, so
# semantics cannot pull keyword-stuffers or honeypots into the top ranks.
SEMANTIC_DENSE_WEIGHT = 0.65    # embedding cosine similarity
SEMANTIC_SPARSE_WEIGHT = 0.35   # BM25 lexical overlap
# If artifacts are missing, rank.py falls back to pure Option A (semantic_fit = 0).

# ═══════════════════════════════════════════════════════════════════════════════
# 8. EXPERIENCE BAND SCORING
# ═══════════════════════════════════════════════════════════════════════════════

def experience_score(years: float) -> float:
    """
    Score experience fit for the JD (ideal: 6-8 yrs).
    Returns float in [0, 1].
    """
    if 6.0 <= years <= 8.0:
        return 1.0
    elif 5.0 <= years < 6.0 or 8.0 < years <= 9.0:
        return 0.85
    elif 4.0 <= years < 5.0 or 9.0 < years <= 12.0:
        return 0.60
    elif 3.0 <= years < 4.0 or 12.0 < years <= 15.0:
        return 0.35
    elif years < 3.0:
        return 0.15
    else:  # > 15
        return 0.20

# ═══════════════════════════════════════════════════════════════════════════════
# 9. OUTPUT CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════

TOP_K = 100  # Number of candidates to output
CSV_COLUMNS = ["candidate_id", "rank", "score", "reasoning"]
