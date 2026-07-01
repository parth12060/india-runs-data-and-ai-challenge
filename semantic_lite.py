"""
semantic_lite.py — Self-contained semantic layer (Option B), ZERO heavy deps.

This is the no-setup version of the Option B semantic signal. Instead of dense
neural embeddings (which need torch + sentence-transformers + a model download),
it computes a TF-IDF cosine similarity between the JD and each candidate's text,
in pure Python (stdlib only — no numpy, no network, no precompute, no artifacts).

It is a legitimate *sparse-semantic* relevance signal: candidates whose career
text overlaps the JD's vocabulary (retrieval, ranking, embeddings, recommendation,
production, eval, ...) score higher, weighted by term rarity (IDF). It captures
much of the Option B value while letting the whole pipeline run with a single
`python rank.py ...` command in VSCode.

If you want the higher-quality dense embeddings (your original Option B / BGE),
run precompute.py to build ./artifacts and rank.py will use those instead.

Two streaming passes over the candidates file:
    Pass 1 — document frequencies -> IDF
    Pass 2 — per-candidate TF-IDF cosine vs. the JD
Runtime ~ a few seconds beyond the base scan; memory is a single vocab dict.
"""

import json
import math
import re
from collections import Counter, defaultdict

from build_text import build_text

_TOKEN_RE = re.compile(r"[a-z0-9][a-z0-9+#.\-]*")


def _tokenize(text: str):
    return _TOKEN_RE.findall(text.lower())


def _stream(path: str, is_sample: bool):
    if is_sample:
        with open(path, "r", encoding="utf-8") as f:
            for c in json.load(f):
                yield c
    else:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    yield json.loads(line)
                except json.JSONDecodeError:
                    continue


def compute_semantic_scores(candidates_path: str, jd_text: str, is_sample: bool = False):
    """
    Return (scores, meta):
        scores : {candidate_id: semantic_fit in [0,1]}
        meta   : {'available': True, 'mode': 'tfidf', 'n': N}
    """
    # ── Pass 1: build_text ONCE per candidate; cache term counts + doc freq ──
    # We cache each candidate's non-zero term frequencies so we never recompute
    # build_text (the expensive step). IDF needs a full pass first, so caching is
    # what lets the whole thing stay a single build_text pass.
    df = defaultdict(int)
    cached = []          # list of (cid, {term: tf})
    n_docs = 0
    for c in _stream(candidates_path, is_sample):
        n_docs += 1
        tf = Counter(_tokenize(build_text(c)))
        cached.append((c.get("candidate_id"), tf))
        for t in tf:
            df[t] += 1

    if n_docs == 0:
        return {}, {"available": False, "mode": "tfidf", "n": 0}

    idf = {t: math.log((n_docs + 1) / (dfi + 1)) + 1.0 for t, dfi in df.items()}

    # ── JD vector (TF-IDF over known vocab) ──────────────────────────────────
    jd_tf = Counter(_tokenize(jd_text))
    jd_vec = {t: (1.0 + math.log(tf)) * idf.get(t, 0.0)
              for t, tf in jd_tf.items() if t in idf}
    jd_norm = math.sqrt(sum(w * w for w in jd_vec.values())) or 1.0

    # ── Pass 2 (in-memory over the cache): cosine(JD, candidate) ─────────────
    raw = {}
    for cid, tf in cached:
        norm_sq = 0.0
        dot = 0.0
        for t, f in tf.items():
            w = (1.0 + math.log(f)) * idf[t]
            norm_sq += w * w
            jw = jd_vec.get(t)
            if jw is not None:
                dot += w * jw
        cand_norm = math.sqrt(norm_sq) or 1.0
        raw[cid] = dot / (jd_norm * cand_norm)

    # ── Min-max normalize to [0, 1] across the pool ──────────────────────────
    vals = list(raw.values())
    mn, mx = min(vals), max(vals)
    rng = (mx - mn) or 1.0
    scores = {cid: (v - mn) / rng for cid, v in raw.items()}
    return scores, {"available": True, "mode": "tfidf", "n": n_docs}
