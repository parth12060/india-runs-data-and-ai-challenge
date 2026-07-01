#!/usr/bin/env python3
"""
rank.py — Combined A+B ranker for the Redrob Hackathon.

ARCHITECTURE  (Option A is the floor, Option B is a feature on top)
    final_score = fit_score * availability_modifier * honeypot_ok

    fit_score  = W.title_career_fit    * title_career_fit(A)
               + W.semantic_fit        * semantic_fit(B)          <-- embeddings + BM25
               + W.skill_trust         * skill_trust(A)
               + W.experience_fit      * experience_fit(A)
               + W.product_company     * product_company(A)
               + W.career_evidence     * career_evidence(A)
               + W.location_fit        * location_fit(A)
               + W.education           * education(A)
               - W.keyword_stuffer     * keyword_stuffer_penalty(A)
               - W.consulting_only     * consulting_only_penalty(A)
               - W.research_only       * research_only_penalty(A)
               - W.cv_speech_only      * cv_speech_penalty(A)
               - W.job_hopper          * job_hopper_penalty(A)

    availability_modifier ∈ [0.5, 1.15]   (behavioral signals, bounded multiplier)
    honeypot_ok ∈ {0, 1}                   (high-precision impossible-profile gate)

WHY THIS SHAPE
    - The structured layer (A) decides ORDER at the top of the list, where 80% of
      the score lives (NDCG@10 + NDCG@50). Its penalties + honeypot gate keep
      keyword-stuffers and honeypots out of the top ranks.
    - The semantic layer (B) is an additive feature that lifts genuinely-strong
      but quietly-worded candidates ("built a recommendation system", never says
      "RAG") that a keyword-only system would miss.
    - We score ALL 100K (a full scan is only ~30s on CPU), so recall is perfect
      and A truly is the floor — no candidate is dropped by a retrieval shortlist.

COMPUTE (rank step): <=5 min, <=16 GB, CPU-only, NO network, NO torch.
    Embeddings are precomputed offline by precompute.py into ./artifacts.
    If ./artifacts is absent, the ranker runs as PURE OPTION A (semantic_fit = 0).

USAGE
    # Combined (needs precomputed artifacts):
    python rank.py --candidates ./candidates.jsonl --artifacts ./artifacts --out ./submission.csv

    # Pure Option A fallback (no artifacts needed):
    python rank.py --candidates ./candidates.jsonl --out ./submission.csv

    # Small sample (JSON array like sample_candidates.json):
    python rank.py --candidates ./sample_candidates.json --out ./out.csv --sample
"""

import argparse
import csv
import heapq
import json
import os
import sys
import time

from config import WEIGHTS, TOP_K, CSV_COLUMNS, SEMANTIC_DENSE_WEIGHT, SEMANTIC_SPARSE_WEIGHT
from features import (
    title_career_fit,
    skill_trust_score,
    experience_fit,
    product_company_signal,
    location_fit,
    education_signal,
    keyword_stuffer_penalty,
    consulting_only_penalty,
    research_only_penalty,
    cv_speech_penalty,
    job_hopper_penalty,
    availability_modifier,
    generate_reasoning,
)
from honeypot import is_honeypot
# NOTE: the semantic backends (semantic.py -> numpy; semantic_lite.py -> stdlib)
# are imported lazily inside run_ranking so the default TF-IDF / pure-A paths need
# no third-party packages installed.


def compute_score(candidate: dict, semantic_fit: float) -> tuple[float, dict]:
    """Combined score for one candidate. semantic_fit in [0,1] (0 if no artifacts)."""
    W = WEIGHTS

    # ── Structured positive features (Option A) ──────────────────────────────
    tcf, title_detail = title_career_fit(candidate)
    is_tech = title_detail.get("is_tech", False)
    sts, skill_detail = skill_trust_score(candidate, is_tech)
    exf, _ = experience_fit(candidate)
    pcs, _ = product_company_signal(candidate)
    lof, loc_detail = location_fit(candidate)
    edu, _ = education_signal(candidate)
    career_ev = title_detail.get("career_evidence", 0)

    fit_score = (
        W["title_career_fit"]      * tcf +
        W["semantic_fit"]          * semantic_fit +          # <-- Option B, fused in
        W["skill_trust"]           * sts +
        W["experience_fit"]        * exf +
        W["product_company"]       * pcs +
        W["career_evidence_bonus"] * career_ev +
        W["location_fit"]          * lof +
        W["education"]             * edu
    )

    # ── Penalties (Option A) ─────────────────────────────────────────────────
    stf_pen = keyword_stuffer_penalty(candidate, title_detail)
    con_pen = consulting_only_penalty(candidate)
    res_pen = research_only_penalty(candidate)
    cvs_pen = cv_speech_penalty(candidate)
    jh_pen = job_hopper_penalty(candidate)

    penalty = (
        W["keyword_stuffer"] * stf_pen +
        W["consulting_only"] * con_pen +
        W["research_only"]   * res_pen +
        W["cv_speech_only"]  * cvs_pen +
        W["job_hopper"]      * jh_pen
    )
    fit_score = max(fit_score - penalty, 0.0)

    # ── Behavioral availability modifier (bounded multiplier) ────────────────
    avail_mod, avail_detail = availability_modifier(candidate)

    # ── Honeypot hard gate (high-precision) ──────────────────────────────────
    hp_flag, hp_reason = is_honeypot(candidate)
    honeypot_ok = 0.0 if hp_flag else 1.0

    final_score = fit_score * avail_mod * honeypot_ok

    features = {
        "title_career_detail": title_detail,
        "skill_detail": skill_detail,
        "location_detail": loc_detail,
        "avail_detail": avail_detail,
        "career_evidence": round(career_ev, 4),
        "semantic_fit": round(semantic_fit, 4),
        "stuffer_penalty": round(stf_pen, 4),
        "consulting_penalty": round(con_pen, 4),
        "honeypot": hp_flag,
        "honeypot_reason": hp_reason,
        "fit_score": round(fit_score, 4),
        "avail_mod": round(avail_mod, 4),
        "final_score": round(final_score, 6),
    }
    return final_score, features


def _augment_reasoning(base: str, features: dict) -> str:
    """Append a short, honest semantic note (no hallucination)."""
    sem = features.get("semantic_fit", 0.0)
    if sem >= 0.80:
        note = "strong semantic match to JD"
    elif sem >= 0.60:
        note = "good semantic match to JD"
    else:
        return base
    # Insert the note before the trailing period.
    base = base.rstrip()
    if base.endswith("."):
        base = base[:-1]
    return f"{base}; {note}."


def load_jsonl_stream(filepath: str):
    with open(filepath, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as e:
                print(f"  [WARN] Skipping malformed line {line_num}: {e}", file=sys.stderr)


def load_json_sample(filepath: str):
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)


def run_ranking(candidates_path, output_path, artifacts_dir, jd_path,
                is_sample=False, semantic_mode="auto"):
    print("=" * 72)
    print("  REDROB HACKATHON — Combined Ranker (A structured + B semantic)")
    print("=" * 72)

    start = time.time()

    # JD text (used by both semantic backends as the query).
    jd_text = ""
    if jd_path and os.path.exists(jd_path):
        with open(jd_path, "r", encoding="utf-8") as f:
            jd_text = f.read()

    # ── Resolve which semantic backend to use ────────────────────────────────
    #   auto  -> dense embeddings if ./artifacts exists, else built-in TF-IDF
    #   tfidf -> built-in TF-IDF (pure stdlib; no setup)
    #   embeddings -> precomputed BGE/MiniLM artifacts (needs numpy + precompute.py)
    #   off   -> pure Option A (semantic_fit = 0)
    has_artifacts = bool(artifacts_dir) and os.path.exists(
        os.path.join(artifacts_dir, "candidate_embeddings.npy")
    )
    resolved = semantic_mode
    if resolved == "auto":
        resolved = "embeddings" if has_artifacts else "tfidf"

    semantic_scores = {}
    if resolved == "embeddings":
        from semantic import load_semantic_scores  # lazy: needs numpy
        semantic_scores, meta = load_semantic_scores(
            artifacts_dir, jd_text,
            dense_weight=SEMANTIC_DENSE_WEIGHT, sparse_weight=SEMANTIC_SPARSE_WEIGHT,
        )
        if meta["available"]:
            comps = [c for c, u in (("dense/embeddings", meta["dense_used"]),
                                    ("sparse/BM25", meta["sparse_used"])) if u]
            print(f"  Semantic layer: ON — dense embeddings ({' + '.join(comps)}; {meta['n']:,} candidates)")
        else:
            print(f"  Semantic layer: embeddings unavailable ({meta['reason']}); using built-in TF-IDF.")
            resolved = "tfidf"

    if resolved == "tfidf":
        from semantic_lite import compute_semantic_scores  # lazy: stdlib only
        semantic_scores, meta = compute_semantic_scores(candidates_path, jd_text, is_sample=is_sample)
        print(f"  Semantic layer: ON — built-in TF-IDF ({meta['n']:,} candidates; no setup required)")
    elif resolved == "off":
        print("  Semantic layer: OFF (--semantic off) -> PURE OPTION A.")

    print(f"  Input : {candidates_path}")
    print(f"  Output: {output_path}")
    print()

    # ── Score every candidate; keep a top-K min-heap ─────────────────────────
    iterator = load_json_sample(candidates_path) if is_sample else load_jsonl_stream(candidates_path)
    heap = []  # (score, cid, candidate, features)
    total = 0
    honeypots = 0

    for candidate in iterator:
        total += 1
        if total % 20000 == 0:
            print(f"  Scored {total:,} in {time.time()-start:.1f}s...")
        cid = candidate.get("candidate_id", f"UNKNOWN_{total}")
        sem = semantic_scores.get(cid, 0.0)
        score, features = compute_score(candidate, sem)
        if features["honeypot"]:
            honeypots += 1
        if len(heap) < TOP_K:
            heapq.heappush(heap, (score, cid, candidate, features))
        elif score > heap[0][0]:
            heapq.heapreplace(heap, (score, cid, candidate, features))

    # ── Sort: score desc, then candidate_id asc (tie-break per spec) ─────────
    top = sorted(heap, key=lambda x: (-round(x[0], 4), x[1]))

    rows = []
    for rank, (score, cid, candidate, features) in enumerate(top, 1):
        reasoning = _augment_reasoning(generate_reasoning(candidate, features), features)
        rows.append({
            "candidate_id": cid,
            "rank": rank,
            "score": f"{score:.4f}",
            "reasoning": reasoning,
        })

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)

    elapsed = time.time() - start

    # ── Diagnostics ──────────────────────────────────────────────────────────
    print(f"\n  Scored {total:,} candidates in {elapsed:.1f}s  (limit 300s)")
    print(f"  Honeypots detected in pool: {honeypots}")
    if rows:
        print(f"  Score range: {rows[0]['score']} (rank 1) -> {rows[-1]['score']} (rank {len(rows)})")

    hp_top = sum(1 for _, _, _, ft in top if ft["honeypot"])
    hp_rate = 100.0 * hp_top / max(len(top), 1)
    print(f"  Honeypot rate in top {len(top)}: {hp_top} ({hp_rate:.1f}%)  "
          f"{'[OK]' if hp_rate <= 10 else '[DQ RISK!]'}")

    # Keyword-stuffer sanity: non-tech titles in the top 25 should be ~none.
    stuffer_top = sum(
        1 for _, _, _, ft in top[:25]
        if ft["stuffer_penalty"] >= 0.5
    )
    print(f"  Keyword-stuffer flags in top 25: {stuffer_top}  {'[OK]' if stuffer_top == 0 else '[CHECK]'}")

    print("\n  TOP 10:")
    for row, (_, _, cand, _) in zip(rows[:10], top[:10]):
        p = cand.get("profile", {})
        print(f"   {row['rank']:>3}. {row['candidate_id']} s={row['score']} "
              f"{p.get('current_title','?')} @ {p.get('current_company','?')} "
              f"({p.get('years_of_experience',0)}y)")

    # Format self-checks (mirror the official validator)
    scores = [float(r["score"]) for r in rows]
    checks = {
        "exactly 100 rows": len(rows) == TOP_K or is_sample,
        "scores non-increasing": all(scores[i] >= scores[i+1] for i in range(len(scores)-1)),
        "unique ranks": len({r["rank"] for r in rows}) == len(rows),
        "unique candidate_ids": len({r["candidate_id"] for r in rows}) == len(rows),
        "scores differentiated": len(set(scores)) > 1,
        "no empty reasoning": all(r["reasoning"].strip() for r in rows),
        "within time budget": elapsed <= 300,
    }
    print("\n  SELF-CHECKS:")
    for k, v in checks.items():
        print(f"   [{'OK' if v else 'FAIL'}] {k}")
    print("=" * 72)

    if not all(checks.values()):
        print("  [WARNING] One or more self-checks failed — inspect before submitting.")


def main():
    ap = argparse.ArgumentParser(description="Redrob combined A+B ranker")
    ap.add_argument("--candidates", required=True, help="candidates.jsonl (or JSON array with --sample)")
    ap.add_argument("--out", default="submission.csv", help="output CSV path")
    ap.add_argument("--artifacts", default="./artifacts",
                    help="precomputed dense-embedding dir (used automatically if present)")
    ap.add_argument("--jd", default="./job_description.txt", help="JD text (semantic + BM25 query)")
    ap.add_argument("--semantic", choices=["auto", "tfidf", "embeddings", "off"], default="auto",
                    help="semantic backend: auto (embeddings if ./artifacts else built-in TF-IDF), "
                         "tfidf (pure stdlib, no setup), embeddings (needs precompute.py), or off (pure A)")
    ap.add_argument("--sample", action="store_true", help="input is a JSON array, not JSONL")
    args = ap.parse_args()

    if not os.path.exists(args.candidates):
        print(f"ERROR: candidates file not found: {args.candidates}", file=sys.stderr)
        sys.exit(1)

    run_ranking(args.candidates, args.out, args.artifacts, args.jd,
                is_sample=args.sample, semantic_mode=args.semantic)


if __name__ == "__main__":
    main()
