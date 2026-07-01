"""
semantic.py — Option B semantic layer, consumed as a feature by the combined ranker.

At RANK TIME this module does no ML and needs no network or torch. It only:
  1. Loads the precomputed artifacts (embeddings, ids, JD embedding, BM25 index)
     produced offline by precompute.py.
  2. Computes a dense cosine score (candidate embeddings . JD embedding) for the
     whole pool in a single matrix-vector product.
  3. Computes a sparse BM25 lexical score for the JD query over the whole pool.
  4. Min-max normalizes each across the pool and blends them into one
     `semantic_fit` value per candidate_id, in [0, 1].

The result is a dict {candidate_id: semantic_fit}. rank.py looks each candidate up
and adds WEIGHTS["semantic_fit"] * semantic_fit to the structured Option A score.

Design guarantees:
  - Reproducible: pure numpy, deterministic.
  - Fast: one matvec + one BM25 query over 100K docs — seconds on CPU.
  - Safe: if artifacts are missing/incomplete, load_semantic_scores() returns
    ({}, meta) and the ranker falls back to pure Option A (semantic_fit = 0).
"""

import os
import pickle

import numpy as np


def _minmax(arr: np.ndarray) -> np.ndarray:
    """Min-max normalize a 1-D array to [0, 1]. Flat array -> zeros."""
    mn = float(arr.min())
    mx = float(arr.max())
    if mx - mn < 1e-9:
        return np.zeros_like(arr, dtype=np.float32)
    return ((arr - mn) / (mx - mn)).astype(np.float32)


def load_semantic_scores(
    artifacts_dir: str,
    jd_text: str,
    dense_weight: float = 0.65,
    sparse_weight: float = 0.35,
) -> tuple[dict, dict]:
    """
    Load artifacts and compute a blended semantic score per candidate_id.

    Returns:
        (scores, meta)
        scores : {candidate_id (str): semantic_fit (float in [0,1])}
                 Empty dict if artifacts are unavailable -> pure Option A fallback.
        meta   : diagnostics (which components were used, counts, availability).
    """
    meta = {
        "available": False,
        "dense_used": False,
        "sparse_used": False,
        "n": 0,
        "reason": "",
    }

    emb_path = os.path.join(artifacts_dir, "candidate_embeddings.npy")
    ids_path = os.path.join(artifacts_dir, "candidate_ids.npy")
    jd_path = os.path.join(artifacts_dir, "jd_embedding.npy")
    bm25_path = os.path.join(artifacts_dir, "bm25_index.pkl")

    # Dense component requires embeddings + ids + JD embedding.
    if not (os.path.exists(emb_path) and os.path.exists(ids_path) and os.path.exists(jd_path)):
        meta["reason"] = f"embeddings/ids/jd not found in {artifacts_dir}"
        return {}, meta

    cand_emb = np.load(emb_path)                                   # (N, D) float32, L2-normalized
    cand_ids = np.load(ids_path, allow_pickle=True)               # (N,)
    jd_emb = np.load(jd_path).astype(np.float32).reshape(-1)      # (D,)

    n = len(cand_ids)
    meta["n"] = int(n)

    # ── Dense: cosine == dot product (both sides L2-normalized in precompute) ──
    dense = cand_emb @ jd_emb                                      # (N,)
    dense_norm = _minmax(dense)
    meta["dense_used"] = True

    # ── Sparse: BM25 over the JD query, if the index is present ───────────────
    sparse_norm = None
    if os.path.exists(bm25_path):
        try:
            with open(bm25_path, "rb") as f:
                bm25 = pickle.load(f)
            jd_tokens = jd_text.lower().split()
            bm25_scores = np.asarray(bm25.get_scores(jd_tokens), dtype=np.float32)
            if bm25_scores.shape[0] == n:
                sparse_norm = _minmax(bm25_scores)
                meta["sparse_used"] = True
        except Exception as e:  # noqa: BLE001 - never let sparse break dense
            meta["reason"] = f"bm25 skipped: {e}"

    # ── Blend ─────────────────────────────────────────────────────────────────
    if sparse_norm is not None:
        total = dense_weight + sparse_weight
        dw, sw = dense_weight / total, sparse_weight / total
        blended = dw * dense_norm + sw * sparse_norm
    else:
        blended = dense_norm  # dense-only if no BM25 index

    blended = _minmax(blended)  # re-normalize the blend to [0, 1]

    scores = {str(cid): float(s) for cid, s in zip(cand_ids, blended)}
    meta["available"] = True
    return scores, meta
