#!/usr/bin/env python3
"""
precompute.py — OFFLINE step (Option B). Runs once; NO time limit; network allowed.

Produces the artifacts that rank.py consumes at rank time. This is the ONLY step
that uses a heavy ML model (sentence-transformers). rank.py never imports torch.

Artifacts written to --out_dir (default ./artifacts):
    candidate_embeddings.npy   float32 (N, D)  L2-normalized candidate vectors
    candidate_ids.npy          str     (N,)    ID order aligned with embeddings
    jd_embedding.npy           float32 (1, D)  JD vector (query-prefixed for BGE)
    bm25_index.pkl             BM25Okapi        sparse lexical index over the corpus

Model choice:
    --model BAAI/bge-base-en-v1.5            (default; higher retrieval quality, 768-d)
    --model sentence-transformers/all-MiniLM-L6-v2   (fast, 384-d, small — good for CPU)

Usage:
    pip install sentence-transformers rank_bm25 tqdm numpy
    python precompute.py --candidates ./candidates.jsonl --jd ./job_description.txt

    # Faster/lighter model:
    python precompute.py --candidates ./candidates.jsonl --model sentence-transformers/all-MiniLM-L6-v2

Notes:
    - Candidate texts are NOT query-prefixed; only the JD (the "query") is, and
      only for BGE-family models (which require it).
    - normalize_embeddings=True so cosine == dot product in rank.py.
    - Handles .jsonl and .jsonl.gz.
"""

import argparse
import gzip
import json
import os
import pickle
import time

import numpy as np

from build_text import build_text

# BGE models require this instruction prefix on the QUERY side only.
BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "


def load_candidates(path, limit=None):
    opener = gzip.open if path.endswith(".gz") else open
    out = []
    with opener(path, "rt", encoding="utf-8") as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            out.append(json.loads(line))
            if limit and len(out) >= limit:
                break
    return out


def main():
    ap = argparse.ArgumentParser(description="Precompute embeddings + BM25 for the combined ranker")
    ap.add_argument("--candidates", default="./candidates.jsonl")
    ap.add_argument("--jd", default="./job_description.txt")
    ap.add_argument("--out_dir", default="./artifacts")
    ap.add_argument("--model", default="BAAI/bge-base-en-v1.5")
    ap.add_argument("--batch_size", type=int, default=256)
    ap.add_argument("--device", default=None, help="'cpu' or 'cuda' (auto-detect if omitted)")
    ap.add_argument("--limit", type=int, default=None, help="only first N candidates (smoke test)")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # Import heavy deps lazily so `python precompute.py --help` works without them.
    from sentence_transformers import SentenceTransformer
    try:
        from rank_bm25 import BM25Okapi
    except ImportError:
        BM25Okapi = None
        print("  [WARN] rank_bm25 not installed — skipping BM25 index (dense-only).")

    if args.device is None:
        try:
            import torch
            args.device = "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:
            args.device = "cpu"

    is_bge = "bge" in args.model.lower()
    print(f"Model : {args.model}  (device={args.device}, bge_prefix={is_bge})")
    model = SentenceTransformer(args.model, device=args.device)
    dim = model.get_sentence_embedding_dimension()
    print(f"Dim   : {dim}")

    # ── Load + build corpus ──────────────────────────────────────────────────
    t0 = time.time()
    candidates = load_candidates(args.candidates, limit=args.limit)
    print(f"Loaded {len(candidates):,} candidates in {time.time()-t0:.1f}s")

    texts, ids = [], []
    for c in candidates:
        try:
            texts.append(build_text(c))
            ids.append(c["candidate_id"])
        except Exception as e:  # noqa: BLE001
            print(f"  [WARN] build_text failed for {c.get('candidate_id','?')}: {e}")

    # ── BM25 sparse index ────────────────────────────────────────────────────
    if BM25Okapi is not None:
        t0 = time.time()
        tokenized = [t.lower().split() for t in texts]
        bm25 = BM25Okapi(tokenized)
        with open(os.path.join(args.out_dir, "bm25_index.pkl"), "wb") as f:
            pickle.dump(bm25, f)
        print(f"BM25 index built in {time.time()-t0:.1f}s")

    # ── Dense embeddings (candidates) ────────────────────────────────────────
    t0 = time.time()
    emb = model.encode(
        texts,
        batch_size=args.batch_size,
        normalize_embeddings=True,     # cosine == dot product downstream
        show_progress_bar=True,
        convert_to_numpy=True,
        device=args.device,
    ).astype(np.float32)
    print(f"Embedded {len(texts):,} candidates in {time.time()-t0:.1f}s -> {emb.shape}")

    # ── JD embedding (query side) ────────────────────────────────────────────
    with open(args.jd, "r", encoding="utf-8") as f:
        jd_text = f.read()
    jd_input = (BGE_QUERY_PREFIX + jd_text) if is_bge else jd_text
    jd_emb = model.encode(
        [jd_input], normalize_embeddings=True, convert_to_numpy=True, device=args.device
    ).astype(np.float32)

    # ── Save ─────────────────────────────────────────────────────────────────
    np.save(os.path.join(args.out_dir, "candidate_embeddings.npy"), emb)
    np.save(os.path.join(args.out_dir, "candidate_ids.npy"), np.array(ids))
    np.save(os.path.join(args.out_dir, "jd_embedding.npy"), jd_emb)

    print(f"\nArtifacts written to {args.out_dir}/")
    for name in ["candidate_embeddings.npy", "candidate_ids.npy", "jd_embedding.npy", "bm25_index.pkl"]:
        p = os.path.join(args.out_dir, name)
        if os.path.exists(p):
            print(f"  {name}: {os.path.getsize(p)/1024/1024:.1f} MB")
    print("\nDone. Copy artifacts/ next to rank.py, then run rank.py (no network needed).")


if __name__ == "__main__":
    main()
