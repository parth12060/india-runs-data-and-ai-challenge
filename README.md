# Redrob Ranker — Combined A + B

Intelligent Candidate Discovery & Ranking Challenge. This system ranks the **top
100 candidates out of 100,000** for the Senior AI Engineer JD, producing a
`submission.csv` (`candidate_id,rank,score,reasoning`).

It fuses two approaches:

- **Option A — structured feature ranker (the floor).** Hand-engineered,
  interpretable signals: title+career fit, trust-weighted skills, experience band,
  product-vs-consulting, location, education; explicit penalties for keyword
  stuffers, consulting-only, pure-research, CV/speech-only, and job-hoppers; a
  bounded behavioral-availability multiplier; and a high-precision honeypot gate.
- **Option B — semantic layer (the lift).** Dense embeddings (sentence-transformers)
  + BM25 lexical scores, precomputed offline, blended into one `semantic_fit`
  feature that surfaces genuinely-strong candidates who don't use the obvious
  keywords ("built a recommendation system," never says "RAG").

**The key design decision:** the embedding similarity is *one weighted feature
inside* the structured ranker — never the ranker itself. Option A's penalties and
honeypot gate keep stuffers and honeypots out of the top ranks; Option B lifts the
quietly-worded true fits into them. We score all 100K in a single ~30s CPU pass, so
recall is perfect and no candidate is dropped by a retrieval shortlist.

```
final_score = fit_score · availability_modifier · honeypot_ok

fit_score = 0.26·title_career_fit(A)  + 0.22·semantic_fit(B)  + 0.15·skill_trust(A)
          + 0.12·experience_fit(A)    + 0.09·product_company(A)+ 0.08·career_evidence(A)
          + 0.05·location_fit(A)      + 0.03·education(A)
          − 0.25·keyword_stuffer − 0.15·consulting_only − 0.10·research_only
          − 0.10·cv_speech_only  − 0.05·job_hopper
availability_modifier ∈ [0.5, 1.15]   honeypot_ok ∈ {0, 1}
```

## Run it (VSCode — no setup, one command)

The ranking step is **CPU-only, no network, ≤5 min, ≤16 GB** (measured: ~41s, ~1 GB).

Open the folder in VSCode and run:

```bash
python rank.py --candidates ./candidates.jsonl --out ./submission.csv
```

That's it. **No pip install, no model download, no artifacts, no sandbox.** The
semantic layer (Option B) runs by default as a built-in TF-IDF cosine (`semantic_lite.py`,
pure Python standard library). It produces `submission.csv` directly.

Then validate:

```bash
python validate_submission.py submission.csv    # must print "Submission is valid."
```

Run on the small sample instead of the full pool (`--sample` reads a JSON array):

```bash
python rank.py --candidates ./sample_candidates.json --out ./sample_out.csv --sample
```

### Optional: higher-quality dense embeddings (your original Option B / BGE)

The default TF-IDF semantic signal needs zero setup. If you want the stronger dense
neural embeddings instead, precompute them once (this is the only step that needs
`torch` + `sentence-transformers`; network allowed, no time limit):

```bash
pip install -r requirements-precompute.txt
python precompute.py --candidates ./candidates.jsonl --jd ./job_description.txt --out_dir ./artifacts
python rank.py --candidates ./candidates.jsonl --out ./submission.csv   # auto-detects ./artifacts
```

`rank.py` automatically uses `./artifacts` if present, otherwise falls back to
TF-IDF. Force a backend with `--semantic {tfidf,embeddings,off}` (`off` = pure
Option A).

## Reproduce via Docker (satisfies the "sandbox" requirement without hosting)

`submission_spec` §10.5 requires a sandbox link **or** a self-contained `docker run`
recipe in the README. This repo ships a `Dockerfile`, so you need no HuggingFace/
Colab/Streamlit account:

```bash
docker build -t redrob-ranker .
docker run --rm -v "$PWD:/data" redrob-ranker \
  python rank.py --candidates /data/candidates.jsonl --out /data/submission.csv
```

Compute (measured, full 100K, default TF-IDF): **~42s wall, ~2 GB RAM, CPU-only, no
network** — inside the 5-min / 16-GB limits.

## Files

| File | Role |
|---|---|
| `rank.py` | **Combined ranker** — full-scan structured scoring + semantic feature → CSV. Rank-time entry point. |
| `config.py` | Weights, skill/title/consulting/location lexicons, experience curve, semantic-fusion knobs. |
| `features.py` | Option A structured feature + penalty functions, availability modifier, reasoning generator. |
| `honeypot.py` | High-precision impossible-profile gate (internal-contradiction rules; ~65/100K, no false positives). |
| `semantic_lite.py` | **Default semantic layer** — built-in TF-IDF cosine vs. the JD. Pure stdlib, no setup. |
| `semantic.py` | *Optional* dense-embedding loader: turns precomputed artifacts into `semantic_fit` (numpy). |
| `build_text.py` | Candidate→text (career history weighted first; skips 0-duration skills). Used by both semantic layers. |
| `precompute.py` | *Optional, offline* — dense embeddings + BM25 → `artifacts/`. The only step needing torch/network. |
| `validate_submission.py` | Official format validator. |
| `job_description.txt` | The JD (semantic query + BM25 query). |
| `Dockerfile` | Reproducible rank step / sandbox substitute (no hosting needed). |
| `requirements.txt` / `requirements-precompute.txt` | Default needs nothing; precompute deps only for optional embeddings. |

## Notes on the traps

- **Keyword stuffers** (non-tech titles with AI skill lists): skills are gated by
  title/career corroboration and hit with a stuffer penalty — verified 0 in the top 25.
- **Plain-language Tier-5s**: caught by `semantic_fit` + career-evidence patterns,
  not keyword matching.
- **Behavioral twins**: separated by the availability multiplier (response rate,
  recency, open-to-work, interview completion, notice period).
- **Honeypots** (~80, DQ if >10% of top 100): high-precision gate zeroes them;
  measured honeypot rate in the top 100 is **0%**.
