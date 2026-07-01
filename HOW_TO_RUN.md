# How to run — 2 steps

No installs, no internet, no sandbox. Just Python 3.8+ (VSCode's built-in terminal is fine).

## Step 1 — (already done) The data file is here
`candidates.jsonl` is already in this folder, next to `rank.py`, so there's nothing
to move. (If you ever run this elsewhere, copy `candidates.jsonl` in first.)

## Step 2 — Run the ranker (one command)
Open this folder in VSCode → Terminal → run:

```bash
python rank.py --candidates candidates.jsonl --out submission.csv
```

Takes ~40 seconds. Prints progress and self-checks, and writes `submission.csv`
(the top-100 ranking). No `pip install` needed — the semantic layer is built in.

## Step 3 — Validate, then rename for upload
```bash
python validate_submission.py submission.csv
```
Must print **"Submission is valid."** Then rename the file to your registered
participant ID before uploading, e.g. `team_1234.csv`.

---

## If `python` doesn't work, try `python3`
```bash
python3 rank.py --candidates candidates.jsonl --out submission.csv
```

## Don't want to move the data file?
Point `--candidates` at wherever `candidates.jsonl` already is:
```bash
python rank.py --candidates "/full/path/to/candidates.jsonl" --out submission.csv
```

## Test quickly on the small sample (50 candidates, no data file needed)
```bash
python rank.py --candidates sample_candidates.json --out sample_out.csv --sample
```

## Optional — higher-quality dense embeddings (your original Option B / BGE)
Only if you want it; the default already produces a valid, strong submission.
```bash
pip install -r requirements-precompute.txt
python precompute.py --candidates candidates.jsonl --jd job_description.txt --out_dir artifacts
python rank.py --candidates candidates.jsonl --out submission.csv   # auto-uses ./artifacts
```

## Optional — run in Docker (no Python setup at all)
```bash
docker build -t redrob-ranker .
docker run --rm -v "$PWD:/data" redrob-ranker \
  python rank.py --candidates /data/candidates.jsonl --out /data/submission.csv
```

---

### What each file does
- `rank.py` — main program (run this). Fuses the structured ranker + semantic layer.
- `config.py`, `features.py`, `honeypot.py` — Option A structured scoring, penalties, honeypot gate.
- `semantic_lite.py` — built-in TF-IDF semantic layer (default; no dependencies).
- `build_text.py`, `semantic.py`, `precompute.py` — optional dense-embedding path (Option B).
- `validate_submission.py` — official format checker.
- `job_description.txt`, `sample_candidates.json` — the JD and a 50-row sample.
- `submission_metadata.yaml` — fill in team name / GitHub before submitting.
- `README.md` — full architecture write-up. `Dockerfile` — container recipe.
