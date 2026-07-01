# Dockerfile — reproducible rank step (Stage-3 reproduction / sandbox substitute).
#
# submission_spec Section 10.5 allows a self-contained `docker run` recipe in the
# README in place of a hosted sandbox. This image builds and runs unmodified.
#
# The DEFAULT ranker needs no third-party packages (built-in TF-IDF semantic layer),
# so this image is just Python + the code. CPU-only, no network at run time.
#
# Build:
#   docker build -t redrob-ranker .
#
# Run (mount a candidates file into /data and read the CSV back out):
#   docker run --rm -v "$PWD:/data" redrob-ranker \
#     python rank.py --candidates /data/candidates.jsonl --out /data/submission.csv
#
# Small-sample demo (uses the bundled sample_candidates.json):
#   docker run --rm -v "$PWD:/data" redrob-ranker \
#     python rank.py --candidates sample_candidates.json --out /data/sample_out.csv --sample

FROM python:3.11-slim

WORKDIR /app
COPY . /app

# No pip install needed for the default (stdlib-only) path. If you switch to the
# optional dense-embedding backend, also run: pip install -r requirements-precompute.txt
# and precompute artifacts before running (see README).

# Default command prints usage; override with your own `python rank.py ...`.
CMD ["python", "rank.py", "--help"]
