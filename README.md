# Surface Code Benchmark Simulator

An interactive benchmark environment for rotated surface-code memory experiments.

## Structure

- `apps/web`: React/Vite TypeScript interface.
- `services/api`: FastAPI simulator and benchmark API.

## Setup

```bash
npm install
python3 -m venv .venv
. .venv/bin/activate
pip install -r services/api/requirements.txt
```

## Run

In one terminal:

```bash
npm run api
```

In another:

```bash
npm run dev
```

The frontend defaults to `http://127.0.0.1:8000` for the API. Set
`VITE_API_BASE_URL` to override it.

## Notes

MWPM decoding uses Stim and PyMatching when installed. CNN, GNN, and Transformer
decoders expose the same evaluation surface but return a checkpoint-required
state until trained artifacts are supplied.

## Public benchmark criteria

The canonical public suite is `surface-code-memory-v1` version `2026.04`.
Leaderboard-eligible submissions must use distances `[3, 5, 7]`, memory-X basis,
`rounds = d`, seed `1337`, and include the MWPM baseline. Benchmark rows report
logical error rate, Wilson 95% confidence intervals, runtime per shot, peak
memory, environment metadata, and deterministic circuit/trace hashes.

Use `GET /api/benchmarks/{job_id}/submission` to export the JSON submission
bundle, then validate it with `POST /api/submissions/validate`.
See [docs/BENCHMARK_SUBMISSIONS.md](docs/BENCHMARK_SUBMISSIONS.md) for the
schema and eligibility rules.
