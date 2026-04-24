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
