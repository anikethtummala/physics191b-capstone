# Benchmark Submissions

The public suite is `surface-code-memory-v1` version `2026.04`. A submission is a
single JSON bundle exported from `GET /api/benchmarks/{job_id}/submission`.

## Eligibility

A bundle is leaderboard-eligible when validation passes and:

- `suite_id` is `surface-code-memory-v1`.
- `suite_version` is `2026.04`.
- `request.distances` is `[3, 5, 7]`.
- `request.rounds` is `null`, meaning `rounds = d` for each case.
- `request.basis` is `x`.
- `request.seed` is `1337`.
- `request.decoders` includes `mwpm`.
- Complete result rows include logical error rate, Wilson 95% confidence
  intervals, runtime, memory, sample seed, case ID, trace ID, and SHA-256 hashes.
- Runtime disclosure includes Python/platform/CPU metadata and dependency
  versions for Stim, PyMatching, NumPy, FastAPI, and Pydantic.

Noise settings and shot count are recorded in the bundle. They are not fixed by
the suite so the same format can publish separate below-threshold and stress-test
tables.

## Bundle Shape

```json
{
  "schema_version": "1.0",
  "suite_id": "surface-code-memory-v1",
  "suite_version": "2026.04",
  "suite_compliant": true,
  "suite_compliance_errors": [],
  "job_id": "uuid",
  "request": {
    "suite_id": "surface-code-memory-v1",
    "suite_version": "2026.04",
    "distances": [3, 5, 7],
    "rounds": null,
    "basis": "x",
    "noise": { "p": 0.001 },
    "shots": 1000,
    "seed": 1337,
    "decoders": ["mwpm"]
  },
  "runtime_environment": {
    "api_version": "0.1.0",
    "python": "3.13.0",
    "platform": "Darwin",
    "platform_release": "25.0.0",
    "machine": "arm64",
    "processor": "arm",
    "cpu_count": 12,
    "dependencies": {
      "stim": "1.13.0",
      "pymatching": "2.1.0",
      "numpy": "2.0.0",
      "fastapi": "0.128.0",
      "pydantic": "2.0.0"
    }
  },
  "results": [],
  "traces": []
}
```

`traces` stores the circuit text plus base64-encoded detection-event and
observable-flip arrays. Validators recompute the circuit, array, and trace hashes
from those payloads, so result rows cannot silently drift from the sampled trace.

## Validate

```bash
curl -X POST http://127.0.0.1:8000/api/submissions/validate \
  -H "Content-Type: application/json" \
  --data-binary @submission.json
```

The validator returns:

```json
{
  "valid": true,
  "leaderboard_eligible": true,
  "warnings": [],
  "errors": []
}
```
